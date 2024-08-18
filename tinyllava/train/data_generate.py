import argparse
import os
import json
import random
import math
from collections import defaultdict
from itertools import combinations

import torch
import deepspeed
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import datasets
import torch.distributed
from PIL import Image
import Levenshtein

from tinyllava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from tinyllava.conversation import conv_templates
from tinyllava.model.builder import load_pretrained_model
from tinyllava.utils import disable_torch_init
from tinyllava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from vcd_add_noise import add_diffusion_noise


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks."""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def prompt_format(prompt, img_path):
    return f"<img>{img_path}</img>{prompt.strip()}"


def filter_dataset(ds, local_rank):
    """Filter the dataset based on the length of the response."""
    if local_rank > 0 and torch.distributed.is_initialized():
        torch.distributed.barrier()

    ds = ds.filter(lambda example: all(len(comp["response"].split()) <= 350 for comp in example["completions"]))

    if local_rank == 0 and torch.distributed.is_initialized():
        torch.distributed.barrier()

    return ds


def make_batch_pairs(sample):
    converted_sample = defaultdict(list)
    for prompt, image, completions in tqdm(zip(sample["prompt"], sample["image"], sample["completions"]), desc="Creating pairs"):
        chosen = completions[0]["response"]
        converted_sample["image"].append(image)
        converted_sample["prompt"].append(prompt)
        converted_sample["chosen"].append(chosen)
        converted_sample["completions"].append(completions)
    return converted_sample


def prepare_dataset(ds, local_rank):
    """Prepare the dataset for processing."""
    ds = filter_dataset(ds, local_rank)
    ds = ds.map(
        make_batch_pairs,
        batched=True,
        remove_columns=set(ds.column_names) - set(["prompt", "chosen", "image", "completions"]),
    )
    return ds


def sample_data(dataset, sample_percentage):
    """Sample a percentage of the dataset."""
    if 'id' not in dataset.column_names:
        dataset = dataset.add_column('id', list(range(dataset.num_rows)))

    ids = [id for id in dataset['id']]
    sampled_indices = random.sample(ids, int(sample_percentage * len(ids)))
    return dataset.select(sampled_indices)


def generate_completions(batch, model, tokenizer, image_processor, args):
    prompts, images, chosens, o_completions = [], [], [], []
    for prompt, image, chosen, completions in zip(batch["prompt"], batch["image"], batch["chosen"], batch["completions"]):
        prompts.append(prompt_format(prompt, image))
        images.append(image)
        chosens.append(chosen)
        o_completions.append(completions)

    input_ids = pad_sequence(
        [tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda() for prompt in prompts],
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    )

    image_tensors = torch.stack([
        add_diffusion_noise(process_images([Image.open(os.path.join(args.image_folder, image)).convert('RGB')], image_processor, model.config)[0], noise).half().cuda()
        for image in images for noise in [950, 850, 750]
    ])

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensors,
            do_sample=args.temperature > 0,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=512,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return outputs, chosens, o_completions


def process_data(loader, model, tokenizer, image_processor, args):
    results = defaultdict(list)
    conv = conv_templates[args.conv_mode].copy()

    for batch in tqdm(loader, desc="Processing batches"):
        outputs, chosens, o_completions = generate_completions(batch, model, tokenizer, image_processor, args)

        for output, chosen, completions in zip(outputs, chosens, o_completions):
            completions.append({"response": output.strip(), "score": Levenshtein.ratio(output.strip(), chosen), "type": "generated"})

            results["image"].append(batch["image"])
            results["prompt"].append(batch["prompt"])
            results["completions"].append(completions)

    return results


def work(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    model = torch.compile(model, mode="max-autotune", fullgraph=True)

    dataset = datasets.load_dataset('json', data_files=args.question_file, split='train')
    dataset = prepare_dataset(dataset, -1)

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    processed_data = process_data(loader, model, tokenizer, image_processor, args)

    # Save the processed data
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    dataset = datasets.Dataset.from_pandas(pd.DataFrame(processed_data))
    dataset.to_json(answers_file, force_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)

    args = parser.parse_args()
    work(args)
