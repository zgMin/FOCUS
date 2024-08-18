# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import copy
import re
import time
from dataclasses import dataclass, field
import json
import logging
from packaging import version
import pathlib
from typing import Dict, Optional, Sequence, List

import torch
import tokenizers
import transformers
from torch.utils.data import Dataset
from peft import prepare_model_for_kbit_training


from tinyllava.train.llava_trainer import LLaVATrainer
from tinyllava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, \
    DEFAULT_IM_END_TOKEN
from tinyllava import conversation as conversation_lib
from tinyllava.model import *
from tinyllava.mm_utils import tokenizer_image_token
from tinyllava.train.train_utils import *
from tinyllava.utils import rank0_print, local_rank
from tinyllava.data.dataset import make_supervised_data_module
from tinyllava.model.model_factory import *
from tinyllava.arguments import *

import datasets
import torch.distributed
from trl.trainer import DPOTrainer
from trl.trainer.utils import DPODataCollatorWithPadding
from PIL import Image
from collections import defaultdict
from itertools import combinations
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v1")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    mm_patch_merge_type: Optional[str] = field(default='flat')
    resampler_hidden_size: Optional[int] = field(default=768)
    num_queries: Optional[int] = field(default=128)
    num_resampler_layers: Optional[int] = field(default=3)
    tune_vision_tower: bool = field(default=False)
    tune_entire_model: bool = field(default=False)
    tune_vit_from_layer: Optional[int] = field(default=100)
    tune_embed_tokens: Optional[int] = field(default=False)

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    dataset_size: int = field(
        default=-1,
    )
    filter_size: int = field(
        default=350,
    )
    test_size: float = 0.05
    subset_percent: float = -1

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    beta: float = field(default=0.1)
    generate_during_eval: bool = field(default=False)
    vision_tower_lr: Optional[float] = None

def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len

def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    # conversation += BEGIN_SIGNAL
    return conversation

def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources
    for source in sources:
        for sentence in source:
            pattern = r"<img>.*?<\/img>"
            # Remove the matched pattern from the string
            sentence['value'] = re.sub(pattern, DEFAULT_IMAGE_TOKEN + '\n', sentence['value'])
            sentence['value'] = sentence['value'].strip()
            if "mmtag" in conversation_lib.default_conversation.version:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)
            if len(sentence['value']) == 0:
                sentence['value'] = " "

    return sources

def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                # print(
                #     f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                #     f" (ignored)"
                # )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    # sources,
    source,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    # for i, source in enumerate(sources):
    if roles[source[0]["from"]] != conv.roles[0]:
        # Skip the first one if it is not from human
        source = source[1:]

    conv.messages = []
    for j, sentence in enumerate(source):
        role = roles[sentence["from"]]
        assert role == conv.roles[j % 2], f"{i}"
        conv.append_message(role, sentence["value"])

    # Tokenize conversations

    input_ids = tokenizer_image_token(conv.get_prompt(), tokenizer, return_tensors='pt')

    targets = [input_ids.clone()]
    instructions = []
    conversations = [conv.get_prompt()]
    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO
    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_ids = tokenizer_image_token(parts[0], tokenizer)
                instruction_len = len(instruction_ids) - 2
                instructions.append(instruction_ids)
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_ids = tokenizer(parts[0]).input_ids
                instructions.append(instruction_ids)
                instruction_len = len(instruction_ids.input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        # target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len <= total_len:
                target[:] = IGNORE_INDEX
                # print(
                #     f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                #     f" (ignored)"
                # )
    input_dict = dict(
        input_ids=input_ids,
        labels=targets[0],
        attention_mask = torch.ones_like(input_ids)
    )
    instruction_dict = dict(
        input_ids=instructions[0],
        labels=instructions[0],
        attention_mask = [1] * len(instructions[0])
    )
    return input_dict, instruction_dict


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)

    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    prompts = []
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)
        targets.append(source[:tokenized_lens[0]])

    return dict(prompt_id = prompts, input_ids=input_ids, labels=targets)

def make_conv(prompt, answer):
    return [
        {
            "from": "human",
            "value": prompt,
        },
        {
            "from": "gpt",
            "value": answer,
        },
    ]

@dataclass
class LLaVADPODataCollator(DPODataCollatorWithPadding):
    def __init__(self, data_args, *args, **kwargs):
        super(LLaVADPODataCollator, self).__init__(*args, **kwargs)
        self.data_args = data_args

    def tokenize_batch_element(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
    ) -> Dict:
        """Tokenize a single batch element.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
            in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the chosen/rejected responses, which are of length equal to
            the sum of the length of the prompt and the chosen/rejected response, with
            label_pad_token_id  for the prompt tokens.
        """
        batch = {}
        chosen_conv = make_conv(prompt, chosen)
        rejected_conv = make_conv(prompt, rejected)
        # processing image
        if "img" in prompt:
            # Define the regular expression pattern
            pattern = r"<img>(.*?)<\/img>"

            # Find all matches of the pattern in the string
            matches = re.findall(pattern, prompt)
            image_file = matches[0]
            processor = self.data_args.image_processor
            # print(self.data_args.image_folder,image_file)
            image = Image.open(os.path.join(self.data_args.image_folder, image_file)).convert('RGB')
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            # replacing img tokens
            chosen_conv,  rejected_conv = preprocess_multimodal([copy.deepcopy(chosen_conv), copy.deepcopy(rejected_conv)], data_args=self.data_args)
            # self.data_args None)
        # tokenize
        chosen_conv_dict, prompt_dict = preprocess_v1(
            chosen_conv,
            self.tokenizer,
            has_image=True)
        rejected_conv_dict, _ = preprocess_v1(
            rejected_conv,
            self.tokenizer,
            has_image=True)

        for k, toks in {
            "chosen": chosen_conv_dict,
            "rejected": rejected_conv_dict,
            "prompt": prompt_dict,
        }.items():
            for type_key, tokens in toks.items():
                if type_key == "token_type_ids":
                    continue
                batch[f"{k}_{type_key}"] = tokens
        batch["images"] = image

        return batch

    def collate(self, batch):
        # first, pad everything to the same length
        padded_batch = super(LLaVADPODataCollator, self).collate(batch)
        images = torch.stack([b["images"] for b in batch])
        padded_batch.update({"images": images})
        return padded_batch

def prompt_format(prompt, img_path):
    out = []
    out.append(f"<img>{img_path}</img>")
    out.append(prompt.strip())
    return "".join(out)


def bpo_paired_dataset(ds, local_rank, data_args):
    def set_format(sample):
        prompt = sample["prompt"]
        img_path = sample["image"]
        sample["prompt"] = prompt_format(prompt, img_path)
        return sample

    ds = ds.map(set_format)
    # format prompt
    if local_rank > 0 and torch.distributed.is_initialized():
        torch.distributed.barrier()

    print(f"original length {len(ds)}")
    ds = ds.filter(lambda example: all(len(comp["response"].split()) <= data_args.filter_size for comp in example["completions"]))
    print(f"filtered length {len(ds)}")

    if local_rank == 0 and torch.distributed.is_initialized():
        torch.distributed.barrier()

    # make comparison pairs from completion list
    if local_rank > 0 and torch.distributed.is_initialized():
        torch.distributed.barrier()

    def make_batch_pairs(sample):
        converted_sample = defaultdict(list)

        for sample_idx, (prompt, image, completions) in enumerate(zip(sample["prompt"], sample["image"], sample["completions"])):
            for comp_idx1, comp_idx2 in combinations(range(len(completions)), 2):
                avg_score1, avg_score2 = completions[comp_idx1]["score"], completions[comp_idx2]["score"]
                # get chosen and rejected responses
                if avg_score1 > avg_score2:
                    chosen = completions[comp_idx1]["response"]
                    rejected = completions[comp_idx2]["response"]
                elif avg_score2 > avg_score1:
                    chosen = completions[comp_idx2]["response"]
                    rejected = completions[comp_idx1]["response"]
                else:
                    continue
                converted_sample["prompt"].append(prompt)
                converted_sample["chosen"].append(chosen)
                converted_sample["rejected"].append(rejected)

        return converted_sample

    ds = ds.map(
        make_batch_pairs,
        batched=True,
        remove_columns=set(ds.column_names) - set(["prompt", "chosen", "rejected"]),
    )

    if local_rank == 0 and torch.distributed.is_initialized():
        torch.distributed.barrier()

    return ds


class CustomDPOTrainer(DPOTrainer):
    def concatenated_forward(
        self, model, batch):
        concatenated_batch = self.concatenated_inputs(batch)
        len_chosen = batch["chosen_labels"].shape[0]
        images = batch["images"].repeat(2, 1,1,1)
        model_kwargs = (
            {
                "labels": concatenated_batch["concatenated_labels"],
                "decoder_input_ids": concatenated_batch.pop("concatenated_decoder_input_ids", None),
            }
            if self.is_encoder_decoder
            else {}
        )
        outputs, all_labels = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            labels=concatenated_batch["concatenated_labels"],
            images=images,
            return_label=True,
            **model_kwargs,
        )
        all_logits = outputs.logits.to(torch.float32)

        all_logps = self._get_batch_logps(
            all_logits,
            all_labels,
            average_log_prob=False,
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)
def train():
    global local_rank
    # 1. load argument
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    # 2. prepare model
    # 2.1 kbit & compute_dtype  ===>  model
    # 2.2 vision_tower.property  and load
    
    # 3. prepare tokenizer
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    bnb_model_from_pretrained_args = get_bnb_model_args(training_args)
    # TODO: vision_tower type check
    if model_args.vision_tower is not None:
        model = ModelSelect(model_args.model_name_or_path).from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args,
            attn_implementation="flash_attention_2",
            torch_dtype=compute_dtype
        )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False
    if model_args.freeze_backbone:
        model.model.requires_grad_(False)
    if training_args.bits in [4, 8]:
        model.config.torch_dtype = (
            torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    if training_args.lora_enable:
        model = lora_setting(model, training_args)
    
    Tokenizer, init_tokenizer = TokenizerSelect(model_args.model_name_or_path)()
    tokenizer = Tokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer = init_tokenizer(tokenizer)

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token

        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.pad_token = tokenizer.pad_token

        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    model.config.tokenizer_padding_side = tokenizer.padding_side
    if model_args.vision_tower is not None:
        # model.config.tune_embed_tokens = training_args.tune_embed_tokens = model_args.tune_embed_tokens
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )

        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
        
        if training_args.gradient_checkpointing:
            vision_tower.vision_tower.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            if hasattr(vision_tower.vision_tower, "enable_input_require_grads"):
                vision_tower.vision_tower.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)
                vision_tower.vision_tower.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        model.config.tune_vision_tower = training_args.tune_vision_tower = model_args.tune_vision_tower
        model.config.tune_entire_model = training_args.tune_entire_model = model_args.tune_entire_model
        if model_args.tune_entire_model:
            rank0_print(f'Tune entire model!')
            lr_of_mlp = training_args.mm_projector_lr if training_args.mm_projector_lr is not None else training_args.learning_rate
            rank0_print(f'Tune the MLP! The LR of MLP is {lr_of_mlp}')
            if training_args.lora_enable:
                unlock_vit(training_args, model_args, vision_tower)
            else:
                model.requires_grad_(True)
                unlock_vit(training_args, model_args, vision_tower)

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    rank0_print(model.get_model().mm_projector)

    if training_args.bits in [4, 8]:
        lora_kbit_setting(model, training_args)



    rank0_print("trainable parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    rank0_print("total parameters: ", sum(p.numel() for p in model.parameters()))

    dataset = datasets.load_dataset('json', data_files=data_args.data_path, split="train")

    def sample_data(dataset, sample_percentage):
        if 'id' not in dataset.column_names:
            dataset = dataset.add_column('id',list(range(0, dataset.num_rows)))
            ids = [id for id in dataset['id']]
            # unique_ids = set(ids)
        else:
            ids = [id.split("-")[0] for id in dataset['id']]
            unique_ids = set(ids)

        sampled_indices = []
        # for unique_id in unique_ids:
        #     indices = [i for i, id_val in enumerate(ids) if id_val == unique_id]
        #     # sampled_indices.extend(random.sample(indices, int(sample_percentage * len(indices))))
        #     sampled_indices.extend(indices[:int(sample_percentage * len(indices))])
        import random
        sampled_indices.extend(random.sample(ids, int(sample_percentage * len(ids))))
        sampled_data = dataset.select(sampled_indices)
        return sampled_data

    if data_args.subset_percent > 0:
        dataset = sample_data(dataset, data_args.subset_percent)
    train_dataset = dataset
    train_dataset = bpo_paired_dataset(train_dataset, local_rank, data_args)
    collator = LLaVADPODataCollator(data_args=data_args, tokenizer=tokenizer, max_length=1024)
    print(f"rank {local_rank} train length {len(train_dataset)}")
    trainer = CustomDPOTrainer(
        model,
        args=training_args,
        beta=training_args.beta,
        train_dataset=train_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        max_length=training_args.model_max_length,
    )


    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=False)
    else:
        trainer.train()

    trainer.save_state()
    model.config.use_cache = True
    if training_args.lora_enable:
        lora_save_model(model, training_args)
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()
