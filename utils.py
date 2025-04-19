from torch import dtype

from modeling_gemma import PaliGemmaForConditionalGeneration, PaliGemmaConfig
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig
from transformers import AutoTokenizer, BitsAndBytesConfig
from transformers import AutoTokenizer
import json
import glob
from typing import Tuple
from safetensors import safe_open
import os
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize


def load_hf_model(
    model_path: str,
    bit: int = 4,
    lora: bool = True,
    lora_rank: int = 16,
    freeze_vision: bool = True,
):
    # 1. 构建量化配置
    if bit == 4:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif bit == 8:
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
        #             llm_int8_enable_fp32_cpu_offload=True,
        )
    else:
        quant_config = None

    # ✅ 加载 config（我们自己重写过）
    config = PaliGemmaConfig.from_pretrained(model_path)

    # ✅ 加载模型，传入 config！
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_path,
        config=config,
        device_map="auto",
        quantization_config=quant_config,
    )

    # ✅ 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")

    # ✅ 冻结 vision 模块（可选）
    if freeze_vision and hasattr(model, "vision_tower"):
        for param in model.vision_tower.parameters():
            param.requires_grad = False
        print("✅ Vision tower frozen")

    # ✅ LoRA（可选）
    if lora:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        model.language_model = prepare_model_for_kbit_training(model.language_model, use_gradient_checkpointing=False)
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj","k_proj","gate_proj", "out_proj", "up_proj", "down_proj","linear"],
        )
        model.language_model = get_peft_model(model.language_model, lora_config)
        model.language_model.print_trainable_parameters()

    return model, tokenizer

# def load_hf_model(model_path: str, device: str, ori_model=True) -> Tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]:
#     # Load the tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right", model_max_length=512)
#     assert tokenizer.padding_side == "right"
#     # Load the model's config
#     with open(os.path.join(model_path, "config.json"), "r") as f:
#         model_config_file = json.load(f)
#         config = PaliGemmaConfig(**model_config_file)
#     model = PaliGemmaForConditionalGeneration(config).to(device)
#     if ori_model:
#         # Find all the *.safetensors files
#         safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))

#         # ... and load them one by one in the tensors dictionary
#         tensors = {}
#         for safetensors_file in safetensors_files:
#             with safe_open(safetensors_file, framework="pt", device="cpu") as f:
#                 for key in f.keys():
#                     tensors[key] = f.get_tensor(key)

#     # Load the state dict of the model
#         model.load_state_dict(tensors, strict=False)
#     # Tie weights
#         model.tie_weights()

#     return (model, tokenizer)


def move_inputs_to_device(model_inputs: dict, device: str):
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    return model_inputs


class LoRAParametrization(nn.Module):
    def __init__(self, features_in, features_out, device, rank=1, alpha=1):
        super().__init__()
        # Section 4.1 of the paper:
        #   We use a random Gaussian initialization for A and zero for B, so ∆W = BA is zero at the beginning of training
        self.lora_A = nn.Parameter(torch.zeros((rank, features_out), dtype=torch.float16).to(device))
        self.lora_B = nn.Parameter(torch.zeros((features_in, rank), dtype=torch.float16).to(device))
        # self.lora_A.requires_grad = True
        # self.lora_B.requires_grad = True
        nn.init.normal_(self.lora_A, mean=0, std=0.05)
        # Section 4.1 of the paper:
        #   We then scale ∆Wx by α/r , where α is a constant in r.
        #   When optimizing with Adam, tuning α is roughly the same as tuning the learning rate if we scale the initialization appropriately.
        #   As a result, we simply set α to the first r we try and do not tune it.
        #   This scaling helps to reduce the need to retune hyperparameters when we vary r.
        self.scale = alpha / rank
        self.enabled = True

    def forward(self, original_weights):
        if self.enabled:
            # Return W + (B*A)*scale
            m1 = torch.matmul(self.lora_B, self.lora_A)
            m2 = m1.view(original_weights.shape)
            return original_weights + m2 * self.scale
        else:
            return original_weights


def linear_layer_parameterization(layer, device, rank=1, lora_alpha=1):
    # Only add the parameterization to the weight matrix, ignore the Bias

    # From section 4.2 of the paper:
    #   We limit our study to only adapting the attention weights for downstream tasks and freeze the MLP modules (so they are not trained in downstream tasks) both for simplicity and parameter-efficiency.
    #   [...]
    #   We leave the empirical investigation of [...], and biases to a future work.

    features_in, features_out = layer.weight.shape
    return LoRAParametrization(features_in, features_out, device, rank=rank, alpha=lora_alpha)


def seek_for_linear(node, output_ist):  # return all linear layers for LoRA application
    children_list = list(node.children())
    if isinstance(node, nn.Linear) and node.in_features < 2e4 and node.out_features < 2e4:
        # if isinstance(node, nn.Linear):
        output_ist.append(node)
        node.weight.requires_grad = False
        return output_ist
    if len(children_list) == 0:
        return output_ist
    for childNode in children_list:
        output_ist = seek_for_linear(childNode, output_ist)
    return output_ist


def enable_disable_lora(layer_list, enabled=True):
    for layer in layer_list:
        layer.parametrizations["weight"][0].enabled = enabled


def apply_lora(model, device, rank=1, lora_alpha=1):
    print('Applying LoRA')
    linear_list = []
    linear_list = seek_for_linear(model, linear_list)
    for ln_layer in linear_list:
        parametrize.register_parametrization(
            ln_layer, "weight", linear_layer_parameterization(ln_layer, device, rank=rank, lora_alpha=lora_alpha)
        )
    for name, param in model.named_parameters():
        if 'parametrizations' in name and 'lora' not in name:
            param.requires_grad = False
    return model


def get_ids(tokenizer, label_words):  # 将不等长的句子批量地转化成ids，并填充
    ids = []
    attention_mask = []
    total_num = 0
    for label in label_words:
        label_split = label.split(' ')
        total_num += len(label_split)
        n = tokenizer(label_split, return_tensors="pt", padding='longest', truncation=True)
        ids.append(n['input_ids'])
        attention_mask.append(n['attention_mask'])
    return pad_and_combine(ids), pad_and_combine(attention_mask), total_num


def pad_and_combine(tensor_list):
    # 找出最长的向量长度
    max_length = max(tensor.size(0) for tensor in tensor_list)

    # 初始化一个空列表来存放填充后的tensor
    padded_tensors = []

    # 对列表中的每个tensor进行填充操作
    for tensor in tensor_list:
        padding = max_length - tensor.size(0)  # 计算需要填充的0的数量
        padded_tensor = torch.cat(
            [tensor, torch.zeros(padding, *tensor.shape[1:], dtype=tensor.dtype, device=tensor.device)], dim=0)
        padded_tensors.append(padded_tensor)

    # 将填充后的tensor列表堆叠成一个二维tensor
    combined_tensor = torch.stack(padded_tensors, dim=0)

    return combined_tensor


