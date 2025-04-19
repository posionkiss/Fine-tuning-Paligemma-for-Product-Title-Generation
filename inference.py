import pickle
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
import os
from modeling_gemma import KVCache
from processing_paligemma import PaliGemmaProcessor, LabelProcessor
from utils import move_inputs_to_device
from prepare_data import ImageInstructionOutputDataset, update_tokenizer, update_embeddings, prepare_dataset
from utils import *
from datetime import datetime

def prepare_model(pretrained_path='paligemma', model_path='my_model/5/model_rank16_epoch_56_.pth'):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = 'cpu'
    print("Device in use: ", device)
    print(f"Loading model")
    model, tokenizer = load_hf_model(pretrained_path)
    model = model.to(device).eval()
    vocab_size = len(tokenizer)

    update_embeddings(model, model.vocab_size, vocab_size, device)
    num_image_tokens = model.config.vision_config.num_image_tokens
    image_size = model.config.vision_config.image_size
    processor = PaliGemmaProcessor(tokenizer, num_image_tokens, (image_size, image_size))
    stop_token = processor.tokenizer.eos_token_id

    model.load_state_dict(torch.load(model_path))
    return model, processor, tokenizer, stop_token, device

def inference(model, processor, tokenizer, stop_token, device):
    while True:
        image_path = input("Image path: ")
        if image_path == "/quit":
            break
        image = cv2.imread(image_path, 1)
        prompt = ['product title']
        model_inputs = processor(prompt, [image])
        model_inputs = move_inputs_to_device(model_inputs, device)

        input_ids = model_inputs['input_ids'].to(device)
        attention_mask = model_inputs['attention_mask'].to(device)
        pixel_values = model_inputs['pixel_values'].half().to(device)
        kv_cache = KVCache()
        generated_words = ''
        for j in range(20):
            outputs, _, _ = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                kv_cache=kv_cache)
            kv_cache = outputs["kv_cache"]
            pred_ids = outputs["logits"][:, -1, :].argmax(1)
            pred_word = tokenizer.decode(pred_ids)
            generated_words = generated_words + pred_word
            generated_words += f'{pred_word} '
            pred_ids = torch.unsqueeze(pred_ids, 0)
            input_ids = pred_ids
            attention_mask = torch.cat([attention_mask, (pred_ids != 0).int()], dim=-1).detach().to(
                device)  # 更新attention_mask，若batch中的某一条样本已经输入完，则mask对应的位置为0，否则为1
            pred_ids = torch.softmax(outputs["logits"][:, -1, :], dim=-1, dtype=torch.float16).argmax(1)
            if pred_ids == stop_token:
                break
            print(pred_word, end=' ')
            # print(pred_ids.item(), end=' ')
        print('')
        # print(f'\t{generated_words}\n')


if __name__ == '__main__':
    model, processor, tokenizer, stop_token, device = prepare_model()
    inference(model, processor, tokenizer, stop_token, device)


