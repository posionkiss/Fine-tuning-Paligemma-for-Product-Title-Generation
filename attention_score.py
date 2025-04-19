import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
# import cv2
from modeling_gemma import KVCache
from processing_paligemma import PaliGemmaProcessor, LabelProcessor
from prepare_data import update_embeddings, ImageInstructionOutputDataset
from utils import *

print(f"Loading data")
with open('data/fashion-dataset/validation_set.pkl', 'rb') as f:
    validation_set = pickle.load(f)
# validation_set.__batch__(1)
validation_set.shuffle(1)
# with open('data/fashion-dataset/validation_set.pkl', 'rb') as f:
#     validation_set = pickle.load(f)
# with open('data/fashion-dataset/testing_set.pkl', 'rb') as f:
#     testing_set = pickle.load(f)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = 'cpu'
print("Device in use: ", device)
print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3} GB')
model_path = 'paligemma'

print(f"Loading model")
tokenizer_modified = True
rank = 16  # LoRA rank
model, tokenizer = load_hf_model(model_path, device, 0)
model = apply_lora(model, device, rank=rank)
model = model.half()
model = model.to(device).eval()
vocab_size = len(tokenizer)

update_embeddings(model, model.vocab_size, vocab_size, device)
num_image_tokens = model.config.vision_config.num_image_tokens
image_size = model.config.vision_config.image_size
processor = PaliGemmaProcessor(tokenizer, num_image_tokens, (image_size, image_size))
label_processor = LabelProcessor(tokenizer)
max_tokens_to_generate = 100
do_sample = False
stop_token = processor.tokenizer.eos_token_id

checkpoint = 'my_model/1/model_2025-01-04_rank16_epoch_44_.pth'
model.load_state_dict(torch.load(checkpoint))

'''
注意力分数可视化
'''
print('Start validating:')
color_correct = 0
total_num_color = 0
pred_colors = []
true_colors = []

with torch.no_grad():
    for i, (image_paths, prompts, label_words, images) in enumerate(validation_set):
        images = [np.squeeze(images[:, :, :, i]) for i in range(images.shape[3])]
        model_inputs = processor(prompts, images)
        model_inputs = move_inputs_to_device(model_inputs, device)
        input_ids = model_inputs['input_ids'].to(device)
        attention_mask = model_inputs['attention_mask'].to(device)
        pixel_values = model_inputs['pixel_values'].half().to(device)
        kv_cache = KVCache()
        labels_ids, labels_attention_mask, total_num = get_ids(tokenizer, label_words)
        labels_ids = labels_ids.to(device)
        labels_attention_mask = labels_attention_mask.to(device)
        generated_words = ''
        loss_list_sample = []
        correct_num = 0
        attn_score_list = []
        for j in range(max_tokens_to_generate):
            outputs, attention_weight_list = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
            )
            kv_cache = outputs["kv_cache"]
            if j == 0:
                for attn_weight in attention_weight_list:
                    attn_score_list.append(np.resize(torch.sum(attn_weight, 1)[0].sum(axis=0).cpu(), (16, 16)))

            pred_ids = outputs["logits"][:, -1, :].argmax(1)
            pred_word = tokenizer.decode(pred_ids)
            generated_words += f'{pred_word} '
            pred_ids = torch.unsqueeze(pred_ids, 0)
            input_ids = pred_ids
            attention_mask = torch.cat([attention_mask, (pred_ids != 0).int()], dim=-1).detach().to(
                device)  # 更新attention_mask，若batch中的某一条样本已经输入完，则mask对应的位置为0，否则为1

            pred_ids = torch.softmax(outputs["logits"][:, -1, :], dim=-1, dtype=torch.float16).argmax(1)
            if pred_ids == stop_token:
                break
        fig, ax = plt.subplots(7, 4, figsize=(40, 40))
        # fig.tight_layout()
        disp_image = plt.imread(image_paths[0])
        ax[0][0].imshow(disp_image)
        ax[0][0].title.set_text(label_words[0])
        for l, attn_score in enumerate(attn_score_list):
            r = (l + 1) // 4
            c = l + 1 - r * 4
            attn_score = (attn_score / attn_score.max() * 255).astype(np.uint8)
            ax[r][c].imshow(attn_score, cmap='gray')
            ax[r][c].set_title(f'Layer {l + 1}', fontsize=30)
        fig.suptitle('Attention Score of Vision-Transformer',fontsize=60)
        plt.show()
        input(f'{i}\n\t{label_words[0]}\n\t{generated_words}')
