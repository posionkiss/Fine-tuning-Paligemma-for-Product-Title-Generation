import pickle
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from modeling_gemma import KVCache
from processing_paligemma import PaliGemmaProcessor, LabelProcessor
from utils import move_inputs_to_device
from prepare_data import ImageInstructionOutputDataset, update_tokenizer, update_embeddings
from utils import *
from datetime import datetime

print(f"Loading data")
with open('data/fashion-dataset/training_set.pkl', 'rb') as f:
    training_set = pickle.load(f)
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
model_path = 'paligemma'


print(f"Loading model")
tokenizer_modified = True
rank = 16  # LoRA rank
model, tokenizer = load_hf_model(model_path, device, 1)
model = apply_lora(model, device, rank=rank)
model = model.half()
model = model.to(device).eval()
vocab_size = len(tokenizer)


update_embeddings(model, model.vocab_size, vocab_size, device)
num_image_tokens = model.config.vision_config.num_image_tokens
image_size = model.config.vision_config.image_size
processor = PaliGemmaProcessor(tokenizer, num_image_tokens, (image_size, image_size))
label_processor = LabelProcessor(tokenizer)
stop_token = processor.tokenizer.eos_token_id


# cp_epoch=49
# checkpoint = f'my_model/model_2025-01-05_rank16_epoch_{cp_epoch-1}_.pth'
# model.load_state_dict(torch.load(checkpoint))


batch_size = 4
lr = 2e-4
optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-3)
loss_fn = torch.nn.CrossEntropyLoss().to(device)
epochs = 3


print('Start training:')
# torch.autograd.set_detect_anomaly(True)
for epoch in range(epochs):
    # epoch += cp_epoch4
    loss_list = []
    acc_list = []
    print('Shuffle training set')
    training_set.shuffle(batch_size)  # shuffle training set
    for i, (image_paths, prompts, label_words, images) in enumerate(training_set):
        images = [images[:, :, :, i] for i in range(images.shape[3])]
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

        for j in range(labels_ids.shape[1]):
            outputs, attention_weight_list = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
            )
            kv_cache = outputs["kv_cache"]
            #             next_token_logits = torch.softmax(outputs["logits"][:, -1, :], dim=-1, dtype=torch.float16)
            word_ids = labels_ids[:, j].to(device)
            word_vec = torch.zeros_like(outputs["logits"][:, -1, :], dtype=torch.float16)
            word_vec[list(range(batch_size)), word_ids[:, 0]] = 1.0
            input_ids = word_ids.detach()  # 标签的下一个（一个batch）的字符作为下一次输入
            attention_mask = torch.cat([attention_mask, (word_ids != 0).int()], dim=-1).detach().to(
                device)  # 更新attention_mask，若batch中的某一条样本已经输入完，则mask对应的位置为0，否则为1

            loss = loss_fn(outputs["logits"][:, -1, :], word_ids[:, 0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            kv_cache.clear_trace()
            loss = loss.item()
            pred_ids = torch.softmax(outputs["logits"][:, -1, :], dim=-1, dtype=torch.float16).argmax(1)
            ei = word_ids[:, 0] != 0
            correct_num += (word_ids[:, 0][ei] == pred_ids[ei]).sum().item()
            loss_list_sample.append(loss)
            print(f'\repoch{epoch}\t{i}\tloss: {round(loss, 4)}', end='')
        # print(f'\repoch{epoch}\t{i}\tacc:{round(correct_num / total_num, 4)}\ttokens to predict:{total_num}\tcorrect:{correct_num}',end='')
        print(f'\repoch{epoch}\t{i}\tacc:{round(correct_num / total_num, 4)}\ttokens to predict:{total_num}\tcorrect:{correct_num}',end='')
        loss_list.append(sum(loss_list_sample) / len(loss_list_sample))
        acc_list.append(correct_num / total_num)
        torch.cuda.empty_cache()
    print('\n''Saving')
    torch.save(model.state_dict(), f'my_model/model_{str(datetime.now().date())}_rank{rank}_epoch_{epoch}_.pth')
    with open(f'track/loss_{str(datetime.now().date())}_rank{rank}_epoch_{epoch}_.pkl', 'wb') as f:
        pickle.dump(loss_list, f)
    with open(f'track/acc_{str(datetime.now().date())}_rank{rank}_epoch_{epoch}_.pkl', 'wb') as f:
        pickle.dump(acc_list, f)