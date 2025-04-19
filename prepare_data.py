import numpy as np
import cv2
import pickle
import json
import re
import random
import pandas as pd
import torch
from transformers import AutoTokenizer


class ImageInstructionOutputDataset:
    def __init__(self, image_paths, json_paths, image_size, root_path, bs, resized=False, ind=None):
        if ind is not None:
            image_paths = list(np.array(image_paths)[ind])
            json_paths = list(np.array(json_paths)[ind])
        self.image_paths = image_paths
        self.resized_paths = []
        self.json_paths = json_paths
        self.image_size = image_size
        self.root_path = root_path
        self.resized = resized
        self.instructions, self.outputs = self.__readJson__(json_paths)
        self.samples = self.__readImage__(image_paths)
        self.baseColour, self.masterCategory, self.subCategory = self.__readMetadata__(json_paths)
        self.inds = list(range(len(self.samples)))
        self.text_map = pickle.load(open('data/fashion-dataset/map_dict.pkl', "rb"))
#         self.augment()
        self.batch_size = bs
        self.__batch__(bs)

    def sample(self, inds):
        piece = ImageInstructionOutputDataset([], [], self.image_size, self.root_path, 1, self.text_map, self.resized)
        piece.instructions = []
        piece.outputs = []
        piece.samples = []
        piece.image_paths = []
        piece.baseColour = []
        piece.masterCategory = []
        piece.subCategory = []
        for ind in inds:
            piece.image_paths.append(self.image_paths[ind])
            piece.instructions.append(self.instructions[ind])
            piece.outputs.append(self.outputs[ind])
            piece.samples.append(self.samples[ind])
            piece.baseColour.append(self.baseColour[ind])
            piece.masterCategory.append(self.masterCategory[ind])
            piece.subCategory.append(self.subCategory[ind])
        piece.inds = list(range(len(piece.samples)))
        return piece

    def shuffle(self, bs):
        #   if self.batch_size > 1:
        self.__flatten__()
        self.__batch__(bs)
        random.shuffle(self.inds)

    def augment(self):
        print('Data augmenting...')
        #         self.flatten()
        l = len(self)
        for i, piece in enumerate(self):
            print(f'\r{i}/{l}', end='')
#             image = piece[3]
#             image[np.sum(image, axis=2) == 765] = np.random.randint(256, size=(1, 3))
#             piece[3] = np.expand_dims(image, axis=3)
            color = piece[4]
            # label = piece[2]
            if color in list(self.text_map.keys()):
                # piece[4] = self.text_map[color].lower()
                # piece[2].replace(color.lower(), self.text_map[color].lower())
                # if pd.isna(self.text_map[color]) or pd.isna(self.text_map[color]):
                self.__setitem__(i, self.text_map[color],
                                 piece[2].replace(color.lower(), self.text_map[color].lower()))
            self.__setitem__(i, label=piece[2].replace(piece[1]+' ',''), instruction='product title')
        print('')

    #         self.__batch__(self.batch_size)
    def __setitem__(self, i, color=None, label=None, instruction=None):
        if color is not None:
            self.baseColour[i] = color
        if label is not None:
            self.outputs[i] = label
        if instruction is not None:
            self.instructions[i] = instruction

    def __batch__(self, bs):
        print(f'Viewing the set into batch_size={bs}')
        l = len(self.samples)
        self.instructions = [[self.instructions[j * bs + i] for i in range(bs)] for j in range(int(l / bs))]
        self.outputs = [[self.outputs[j * bs + i] for i in range(bs)] for j in range(int(l / bs))]
        self.image_paths = [[self.image_paths[j * bs + i] for i in range(bs)] for j in range(int(l / bs))]
        self.samples = [np.stack(self.samples[j * bs:j * bs + bs], 3) for j in range(int(l / bs))]
        self.baseColour = [[self.baseColour[j * bs + i] for i in range(bs)] for j in range(int(l / bs))]
        self.masterCategory = [[self.masterCategory[j * bs + i] for i in range(bs)] for j in range(int(l / bs))]
        self.subCategory = [[self.subCategory[j * bs + i] for i in range(bs)] for j in range(int(l / bs))]
        self.inds = list(range(len(self.samples)))
        self.batch_size = bs

    def __flatten__(self):
        self.instructions = sum(self.instructions, [])
        self.outputs = sum(self.outputs, [])
        self.image_paths = sum(self.image_paths, [])
        # if self.batch_size > 1:
        self.samples = [images[:, :, :, i] for images in self.samples for i in range(images.shape[3])]
        self.baseColour = sum(self.baseColour, [])
        self.masterCategory = sum(self.masterCategory, [])
        self.subCategory = sum(self.subCategory, [])
        self.inds = list(range(len(self.samples)))

    def __readImage__(self, image_paths):
        print('Loading (and resizing) images...')  # We resize images, in case we run out of memory
        image_list = []
        n = len(image_paths)
        i = 0
        if self.resized:
            for path in image_paths:
                i += 1
                print(f'\r{i}/{n}', end='')
                path = self.root_path + 'images/' + path.split('/')[-1]
                img = cv2.imread(path, 1)
                image_list.append(img)
        else:
            for path in image_paths:
                i += 1
                print(f'\r{i}/{n}', end='')
                img = cv2.imread(path, 1)
                img = cv2.resize(img, self.image_size)
                cv2.imwrite(f"{self.root_path}/images/{path.split('/')[-1]}", img)
                image_list.append(img)
        print('')
        return image_list

    def __readJson__(self, json_paths):
        print('Loading json files to instructions and labels...')  # We resize images, in case we run out of memory
        instruction_list = []
        output_list = []
        n = len(json_paths)
        i = 0
        for path in json_paths:
            i += 1
            print(f'\r{i}/{n}', end='')
            with open(path, 'rb') as f:
                json_data = json.load(f)
                instruction_list.append(clean_and_lowercase(json_data['data']['brandName']))
                # if json_data['data']['productDisplayName'] == 'NA' or json_data['data']['productDisplayName'] is None:
                #     pass
                output_list.append(clean_and_lowercase(json_data['data']['productDisplayName']) + ' <eos>')
        print('')
        return instruction_list, output_list

    def __readMetadata__(self, json_paths):
        print('Loading other metadata...')  # We resize images, in case we run out of memory
        color_list = []
        master_cate_list = []
        sub_cate_list = []
        metadata = pd.read_csv('data/fashion-dataset/styles.csv')
        for path in json_paths:
            id_ = int(path.split('/')[-1].split('.')[0])
            piece = metadata[metadata['id'] == id_]
            color_list.append(piece['baseColour'].to_numpy()[0])
            master_cate_list.append(piece['masterCategory'].to_numpy()[0])
            sub_cate_list.append(piece['subCategory'].to_numpy()[0])
        return color_list, master_cate_list, sub_cate_list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return [self.image_paths[self.inds[idx]],
                self.instructions[self.inds[idx]],
                self.outputs[self.inds[idx]],
                self.samples[self.inds[idx]],
                self.baseColour[self.inds[idx]],
                self.masterCategory[self.inds[idx]],
                self.subCategory[self.inds[idx]]]


def clean_and_lowercase(input_string):  # lowercase chars, delete special chars and duplicated blank
    cleaned_string = re.sub(r'[^a-zA-Z0-9\s\-\']', ' ', input_string)
    lowercased_string = cleaned_string.lower().strip()
    lowercased_string = re.sub(r'\s+', ' ', lowercased_string)
    return lowercased_string


def update_tokenizer(tokenizer, data_set):
    print('Update tokenizer')
    i = 0
    for (image_path, prompt, label, image,_,_,_) in data_set:
        print(f'{i}', end='\r')
        i += 1
        word_list = list(set(prompt[0].split(' ') + label[0].split(' ')))
        tokenizer.add_tokens(word_list)
    print('')


def update_embeddings(model, ori_vocab_size, dst_vocab_size, device):
    print('Update embeddings')
    if dst_vocab_size - ori_vocab_size <= 0:
        print('Don\'t need to update')
        return
    dim = model.language_model.base_model.model.model.embed_tokens.weight.shape[1]
    embed_weight = torch.rand(dst_vocab_size - ori_vocab_size, dim, dtype=torch.float32).to(device) - 0.5
#     embed_bias = torch.zeros(dst_vocab_size - ori_vocab_size, dtype=torch.float32).to(device)
    # add new rows for the embedding layer
    model.language_model.base_model.model.model.embed_tokens.weight = \
        torch.nn.Parameter(torch.cat([model.language_model.base_model.model.model.embed_tokens.weight, embed_weight], 0))
    model.language_model.base_model.model.model.embed_tokens.num_embeddings = dst_vocab_size
    # add new rows for the output logits layer
    model.language_model.tie_weights()
    model.language_model.lm_head.out_features = dst_vocab_size
#     model.language_model.lm_head.bias = \
#         torch.nn.Parameter(torch.cat([model.language_model.lm_head.bias, embed_bias], 0))


def prepare_dataset(bs=4, training=True, validation=True,augment=True):
    if training:
        with open('data/fashion-dataset/training_set.pkl', 'rb') as f:
            training_set = pickle.load(f)
            setattr(training_set,'text_map',pickle.load(open('data/fashion-dataset/map_dict.pkl', "rb")))
            training_set.__flatten__()
            if augment:
                training_set.augment()
            training_set.__batch__(bs)
    if validation:
        with open('data/fashion-dataset/validation_set.pkl', 'rb') as f:
            validation_set = pickle.load(f)
            setattr(validation_set,'text_map',pickle.load(open('data/fashion-dataset/map_dict.pkl', "rb")))
            validation_set.__flatten__()
            if augment:
                validation_set.augment()
            validation_set.__batch__(bs)
    if training and validation:
        return training_set, validation_set
    elif training and not validation:
        return training_set
    elif not training and validation:
        return validation_set
    else:
        return None

    
# if __name__ == '__main__':
    # images_roots = 'data/fashion-dataset/images/'
    # images_name = os.listdir(images_roots)
    # total_num = len(images_name)
    #
    # json_roots = 'data/fashion-dataset/styles/'
    # json_name = os.listdir(json_roots)
    # images_paths = []
    # json_paths = []
    # min_id = 1163
    # max_id = 60000
    # print('One-to-one corresponding check for the whole dataset...')
    # for id_ in np.arange(min_id, max_id + 1):
    #     print(f'{id_}/{max_id}',end='\r')
    #     id_ = str(id_)
    #     if id_ + '.jpg' in images_name and id_ + '.json' in json_name:
    #         images_paths.append(images_roots + id_ + '.jpg')
    #         json_paths.append(json_roots + id_ + '.json')





#     with open('data/fashion-dataset/training_set.pkl', 'wb') as f:
#         pickle.dump(training_set, f)

#     with open('data/fashion-dataset/validation_set.pkl', 'wb') as f:
#         pickle.dump(validation_set, f)

#     with open('data/fashion-dataset/testing_set.pkl', 'wb') as f:
#         pickle.dump(testing_set, f)
