import pickle
import re


if __name__ == '__main__':
    print(f"Loading data")
    with open('data/fashion-dataset/training_set.pkl', 'rb') as f:
        training_set = pickle.load(f)
    with open('data/fashion-dataset/validation_set.pkl', 'rb') as f:
        validation_set = pickle.load(f)
    with open('data/fashion-dataset/testing_set.pkl', 'rb') as f:
        testing_set = pickle.load(f)

    for i, (image_path, prompt, label, image) in enumerate(training_set):


    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    # update_tokenizer(tokenizer, validation_set)
    # update_tokenizer(tokenizer, training_set)
    # update_tokenizer(tokenizer, testing_set)