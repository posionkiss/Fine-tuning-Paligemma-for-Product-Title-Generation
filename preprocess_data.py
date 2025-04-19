import numpy as np
from prepare_data import *
import matplotlib.pyplot as plt
print(f"Loading data")
with open('data/fashion-dataset/validation_set.pkl', 'rb') as f:
    validation_set = pickle.load(f)

# with open('data/fashion-dataset/training_set.pkl', 'rb') as f:
#     training_set = pickle.load(f)
# training_set.shuffle(1)
# with open('data/fashion-dataset/testing_set.pkl', 'rb') as f:
#     testing_set = pickle.load(f)

for piece in validation_set:
    image = piece[3][:,:,:,0]
    image[np.sum(image,axis=2) == 765] = np.random.randint(256,size=(1,3))
    piece = list(piece)
    piece[3] = np.expand_dims(image,axis=3)
    piece = tuple(piece)

for piece in training_set:
    image = piece[3][:,:,:,0]
    image[np.sum(image,axis=2) == 765] = np.random.randint(256,size=(1,3))
    piece = list(piece)
    piece[3] = np.expand_dims(image,axis=3)
    piece = tuple(piece)

for piece in testing_set:
    image = piece[3][:,:,:,0]
    image[np.sum(image,axis=2) == 765] = np.random.randint(256,size=(1,3))
    piece = list(piece)
    piece[3] = np.expand_dims(image,axis=3)
    piece = tuple(piece)

training_set.shuffle(4)
with open('data/fashion-dataset/training_set.pkl', 'wb') as f:
    pickle.dump(training_set, f)

with open('data/fashion-dataset/validation_set.pkl', 'wb') as f:
    pickle.dump(validation_set, f)

with open('data/fashion-dataset/testing_set.pkl', 'wb') as f:
    pickle.dump(testing_set, f)