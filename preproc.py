import numpy as np
import skimage.io as imio
from skimage import color
import os, glob
import resnet
import torch
import pickle
from sklearn import preprocessing as p

path = os.getcwd()
train_data_path = path + "/data/train"
labels = ['n01615121', 'n02099601', 'n02123159',
          'n02129604', 'n02317335', 'n02391049',
          'n02410509', 'n02422699', 'n02481823', 'n02504458']


def load_all_images():
    images = {}
    for label in labels:
        images[label] = glob.glob(train_data_path + '/' + label + '/*.JPEG', recursive=True)
    return images


def normalize(image_file):
    img = imio.imread(image_file) # load the image
    shape = img.shape
    height, width = shape[0], shape[1]
    if shape[2] == 4: # if image has alpha channel
        img =  color.rgba2rgb(img) #convert to the rgb
    padded = np.array(img)
    if height < width:
        diff = width - height
        padded = np.pad(padded,((diff//2, diff- diff//2),(0,0),(0,0)),mode='constant')
    elif height > width:
        diff = height - width
        padded = np.pad(padded, ((0,0),(diff // 2, diff - diff // 2),(0,0)), mode='constant')
    padded.resize(224, 224, 3, refcheck=False)

    return padded


def extract_feature(image):
    # we append an augmented dimension to indicate batch_size, which is one
    image = np.reshape(image, [1, 224, 224, 3])
    # model takes as input images of size [batch_size, 3, im_height, im_width]
    image = np.transpose(image, [0, 3, 1, 2])
    # convert the Numpy image to torch.FloatTensor
    image = torch.from_numpy(image)
    image = image.type(torch.FloatTensor)
    # extract features
    model = resnet.resnet50(pretrained=True)
    feature_vector = model.forward(image)
    # convert the features of type torch.FloatTensor to a Numpy array
    # so that you can either work with them within"rb" the sklearn environment
    # or save them as .mat files
    feature_vector = feature_vector.detach().numpy()

    # return normalized vector
    return feature_vector[0]

def get_all_features_by_labels():
    all_images = load_all_images()
    padded_img_vectors = []
    labels_of_vector = []
    count = 0
    for i,label in enumerate(labels):
        for img_path in all_images[label]:
            feature = extract_feature(normalize(img_path))
            padded_img_vectors.append(feature)
            labels_of_vector.append(i)
            count += 1
            print(count)

    pickle.dump(padded_img_vectors, open('training_features.pickle', 'wb'))
    pickle.dump(labels_of_vector, open('training_labels.pickle', 'wb'))

a = get_all_features_by_labels()
b = a
pass