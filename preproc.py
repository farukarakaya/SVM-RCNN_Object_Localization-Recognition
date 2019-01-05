import numpy as np
import skimage.io as imio
from skimage import color
import os, glob
import resnet
import torch
import pickle

path = os.getcwd()
train_data_path = path + "/data/train"
labels = ['n01615121', 'n02099601', 'n02123159',
          'n02129604', 'n02317335', 'n02391049',
          'n02410509', 'n02422699', 'n02481823', 'n02504458']


def load_all_images():
    images = {}
    for label in labels:
        images[label] = glob.glob(train_data_path + '/**/*.JPEG', recursive=True)
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
    padded.resize(224,224,3)
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
    # so that you can either work with them within the sklearn environment
    # or save them as .mat files
    feature_vector = feature_vector.detach().numpy()
    return feature_vector

def get_all_features_by_labels():
    all_images = load_all_images()
    padded_img_vectors_by_label = {}
    count = 0
    for label in labels:
        padded_img_vectors_by_label[label] = []
        for img_path in all_images[label]:
            padded_img_vectors_by_label[label].append(extract_feature(normalize(img_path)))
            count += 1
            print(count)

    outfile = open('training_features.pickle', 'wb')
    pickle.dump(padded_img_vectors_by_label, outfile)

a = get_all_features_by_labels()
b = a
pass