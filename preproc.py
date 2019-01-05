import numpy as np
import skimage.io as imio
from skimage import color
import os, glob


path = os.getcwd()
train_data_path = path + "/data/train"


def load_all_images():
    images = glob.glob(train_data_path + '/**/*.JPEG', recursive=True)
    return images


def normalize(image_file):
    img = imio.imread(image_file) # load the image
    shape = img.shape
    height, width = shape[0], shape[1]
    if shape[2] == 4: # if image has alpha channel
        img =  color.rgba2rgb(img) #convert to the rgb
    padded = np.array(img)
    print(padded.shape)
    if height < width:
        diff = width - height
        padded = np.pad(padded,((diff//2, diff- diff//2),(0,0),(0,0)),mode='constant')
        print(padded.shape)
    elif height > width:
        diff = height - width
        padded = np.pad(padded, ((0,0),(diff // 2, diff - diff // 2),(0,0)), mode='constant')
    padded.resize(224,224,3)


all_images = load_all_images()
normalize(all_images[0])