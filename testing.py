from sklearn.metrics import confusion_matrix
from classification import classify_image, labels
from preproc import path
import cv2
import glob

def classification_accuracy():
    y_true = [] #true labels
    with open('data/test/bounding_box.txt', 'r') as file:
        for i,line in enumerate(file):
            y_true.append(line[0:line.find(',')])
    y_predicted = [] # classifier's prediction

    def classify_test_images():
        test_images_path = 'data/test/images'
        #test_images = glob.glob(path + '/data/test/images' + '/*.JPEG', recursive=True)
        for i in range(100):
            image = cv2.imread(test_images_path + '/' + str(i) +'.JPEG')
            cur_label = classify_image(image)
            y_predicted.append(cur_label)
            print(i, cur_label, test_images_path + '/' + str(i) +'.JPEG')

    classify_test_images()

    cm = confusion_matrix(y_true,y_predicted,labels)
    return cm

def localization_accuracy():
    return

cm = classification_accuracy()
print(cm)