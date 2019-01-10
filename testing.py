import numpy as np
import cv2
from sklearn.metrics import confusion_matrix
from classification import classify_image, labels


def compute_accuracy():
    y_true = [] #true labels
    true_windows = []
    with open('data/test/bounding_box.txt', 'r') as file:
        for i,line in enumerate(file):
            y_true.append(line[0:line.find(',')])
            fields = line.split(',')
            x1 = int(fields[1])
            y1 = int(fields[2])
            x2 = int(fields[3])
            y2 = int(fields[4])
            true_windows.append((x1,y1,x2-x1,y2-y1))
    y_predicted = [] # classifier's prediction
    overlap_percentage = []
    is_classified_true = []

    def classify_localize_test_images():
        test_images_path = 'data/test/images'
        for i in range(100):
            image = cv2.imread('data/test/images/%d.JPEG' %(i))
            [cur_label,cur_window] = classify_image(image)
            #classification
            y_predicted.append(cur_label)
            is_classified_true.append(cur_label.__eq__(y_true[i]))
            print(i, cur_label, test_images_path + '/' + str(i) +'.JPEG')
            #localization
            x, y, w, h = cur_window

            def compute_intersected_area(w2):
                w1_x2 = x + w
                w1_y2 = y + h
                w2_x2 = w2[0] + w2[2]
                w2_y2 = w2[1] + w2[3]
                dx = min(w1_x2,w2_x2) - max(x,w2[0])
                dy = min(w1_y2,w2_y2) - max(y,w2[1])
                print(dx,dy)
                if dx >= 0 and dy >= 0:
                    return dx * dy
                else:
                    return -1

            intersected_area = compute_intersected_area(true_windows[i])
            if intersected_area == -1:
                print("no intersection")
                overlap_percentage.append(0)
            #elif(intersected_area == 0):
            else:
                cur_overlap = intersected_area / (w*h + true_windows[i][2] * true_windows[i][3] - intersected_area)
                overlap_percentage.append(cur_overlap)



    classify_localize_test_images()
    cm = confusion_matrix(y_true,y_predicted,labels)
    return cm, is_classified_true, overlap_percentage, y_predicted


def get_statistics(data):

    overlap_percantage = data[2]
    predictions = data[3]
    confusion_matrix = data[0]

    m = np.asarray(confusion_matrix)
    matrix = np.matrix(m)
    with open('confusion_matrix.txt','wb') as f:
        for line in matrix:
            np.savetxt(f,line,fmt='%.2f')
    true_positives = np.trace(m)
    classification_percantage = true_positives/100
    with open('eval.txt','wb') as f:
        f.write('Classification Percentage: ' + str(classification_percantage))
        f.write("Predictions: ")
        np.savetxt(f,predictions,fmt='%.2f',delimiter=', ')
        f.write('Overlap Percentage:')
        np.savetxt(f,overlap_percantage,fmt='%.2f',delimiter=', ')




accuracy_data = compute_accuracy()
get_statistics(accuracy_data)
