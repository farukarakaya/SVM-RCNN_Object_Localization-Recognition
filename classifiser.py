import copy

from sklearn import svm
import pickle
import numpy as np

global_model = None

def train_classfier():
    global global_model
    all_features = pickle.load(open('training_features.pickle', "rb"))
    all_labels = pickle.load(open('training_labels.pickle', "rb"))

    all_features = np.asarray(all_features)
    all_features.reshape(1,-1)

    model = svm.LinearSVC(max_iter= 10000, C=0.5, multi_class='crammer_singer')   #SVC(C =1500, gamma= "scale", decision_function_shape='ovo') #
    model.fit(all_features, all_labels)
    global_model = model

    pickle.dump(global_model, open('model.pickle', 'wb'))


def make_prediction(feature):
    global global_model
    if global_model == None:
        global_model = pickle.load(open('model.pickle', "rb"))

    global_model.decision_function_shape = "ovr"
    return global_model.predict([feature])


def confidence_score(feature):
    global global_model
    if global_model == None:
        global_model = pickle.load(open('model.pickle', "rb"))

    global_model.decision_function_shape = "ovr"
    return global_model.decision_function([feature])



train_classfier()
all_features = pickle.load(open('training_features.pickle', "rb"))
labels = pickle.load(open('training_labels.pickle', "rb"))

for i,feature in enumerate(all_features):
    print(str(labels[i]) + ' -> ' + str(make_prediction(feature)))

