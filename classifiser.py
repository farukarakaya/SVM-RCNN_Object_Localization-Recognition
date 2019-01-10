import matplotlib.pyplot as plt
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

    model = svm.LinearSVC(max_iter=1000, multi_class='crammer_singer') #SVC(C =1000, gamma= 1, decision_function_shape='ovo')
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


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

'''
train_classfier()
all_features = pickle.load(open('training_features.pickle', "rb"))
labels = pickle.load(open('training_labels.pickle', "rb"))

for i,feature in enumerate(all_features):
    print(str(labels[i]) + ' -> ' + str(make_prediction(feature)))
'''