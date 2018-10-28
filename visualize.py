import itertools

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from data import get_genres


def draw_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def plot_cnf(model, dataset_x, dataset_y, GENRES):
    true_y = dataset_y
    true_x = dataset_x
    pred = model.predict(true_x)

    print("---------------PERFORMANCE ANALYSIS FOR THE MODEL----------------\n")

    # print("Real Test dataset labels: \n{}\n".format(true_y))
    # print("Predicted Test dataset labels: \n{}".format(pred))

    cnf_matrix = confusion_matrix(true_y, pred)
    plt.figure()
    draw_confusion_matrix(cnf_matrix, classes=GENRES, title='Confusion matrix')


def histogram():
    genres_labels = get_genres()
    genres = ['Pop_Rock',
              'Electronic',
              'Rap',
              'Jazz',
              'Latin',
              'RnB',
              'International',
              'Country',
              'Reggae',
              'Blues']

    plt.figure(1)
    plt.title("Distribution of genres in the training data")
    original_bins = np.arange(1, 12) - 0.5
    counts, bins, patches = plt.hist(genres_labels, bins=original_bins, rwidth=0.75)
    plt.xticks(range(1, 11), genres, rotation='vertical')
    plt.yticks(np.arange(0, np.max(counts), 150))
    plt.xlim([0.5, 10.5])
    plt.show()