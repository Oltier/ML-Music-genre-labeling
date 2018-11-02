import itertools

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from data import get_genres, visualisation_data

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


def draw_confusion_matrix(cm: np.ndarray, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    # Normalize
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    fmt = '.2f'

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def plot_cnf(model, true_x, true_y):
    pred = model.predict(true_x)
    cnf_matrix = confusion_matrix(true_y, pred)
    fig = plt.figure(figsize=(5,5))
    draw_confusion_matrix(cnf_matrix, classes=genres, title=r"$\bf{Figure\ 4.}$Confusion matrix")


def histogram():
    genres_labels = get_genres()

    plt.figure(1)
    plt.title(r"$\bf{Figure\ 1.}$Distribution of genres")
    original_bins = np.arange(1, 12) - 0.5
    counts, bins, patches = plt.hist(genres_labels, bins=original_bins, rwidth=0.75)
    plt.xticks(range(1, 11), genres, rotation='vertical')
    plt.yticks(np.arange(0, np.max(counts), 150))
    plt.xlim([0.5, 10.5])


def plot_data():
    training_data_set, genres_labels, genres = visualisation_data()
    fig = plt.figure(figsize=(10, 12))
    fig.suptitle('Representatives by class', fontsize=16, y=1)
    fig.tight_layout(rect=[0, 0.03, -1, 0.95])
    for i in genres:
        plt.subplot(5, 2, i)
        i_th_label = np.where(genres_labels == i)[0]
        i_th_data = training_data_set[i_th_label]
        plt.title("Representatives of class {}".format(i))
        plt.scatter(i_th_data[:, 0], i_th_data[:, 1], )
    plt.subplots_adjust(hspace=0.3)
    plt.show()

# plot_data()
