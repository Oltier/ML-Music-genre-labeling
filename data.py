import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

train_data_path = "data/train_data.csv"
labels_path = "data/train_labels.csv"
test_data_path = "data/test_data.csv"


def load_data_train_test_data():
    genres_labels = np.array(pd.read_csv(labels_path, index_col=False, header=None))
    genres = range(1, 11)
    data_set = np.array(pd.read_csv(train_data_path, index_col=False))
    data_set = np.append(data_set, genres_labels, 1)
    number_of_cols = data_set.shape[1]
    train, test = train_test_split(data_set, test_size=0.85, random_state=2, stratify=data_set[:, number_of_cols - 1])
    train_x = train[:, :number_of_cols - 1]
    train_y = train[:, number_of_cols - 1]

    test_x = test[:, :number_of_cols - 1]
    test_y = test[:, number_of_cols - 1]
    return train_x, train_y, test_x, test_y, genres


def load_test_data():
    return np.array(pd.read_csv(test_data_path, index_col=False))


def load_train_data():
    return np.array(pd.read_csv(train_data_path, index_col=False))


def write_accuracy(data):
    headers = ['Sample_label']
    accuary_result_path = "results/accuracy.csv"
    df = pd.DataFrame(data)
    df.index = np.arange(1, len(df) + 1)
    df.to_csv(accuary_result_path, header=headers, index=True, index_label='Sample_id')
    print("Saved accuracy.csv")


def write_logloss(data):
    headers = list(map(lambda i: "Class_" + str(i), list(range(1, 11))))
    logloss_result_path = "results/logloss.csv"
    df = pd.DataFrame(data)
    df.index = np.arange(1, len(df) + 1)
    df.to_csv(logloss_result_path, header=headers, index=True, index_label='Sample_id')
    print("Saved logloss.csv")


def get_genres():
    return np.array(pd.read_csv(labels_path, index_col=False, header=None))
