import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

train_data_path = "train_data.csv"
labels_path = "train_labels.csv"
test_data_path = "test_data.csv"

def load_data_train_test_data():
    genres_labels = np.array(pd.read_csv(labels_path, index_col=False, header=None))
    genres = range(1, 11)
    training_data_set = np.array(pd.read_csv(train_data_path, index_col=False, header=None))
    pca = PCA(79)
    scaler = StandardScaler()
    training_data_set = scaler.fit_transform(training_data_set)
    training_data_set = pca.fit_transform(training_data_set)
    training_data_set = np.append(training_data_set, genres_labels, 1)
    number_of_cols = training_data_set.shape[1]
    train, test = train_test_split(training_data_set, test_size=0.25, random_state=12,
                                   stratify=training_data_set[:, number_of_cols - 1])
    train_x = train[:, :number_of_cols - 1]
    train_y = train[:, number_of_cols - 1]

    # sm = SMOTE()
    # x_train_res, y_train_res = sm.fit_resample(train_x, train_y)

    # train_x = preprocessing.normalize(train_x, norm='l2')

    test_x = test[:, :number_of_cols - 1]
    test_y = test[:, number_of_cols - 1]

    return train_x, train_y, test_x, test_y, genres, scaler, pca


def load_train_data_rythym_only():
    genres_labels = np.array(pd.read_csv(labels_path, index_col=False, header=None))
    genres = range(1, 11)
    training_data_set = np.array(pd.read_csv(train_data_path, index_col=False, header=None))[:, :168]
    pca = PCA(100)
    scaler = StandardScaler()
    training_data_set = scaler.fit_transform(training_data_set)
    # training_data_set = preprocessing.normalize(training_data_set, norm='l2')
    training_data_set = pca.fit_transform(training_data_set)
    training_data_set = np.append(training_data_set, genres_labels, 1)
    number_of_cols = training_data_set.shape[1]
    train, test = train_test_split(training_data_set, test_size=0.25, random_state=12,
                                   stratify=training_data_set[:, number_of_cols - 1])
    train_x = train[:, :number_of_cols - 1]
    train_y = train[:, number_of_cols - 1]

    # sm = SMOTE()
    # x_train_res, y_train_res = sm.fit_resample(train_x, train_y)

    # train_x = preprocessing.normalize(train_x, norm='l2')

    test_x = test[:, :number_of_cols - 1]
    test_y = test[:, number_of_cols - 1]

    return train_x, train_y, test_x, test_y, genres, scaler, pca


def load_train_data_chroma_only():
    genres_labels = np.array(pd.read_csv(labels_path, index_col=False, header=None))
    genres = range(1, 11)
    training_data_set = np.array(pd.read_csv(train_data_path, index_col=False, header=None))[:, 169:216]
    pca = PCA(40)
    scaler = StandardScaler()
    training_data_set = scaler.fit_transform(training_data_set)
    training_data_set = pca.fit_transform(training_data_set)
    training_data_set = np.append(training_data_set, genres_labels, 1)
    number_of_cols = training_data_set.shape[1]
    train, test = train_test_split(training_data_set, test_size=0.25, random_state=12,
                                   stratify=training_data_set[:, number_of_cols - 1])
    train_x = train[:, :number_of_cols - 1]
    train_y = train[:, number_of_cols - 1]

    # sm = SMOTE()
    # x_train_res, y_train_res = sm.fit_resample(train_x, train_y)

    # train_x = preprocessing.normalize(train_x, norm='l2')

    test_x = test[:, :number_of_cols - 1]
    test_y = test[:, number_of_cols - 1]

    return train_x, train_y, test_x, test_y, genres, scaler, pca


def load_train_data_MFCC_only():
    genres_labels = np.array(pd.read_csv(labels_path, index_col=False, header=None))
    genres = range(1, 11)
    training_data_set = np.array(pd.read_csv(train_data_path, index_col=False, header=None))[:, 217:]
    pca = PCA(40)
    scaler = StandardScaler()
    training_data_set = scaler.fit_transform(training_data_set)
    training_data_set = pca.fit_transform(training_data_set)
    training_data_set = np.append(training_data_set, genres_labels, 1)
    number_of_cols = training_data_set.shape[1]
    train, test = train_test_split(training_data_set, test_size=0.25, random_state=12,
                                   stratify=training_data_set[:, number_of_cols - 1])
    train_x = train[:, :number_of_cols - 1]
    train_y = train[:, number_of_cols - 1]

    # sm = SMOTE()
    # x_train_res, y_train_res = sm.fit_resample(train_x, train_y)

    # train_x = preprocessing.normalize(train_x, norm='l2')

    test_x = test[:, :number_of_cols - 1]
    test_y = test[:, number_of_cols - 1]

    return train_x, train_y, test_x, test_y, genres, scaler, pca


def load_train_data_with_PCA_per_type():
    genres_labels = np.array(pd.read_csv(labels_path, index_col=False, header=None))
    genres_labels = genres_labels.reshape((genres_labels.shape[0],))
    genres = range(1, 11)
    training_data_set = np.array(pd.read_csv(train_data_path, index_col=False, header=None))

    rythym = training_data_set[:, :168]
    chroma = training_data_set[:, 169:216]
    mfcc = training_data_set[:, 220:]

    # pca_rythym = PCA(0.8)
    # pca_chroma = PCA(0.8)
    # pca_mfcc = PCA(0.8)

    scaler_rythym = StandardScaler()
    scaler_chroma = StandardScaler()
    scaler_mfcc   = StandardScaler()

    rythym = scaler_rythym.fit_transform(rythym)
    chroma = scaler_chroma.fit_transform(chroma)
    mfcc = scaler_mfcc.fit_transform(mfcc)

    rythym = preprocessing.normalize(rythym, norm='l2')
    chroma = preprocessing.normalize(chroma, norm='l2')
    mfcc = preprocessing.normalize(mfcc, norm='l2')

    # rythym = pca_rythym.fit_transform(rythym)
    # chroma = pca_chroma.fit_transform(chroma)
    # mfcc = pca_mfcc.fit_transform(mfcc)

    training_data_set = np.concatenate((rythym, chroma, mfcc), axis=1)

    # sm = SMOTE()
    # train_x, train_y = sm.fit_resample(train_x, train_y)

    # train_x = preprocessing.normalize(train_x, norm='l2')

    return training_data_set, genres_labels, genres, scaler_rythym, scaler_chroma, scaler_mfcc \
           # pca_rythym, pca_chroma, pca_mfcc


def load_test_data():
    return np.array(pd.read_csv(test_data_path, index_col=False, header=None))


def load_train_data():
    return np.array(pd.read_csv(train_data_path, index_col=False, header=None))


def write_accuracy(data):
    headers = ['Sample_label']
    accuracy_result_path = "results/accuracy.csv"
    df = pd.DataFrame(data)
    df.index = np.arange(1, len(df) + 1)
    df.to_csv(accuracy_result_path, header=headers, index=True, index_label='Sample_id')
    print("Saved accuracy.csv")


def write_logloss(data):
    headers = list(map(lambda i: "Class_" + str(i), list(range(1, 11))))
    log_loss_result_path = "results/logloss.csv"
    df = pd.DataFrame(data)
    df.index = np.arange(1, len(df) + 1)
    df.to_csv(log_loss_result_path, header=headers, index=True, index_label='Sample_id')
    print("Saved logloss.csv")


def get_genres():
    return np.array(pd.read_csv(labels_path, index_col=False, header=None))
