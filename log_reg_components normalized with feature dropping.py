import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from data import load_test_data, write_accuracy, write_logloss, load_train_data_with_PCA_per_type, \
    load_train_data_with_PCA_per_type_drop_column
from visualize import plot_cnf


ids = np.concatenate((np.arange(0, 168), np.arange(169, 216), np.arange(220, 256)))
results_logreg = []

for i in ids:
    which_column = i
    train_x, train_y, genres, scaler_rythym, scaler_chroma, scaler_mfcc = load_train_data_with_PCA_per_type_drop_column(which_column)

    logreg = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=4000, random_state=0, penalty='l2')
    logreg.fit(train_x, train_y)
    scores = cross_val_score(logreg, train_x, train_y, cv=5, scoring='accuracy')
    print("Cross val accuracy: ", i, scores.mean(), scores.std())
    # print("Training Score: {:.3f}".format(logreg.score(train_x, train_y)))
    results_logreg.append((id, scores.mean()))

results_logreg = sorted(results_logreg, key=lambda x: x[1], reverse=True)

results_logreg = np.array(results_logreg)

print(results_logreg)