import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from data import load_test_data, write_accuracy, write_logloss, \
    load_train_data_with_PCA_per_type
from visualize import plot_cnf

train_x, train_y, test_x, test_y, genres, scaler_rythym, scaler_chroma, scaler_mfcc = load_train_data_with_PCA_per_type()

# results_knn = []
# n_tests = np.arange(1, 50)
# for i in n_tests:
#     print(i, "th neighbor")
#     knn = KNeighborsClassifier(n_neighbors=i)
#     knn.fit(train_x, train_y)
#     results_knn.append(knn.score(test_x, test_y))
#
# max_accuracy_knn = max(results_knn)
# best_k = 1 + results_knn.index(max(results_knn))
# print("Max Accuracy is {:.3f} on test dataset with {} neighbors.\n".format(max_accuracy_knn, best_k))
#
# plt.plot(n_tests, results_knn)
# plt.xlabel("n Neighbors")
# plt.ylabel("Accuracy")

# plt.show()

best_k = 26
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(train_x, train_y)
scores = cross_val_score(knn, train_x, train_y, cv=5, scoring='accuracy')
print("Cross val accuracy: ", scores.mean(), scores.std())
preds = knn.predict_proba(test_x)
preds = np.argmax(preds, axis=-1)
print('Test Set F-score =  {0:.3f}'.format(f1_score(test_y, preds, average='weighted')))
predictions_on_train = knn.predict_proba(train_x)
log_loss_score = log_loss(train_y, predictions_on_train, eps=1e-15)
print("Train logloss:", log_loss_score)

plot_cnf(knn, test_x, test_y)

test_data = load_test_data()

rythym = test_data[:, :168]
chroma = test_data[:, 169:216]
mfcc = test_data[:, 217:]

rythym = scaler_rythym.fit_transform(rythym)
chroma = scaler_chroma.fit_transform(chroma)
mfcc = scaler_mfcc.fit_transform(mfcc)

rythym = preprocessing.normalize(rythym, norm='l2')
chroma = preprocessing.normalize(chroma, norm='l2')
mfcc = preprocessing.normalize(mfcc, norm='l2')

# rythym = pca_rythym.fit_transform(rythym)
# chroma = pca_chroma.fit_transform(chroma)
# mfcc = pca_mfcc.fit_transform(mfcc)

test_data = np.concatenate((rythym, chroma, mfcc), axis=1)

N = test_data.shape[0]
predictions = knn.predict(test_data)

predictions = predictions.reshape((predictions.shape[0], 1))

accuracy_data = predictions.astype(np.uint64)
write_accuracy(accuracy_data)

y_pred = knn.predict_proba(test_data)

write_logloss(y_pred)

plt.show()
