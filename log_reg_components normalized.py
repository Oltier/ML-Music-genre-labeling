import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from data import load_test_data, write_accuracy, write_logloss, load_train_data_with_PCA_per_type
from visualize import plot_cnf

train_x, train_y, genres, scaler_rythym, scaler_chroma, scaler_mfcc = load_train_data_with_PCA_per_type()

logreg = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=4000, random_state=0, penalty='l2')
logreg.fit(train_x, train_y)
scores = cross_val_score(logreg, train_x, train_y, cv=5, scoring='accuracy')
print("Cross val accuracy: ", scores.mean(), scores.std())
print("Training Score: {:.3f}".format(logreg.score(train_x, train_y)))
# print("Test score: {:.3f}".format(logreg.score(test_x, test_y)))

# TODO Ha kéne confusion matrix, akkor előkapjuk ezt
# plot_cnf(logreg, test_x, test_y)

test_data = load_test_data()

rythym = test_data[:, :168]
chroma = test_data[:, 168:216]
mfcc = test_data[:, 216:]

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
predictions = logreg.predict(test_data)

predictions = predictions.reshape((predictions.shape[0], 1))

accuracy_data = predictions.astype(np.uint64)
write_accuracy(accuracy_data)

y_pred = logreg.predict_proba(test_data)

write_logloss(y_pred)

# plt.show()
