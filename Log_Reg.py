import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from data import load_data_train_test_data, load_test_data, write_accuracy, write_logloss
from visualize import plot_cnf

train_x, train_y, test_x, test_y, genres, scaler, pca = load_data_train_test_data()


logreg = LogisticRegression(solver='liblinear', multi_class='ovr')
logreg.fit(train_x, train_y)
scores = cross_val_score(logreg, train_x, train_y, cv=5, scoring='accuracy')
print("Cross val accuracy: ", scores.mean(), scores.std())
print("Training Score: {:.3f}".format(logreg.score(train_x, train_y)))
print("Test score: {:.3f}".format(logreg.score(test_x, test_y)))

plot_cnf(logreg, test_x, test_y)


test_data = load_test_data()

test_data = scaler.transform(test_data)
test_data = pca.transform(test_data)

N = test_data.shape[0]
predictions = logreg.predict(test_data)

predictions = predictions.reshape((predictions.shape[0], 1))

accuracy_data = predictions.astype(np.uint64)
write_accuracy(accuracy_data)

y_pred = logreg.predict_proba(test_data)

write_logloss(y_pred)

# plt.show()
