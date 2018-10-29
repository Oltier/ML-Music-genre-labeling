import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt

from data import load_data_train_test_data, load_test_data, write_accuracy, write_logloss
from visualize import plot_cnf

train_x, train_y, test_x, test_y, genres = load_data_train_test_data()

svm = SVC(C=100, gamma=0.2, probability=True)
svm.fit(train_x, train_y)
print("Training Score: {:.3f}".format(svm.score(train_x, train_y)))
print("Test score: {:.3f}".format(svm.score(test_x, test_y)))

plot_cnf(svm, test_x, test_y)

test_data = load_test_data()
N = test_data.shape[0]
predictions = svm.predict(test_data)

predictions = predictions.reshape((predictions.shape[0], 1))

accuracy_data = predictions.astype(np.uint64)
write_accuracy(accuracy_data)

y_pred = svm.predict_proba(test_data)

write_logloss(y_pred)

# plt.show()
