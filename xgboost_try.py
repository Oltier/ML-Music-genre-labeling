import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier

from data import load_data_train_test_data, load_test_data, write_accuracy, write_logloss, get_pca
from visualize import plot_cnf

train_x, train_y, test_x, test_y, genres = load_data_train_test_data()

cls = XGBClassifier()
cls.fit(train_x, train_y)
print("Training Score: {:.3f}".format(cls.score(train_x, train_y)))
print("Test score: {:.3f}".format(cls.score(test_x, test_y)))

plot_cnf(cls, test_x, test_y)


test_data = load_test_data()

scaler, pca = get_pca()
test_data = scaler.transform(test_data)
test_data = pca.transform(test_data)

N = test_data.shape[0]
predictions = cls.predict(test_data)

predictions = predictions.reshape((predictions.shape[0], 1))

accuracy_data = predictions.astype(np.uint64)
write_accuracy(accuracy_data)

y_pred = cls.predict_proba(test_data)

write_logloss(y_pred)

# plt.show()
