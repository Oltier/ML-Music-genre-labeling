import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from data import load_data_train_test_data, load_test_data, write_accuracy
from visualize import plot_cnf

train_x, train_y, test_x, test_y, genres = load_data_train_test_data()


# results_knn = []
# for i in range(1, 11):
#     knn = KNeighborsClassifier(n_neighbors=i)
#     knn.fit(train_x, train_y)
#     results_knn.append(knn.score(test_x, test_y))
#
# max_accuracy_knn = max(results_knn)
# best_k = 1 + results_knn.index(max(results_knn))
# print("Max Accuracy is {:.3f} on test dataset with {} neighbors.\n".format(max_accuracy_knn, best_k))
#
# plt.plot(numpy.arange(1, 11), results_knn)
# plt.xlabel("n Neighbors")
# plt.ylabel("Accuracy")

best_k = 9
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(train_x, train_y)
print("Training Score: {:.3f}".format(knn.score(train_x, train_y)))
print("Test score: {:.3f}".format(knn.score(test_x, test_y)))

# plot_cnf(knn, test_x, test_y, genres)


test_data = load_test_data()
N = test_data.shape[0]
predictions = knn.predict(test_data)

ids = np.arange(1, len(predictions) + 1)
# print(len(ids), len(predictions))

predictions = predictions.reshape((predictions.shape[0], 1))
ids = ids.reshape((ids.shape[0], 1))

data = np.append(ids, predictions, axis=1).astype(np.uint64)
write_accuracy(data)


print(len(ids), len(predictions))
# plt.show()
