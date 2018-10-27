import matplotlib.pyplot as plt
import numpy
from sklearn.neighbors import KNeighborsClassifier

from data import load_data_train_test_data
from visualize import plot_cnf

train_x, train_y, test_x, test_y, genres = load_data_train_test_data()


def knn(train_x, train_y, test_x, test_y):
    results_knn = []
    for i in range(1, 11):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(train_x, train_y)
        results_knn.append(knn.score(test_x, test_y))
    return results_knn


results_knn = knn(train_x, train_y, test_x, test_y)

max_accuracy_knn = max(results_knn)
best_k = 1 + results_knn.index(max(results_knn))
print("Max Accuracy is {:.3f} on test dataset with {} neighbors.\n".format(max_accuracy_knn, best_k))

plt.plot(numpy.arange(1, 11), results_knn)
plt.xlabel("n Neighbors")
plt.ylabel("Accuracy")

knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(train_x, train_y)
print("Training Score: {:.3f}".format(knn.score(train_x, train_y)))
print("Test score: {:.3f}".format(knn.score(test_x, test_y)))

plot_cnf(knn, test_x, test_y, genres)

# plt.show()
