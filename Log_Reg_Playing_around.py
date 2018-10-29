import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from data import load_data_train_test_data, load_test_data, write_accuracy, write_logloss, get_pca
from visualize import plot_cnf


# logreg = LogisticRegression()
# logreg.fit(train_x, train_y)
# scores = cross_val_score(logreg, train_x, train_y, cv=5, scoring='accuracy')
# print("Cross val accuracy: ", scores.mean(), scores.std())
# print("Training Score: {:.3f}".format(logreg.score(train_x, train_y)))
# print("Test score: {:.3f}".format(logreg.score(test_x, test_y)))


results_logreg = []
n_tests = np.arange(1, 251)
for i in n_tests:
    print(i, "th pca")
    train_x, train_y, test_x, test_y, genres = load_data_train_test_data(i)
    logreg = LogisticRegression(solver='liblinear', multi_class='ovr')
    logreg.fit(train_x, train_y)
    results_logreg.append(logreg.score(test_x, test_y))

max_accuracy_logreg = max(results_logreg)
best_k = 1 + results_logreg.index(max(results_logreg))
print("Max Accuracy is {:.3f} on test dataset with {} neighbors.\n".format(max_accuracy_logreg, best_k))

plt.plot(n_tests, results_logreg)
plt.xlabel("n PCA")
plt.ylabel("Accuracy")
#
plt.show()
