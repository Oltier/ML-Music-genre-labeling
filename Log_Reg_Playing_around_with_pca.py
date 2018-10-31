import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from data import load_data_train_test_data, load_test_data, write_accuracy, write_logloss, \
    load_train_data_with_PCA_per_type
from visualize import plot_cnf


# logreg = LogisticRegression()
# logreg.fit(train_x, train_y)
# scores = cross_val_score(logreg, train_x, train_y, cv=5, scoring='accuracy')
# print("Cross val accuracy: ", scores.mean(), scores.std())
# print("Training Score: {:.3f}".format(logreg.score(train_x, train_y)))
# print("Test score: {:.3f}".format(logreg.score(test_x, test_y)))


results_logreg = []
tol = np.linspace(1, 0, 10)
for i in tol:
    print("C: ", i)
    train_x, train_y, test_x, test_y, genres, scaler_rythym, scaler_chroma, scaler_mfcc = load_train_data_with_PCA_per_type()
    logreg = LogisticRegression(tol=i, solver='lbfgs', multi_class='multinomial', max_iter=1000, random_state=0)
    logreg.fit(train_x, train_y)
    scores = cross_val_score(logreg, train_x, train_y, cv=5, scoring='accuracy')
    results_logreg.append(scores.mean())


max_accuracy_logreg = max(results_logreg)
best_k = 1 + results_logreg.index(max(results_logreg))
print("Max Accuracy is {:.3f} on test dataset with {} pca.\n".format(max_accuracy_logreg, best_k))

plt.plot(tol, results_logreg)
plt.xlabel("C")
plt.ylabel("Accuracy")
#
plt.show()
