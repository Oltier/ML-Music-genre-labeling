import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from data import load_data_train_test_data, load_test_data, write_accuracy, write_logloss, \
    load_train_data_with_PCA_per_type, load_train_data_with_PCA_per_type_feature_testing
from visualize import plot_cnf


# logreg = LogisticRegression()
# logreg.fit(train_x, train_y)
# scores = cross_val_score(logreg, train_x, train_y, cv=5, scoring='accuracy')
# print("Cross val accuracy: ", scores.mean(), scores.std())
# print("Training Score: {:.3f}".format(logreg.score(train_x, train_y)))
# print("Test score: {:.3f}".format(logreg.score(test_x, test_y)))


results_logreg = []
rythym = np.arange(0, 265)
for i in rythym:
    train_x, train_y, genres, scaler_rythym, scaler_chroma, scaler_mfcc = load_train_data_with_PCA_per_type_feature_testing(i)
    logreg = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=4000, random_state=0, penalty='l2')
    logreg.fit(train_x, train_y)
    scores = cross_val_score(logreg, train_x, train_y, cv=5, scoring='accuracy')
    print(i, "th column score:", scores.mean())
    results_logreg.append(scores.mean())


max_accuracy_logreg = max(results_logreg)
best_k = 1 + results_logreg.index(max(results_logreg))
print("Max Accuracy is {} on test dataset with {} pca.\n".format(max_accuracy_logreg, best_k))
results_with_index = []
for i, a in enumerate(results_logreg):
    results_with_index.append((i, a))

results_with_index = sorted(results_with_index, key=lambda x: x[1])

results_with_index = np.array(results_with_index)

results_logreg = np.array(results_logreg)
print("Results basic: ", results_logreg)
print("Results sorted", results_with_index)

plt.plot(rythym, results_logreg)
plt.xlabel("Column removed")
plt.ylabel("Accuracy")
#
plt.show()
