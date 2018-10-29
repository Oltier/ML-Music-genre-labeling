import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier, GradientBoostingRegressor, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from data import load_data_train_test_data, load_test_data, write_accuracy, write_logloss, get_pca
from visualize import plot_cnf

train_x, train_y, test_x, test_y, genres = load_data_train_test_data()


clf1 = DecisionTreeClassifier(max_depth=4)
clf2 = KNeighborsClassifier(n_neighbors=7)
clf3 = SVC(gamma='scale', kernel='rbf', probability=True)
eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2), ('svc', clf3)], voting='soft', weights=[2,1,2])

clf1 = clf1.fit(train_x, train_y)
clf2 = clf2.fit(train_x, train_y)
clf3 = clf3.fit(train_x, train_y)
eclf = eclf.fit(train_x, train_y)

plot_cnf(eclf, test_x, test_y)


for clf, label in zip([clf1, clf2, clf3, eclf], ['Logistic Regression', 'Random Forest', 'naive Bayes', 'Ensemble']):
    scores = cross_val_score(clf, train_x, train_y, cv=5, scoring='accuracy')
    print("Accuracy: ", scores.mean(), scores.std(), label)

test_data = load_test_data()

scaler, pca = get_pca()
test_data = scaler.transform(test_data)
test_data = pca.transform(test_data)

N = test_data.shape[0]
predictions = eclf.predict(test_data)

predictions = predictions.reshape((predictions.shape[0], 1))

accuracy_data = predictions.astype(np.uint64)
write_accuracy(accuracy_data)

y_pred = eclf.predict_proba(test_data)

write_logloss(y_pred)

# plt.show()
