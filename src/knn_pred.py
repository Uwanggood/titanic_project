from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


def knn_prediction(x_train, x_test, y_train, y_test):
	clf_knn = KNeighborsClassifier(n_neighbors=2)
	clf_knn.fit(x_train, y_train)

	knn_pred = clf_knn.predict(x_test)
	score = accuracy_score(y_test, knn_pred)
	return knn_pred, score
