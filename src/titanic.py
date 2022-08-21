import pandas as pd
import os
import platform
import titanic_preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

splitter = '\\' if platform.system() == 'Windows' else '/'
pwd = os.path.dirname(os.getcwd()) + splitter

if 'titanic' not in pwd:
	pwd = pwd + 'titanic_project' + splitter

trd = pd.read_csv(pwd + f'data{splitter}train.csv')
y_test = pd.read_csv(pwd + f'data{splitter}gender_submission.csv').Survived
tst = pd.read_csv(pwd + f'data{splitter}test.csv')

preprocessing_data = titanic_preprocessing.preprocessing(trd)
preprocessing_test_data = titanic_preprocessing.preprocessing(tst)
y_train = preprocessing_data.Survived
preprocessing_data.drop(['Survived'], axis=1, inplace=True)

clf_knn = KNeighborsClassifier(n_neighbors=2)
clf_knn.fit(preprocessing_data, y_train)

knn_pred = clf_knn.predict(preprocessing_test_data)
score = accuracy_score(y_test, knn_pred)

result = pd.concat([preprocessing_test_data, pd.DataFrame(knn_pred, columns=['Survived'])], axis=1)[['PassengerId', 'Survived']]
result.to_csv(pwd + f'results{splitter}knn_result.csv', index=False)

