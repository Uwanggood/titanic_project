import pandas as pd
import os
import platform
import titanic_preprocessing
import knn_pred
import linear_regression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import cnn_pred

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

'''
knn 사용
'''
# pred, score = knn_pred.knn_prediction(preprocessing_data, preprocessing_test_data, y_train, y_test)
'''
linear regression 사용
'''
# pred_lr, pred_ridge, pred_lasso, pred_elastic_net = linear_regression.pred_linear_regression(
# 	preprocessing_data, preprocessing_test_data, y_train, y_test)
#
# pred = pred_lasso

'''
cnn 사용
'''
preprocessing_data.drop(['PassengerId'], axis=1, inplace=True)
preprocessing_test_data.drop(['PassengerId'], axis=1, inplace=True)
pred = cnn_pred.cnn_pred(preprocessing_data, preprocessing_test_data, y_train, y_test)
tst['Survived'] = pred.tolist()
result = tst[['PassengerId', 'Survived']]

# result = pd.concat([preprocessing_test_data, pd.DataFrame(pred, columns=['Survived'])], axis=1)[
# 	['PassengerId', 'Survived']]
result.to_csv(pwd + f'results{splitter}result.csv', index=False)
