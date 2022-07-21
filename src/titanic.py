import pandas as pd
import os
import platform
import titanic_preprocessing

splitter = '\\' if platform.system() == 'Windows' else '/'
pwd = os.path.dirname(os.getcwd()) + splitter

if 'titanic' not in pwd:
	pwd = pwd + 'titanic_project' + splitter

trd = pd.read_csv(pwd + f'data{splitter}train.csv')
gsc = pd.read_csv(pwd + f'data{splitter}gender_submission.csv')
trd.append(gsc)
preprocessing_data = titanic_preprocessing.preprocessing(trd)

test_data = pd.read_csv(pwd + f'data{splitter}test.csv')
