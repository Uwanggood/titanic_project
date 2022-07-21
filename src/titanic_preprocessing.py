"""
Preprocesses the titanic dataframe.
"""
import os
import platform

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from spyder_kernels.utils.lazymodules import pandas


def preprocessing(trd):
	#  column list
	# 'PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
	# 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'
	pd.set_option('display.max_row', 100)
	print(trd.columns)

	ctrd = pandas.DataFrame.copy(trd)
	title = trd.Name.str.split(',').str[-1].str.split('.').str[0]
	""""
	 독특한 호칭이 존재함
	 Mr              517 남성(1)
	 Miss            182 미혼여성(2)
	 Mrs             125 기혼자여성(3)
	
	 위의 세개를 제외하고 나머지는 
	 이름의 호칭을 성별에 따라 위의 클래스(숫자) 로 변경
	 Master           40 나리 ,주인님 
	 Dr                7 의사 
	 Rev               6 목사
	 Mlle              2 여자아이 미드무아젤
	 Major             2 소령
	 Col               2 대령
	 the Countess      1 백작부인
	 Capt              1 캡틴
	 Ms                1 여자
	 Sir               1 경칭
	 Lady              1 여성
	 Mme               1 마담 호칭
	 Don               1 두목 호칭
	 Jonkheer          1 귀족 호칭
		
	"""
	Miss_age_mean = int(round(trd[trd.Name.str.contains(' Miss.')].Age.mean()))
	ctrd.loc[ctrd.Name.str.contains(' Mr.', na=False), 'Name'] = 1
	ctrd.loc[ctrd.Name.str.contains(' Miss.', na=False), 'Name'] = 2
	ctrd.loc[ctrd.Name.str.contains(' Mrs.', na=False), 'Name'] = 3

	another_title_list = [
		' Master.',
		' Dr.',
		' Rev.',
		' Mlle.',
		' Major.',
		' Col.',
		' the Countess.',
		' Capt.',
		' Ms.',
		' Sir.',
		' Lady.',
		' Mme.',
		' Don.',
		' Jonkheer.',
	]
	for another_title in another_title_list:
		ctrd.loc[
			(ctrd.Name.str.contains(another_title, na=False)) & (ctrd.Sex == 'male'), 'Name'] = 1
		ctrd.loc[(ctrd.Name.str.contains(another_title, na=False)) & (ctrd.Sex == 'female') & (
			ctrd.Age < Miss_age_mean), 'Name'] = 2
		ctrd.loc[(ctrd.Name.str.contains(another_title, na=False)) & (ctrd.Sex == 'female') & (
			ctrd.Age > Miss_age_mean), 'Name'] = 3


	'''
	SibSp(Drop) 
	분석 결과 SibSp가 세명이상인 경우는 많이 없으므로 표본이 되지 못함 
	0, 1 ,2의 경우 성별을 따라가는 경향이 좀 더 강하므로 SibSp는 생존률에 별 도움이 되지 않는다고 판단한다. 
	'''

	# ctrd.drop(['SibSp'], axis=1, inplace=True)

	'''
	Parch - 탑승한 부모 자식수
	'''



	# 가설 1. 남성이 여성보다 많이 죽었을것이다.
	# 가설 2. 어린아이, 노인의 생존률이 더 높았을 것이다. (검증 결과 어린아이의 생존률만 높음)
	# 가설 1과 2를 조합하여 나이대별로도 클래스를 구분하고 싶지만
	# 실제로 그러기는 어려우므로 남자와 여자로 구분한다.

	# 가설 1 남자 생존률 18% , 여자 생존률 74%.
	print(trd[trd.Sex == 'male'].Survived.mean())
	print(trd[trd.Sex == 'female'].Survived.mean())

	# 가설 2
	# 5세 이하 생존률 70% , 44명.
	# print(trd[trd.Age.between(0, 5)].Survived.mean())

	# 6~ 10세 이하 생존률 35% : 20명.
	# print(trd[trd.Age.between(6, 10)].Survived.mean())

	# 가설 2를 증명하기 위해 그래프를 그린다.
	# 문제 1 : 나이에 null이 있는 사람들이 있음.
	# 문제 2 : 나이가 소수점인 사람이 있음.

	# 해결 1 : 나이에 null인 사람을 평균으로 맞춘다.
	# Pclass와 사망에 따른 평균으로 바꿔놓는다.

	for data in [[1, 0], [1, 1], [2, 0], [2, 1], [3, 0], [3, 1]]:
		pclass = data[0]
		survived = data[1]
		na_option = (trd.Age.isna()) & (trd.Pclass == pclass) & (trd.Survived == survived)
		fill_option = (trd.Pclass == pclass) & (trd.Survived == survived) & (trd.Age.notna())
		ctrd.loc[na_option, 'Age'] = round(ctrd[na_option].Age.fillna(ctrd[fill_option].Age.mean()))

	# 해결 2 : 나이가 소수점인 사람들은 올림처리.
	ctrd.loc[ctrd.Age - ctrd.Age.round(0) != 0, 'Age'] = \
		ctrd[ctrd.Age - ctrd.Age.round(0) != 0].Age.apply(np.ceil)

	age_list = list(range(int(ctrd.Age.min()), int(ctrd.Age.max()) + 2, 10))
	survived_rate_list = []

	for idx, age in enumerate(age_list):
		if idx == 0:
			continue
		survived_rate_list.append(ctrd[ctrd['Age'].between(age_list[idx - 1], age)].Survived.mean())

	# plt.plot(age_list[1:], survived_rate_list)
	# plt.show()

	# 가설 2:
	# 어린아이는 생존률이 높지만 나이든 사람의 생존률은 높지 않다.

	return ctrd


splitter = '\\' if platform.system() == 'Windows' else '/'
pwd = os.path.dirname(os.getcwd()) + splitter

if 'titanic' not in pwd:
	pwd = pwd + 'titanic_project' + splitter

trd = pd.read_csv(pwd + f'data{splitter}train.csv')
gsc = pd.read_csv(pwd + f'data{splitter}gender_submission.csv')
trd.append(gsc)
ctrd = preprocessing(trd)
