import pandas
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

pwd = os.getcwd()

# survived data
gender_submission_data = pd.read_csv(pwd + '/data/gender_submission.csv')
# train_data
trd = pd.read_csv(pwd + '/data/train.csv')
trd.append(gender_submission_data)
# copied train_data
ctrd = pandas.DataFrame.copy(trd)
# test_data
ted = pd.read_csv(pwd + '/data/test.csv')
# copied test_data
cted = pandas.DataFrame.copy(ted)

# 'PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
# 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'
print(trd.columns)

title = trd.Name.str.split(',').str[-1].str.split('.').str[0]

# 독특한 호칭이 존재함
# Mr              517 남성
# Miss            182 미혼여성
# Mrs             125 기혼자여성
# Master           40 나리 ,주인님
# Dr                7 의사
# Rev               6 목사
# Mlle              2 여자아이 미드무아젤
# Major             2 소령
# Col               2 대령
# the Countess      1 백작부인
# Capt              1 캡틴
# Ms                1 여자
# Sir               1 경칭
# Lady              1 여성
# Mme               1 마담 호칭
# Don               1 두목 호칭
# Jonkheer          1 귀족 호칭

# 가설 1. 남성이 여성보다 많이 죽었을것이다.
# 가설 2. 어린아이, 노인의 생존률이 더 높았을 것이다.
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
	ctrd.loc[na_option, 'Age'] = ctrd[na_option].Age.fillna(ctrd[fill_option].Age.mean().round(0))

# 해결 2 : 나이가 소수점인 사람들은 올림처리.
ctrd.loc[ctrd.Age - ctrd.Age.round(0) != 0, 'Age'] = \
	ctrd[ctrd.Age - ctrd.Age.round(0) != 0].Age.apply(np.ceil)

for i in range(ctrd.Age.min(), ctrd.Age.max()):
	plt.plot(i, ctrd[ctrd.Age.between(i, i + 10)].Survived, 'o')
	plt.show()

# print(title.value_counts())
# 
# print(trd)
# print(ted)
# print(gender_submission_data)
