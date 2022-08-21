"""
Preprocesses the titanic dataframe.
"""
import os
import platform

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


#  column list
# 'PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
# 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'\
# pd.set_option('display.max_row', 500)
# pd.set_option('display.max_columns', 10)
def preprocessing(trd):
	split_name = trd.Name.str.split(',')
	ctrd = pd.DataFrame.copy(trd)
	title = split_name.str[1].str.split('.').str[0]
	family_name = split_name.str[0]
	unique_title = another_title_list = title.unique()

	another_title_list = np.delete(another_title_list, np.argwhere(another_title_list == ' Mr'))
	another_title_list = np.delete(another_title_list, np.argwhere(another_title_list == ' Mrs'))
	another_title_list = np.delete(another_title_list, np.argwhere(another_title_list == ' Miss'))


	# 가설 1 남자 생존률 18% , 여자 생존률 74%.
	# print(trd[trd.Sex == 'male'].Survived.mean())
	# print(trd[trd.Sex == 'female'].Survived.mean())

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

	# -> change -> Survived와 관련되어 삭제

	for data in [1, 2, 3]:
		pclass = data
		na_option = (trd.Age.isna()) & (trd.Pclass == pclass)
		fill_option = (ctrd.Pclass == pclass) & (ctrd.Age.notna())
		ctrd.loc[na_option, 'Age'] = round(ctrd[na_option].Age.fillna(ctrd[fill_option].Age.mean()))

	# 해결 2 : 나이가 소수점인 사람들은 올림처리.
	ctrd.loc[ctrd.Age - ctrd.Age.round(0) != 0, 'Age'] = \
		ctrd[ctrd.Age - ctrd.Age.round(0) != 0].Age.apply(np.ceil)

	# age_list = list(range(int(ctrd.Age.min()), int(ctrd.Age.max()) + 2, 10))
	# survived_rate_list = []
	#
	# for idx, age in enumerate(age_list):
	# 	if idx == 0:
	# 		continue
	# survived_rate_list.append(ctrd[ctrd['Age'].between(age_list[idx - 1], age)].Survived.mean())

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

	for another_title in another_title_list:
		ctrd.loc[
			(ctrd.Name.str.contains(another_title, na=False)) & (ctrd.Sex == 'male'), 'Name'] = 1
		ctrd.loc[(ctrd.Name.str.contains(another_title, na=False)) & (ctrd.Sex == 'female') & (
			ctrd.Age < Miss_age_mean), 'Name'] = 2
		ctrd.loc[(ctrd.Name.str.contains(another_title, na=False)) & (ctrd.Sex == 'female') & (
			ctrd.Age >= Miss_age_mean), 'Name'] = 3

	'''
	Sex 
	남자 0 여자 1 
	'''
	ctrd.loc[ctrd.Sex == 'male', 'Sex'] = 0
	ctrd.loc[ctrd.Sex == 'female', 'Sex'] = 1
	'''
	SibSp 와 Parch를 분석하기에 앞서 성씨와 호를 나눠서 컬럼으로 만든다.
	SibSp - 형제자매, 배우자 
	SibSp가 0인것은 0 나머지는  1로
	has_family_list = trd[
		(trd.Name.str.split(',').str[0].isin(family_name[family_name.groupby(family_name)
											 .transform('count') > 1])) &
		((trd.Parch > 0) |
		 (trd.SibSp > 0))
		]
	has_family_list[(has_family_list.SibSp ==1) & (has_family_list.Pclass == 3) & (has_family_list.Name.str.contains("\)")) ].Name.str.split(',').str[0].apply(lambda x : trd[trd.Name.str.contains(x)].Survived)
	'''

	ctrd.loc[(ctrd.SibSp > 0), 'SibSp'] = 1

	# ctrd.drop(['SibSp'], axis=1, inplace=True)

	'''
	Parch - 탑승한 부모 자식수
	Parch가 0인것은 0 나머지는 1로
	'''

	ctrd.loc[(ctrd.Parch > 0), 'Parch'] = 1

	# ctrd.drop(['Parch'], axis=1, inplace=True)

	'''
	Ticket - 티켓 번호
	티켓은... Pclass가 있으므로 삭제
	ctrd[(ctrd['Pclass'] == 1) & (ctrd.Ticket.str.contains("[a-zA-Z]"))]\
	.Ticket.str.split(' ').str[0].unique()
	
	1등석 : 	
	array(['PC', 'W.E.P.', 'WE/P', 'F.C.'], dtype=object)
	2등석 : 
	array(['C.A.', 'SC/Paris', 'S.O.C.', 'SO/C', 'SC/PARIS', 'S.O.P.',
	'F.C.C.', 'W/C', 'SW/PP', 'SCO/W', 'W./C.', 'P/PP', 'SC', 'SC/AH',
	'S.W./PP', 'S.O./P.P.', 'S.C./PARIS', 'C.A./SOTON'], dtype=object)
	3등석 :
	array(['A/5', 'STON/O2.', 'PP', 'A/5.', 'A./5.', 'S.C./A.4.', 'A/4.',
	'CA', 'S.P.', 'W./C.', 'SOTON/OQ', 'C.A.', 'STON/O', 'A4.', 'C',
	'SOTON/O.Q.', 'A.5.', 'Fa', 'CA.', 'LINE', 'A/S', 'A/4',
	'S.O./P.P.', 'SOTON/O2'], dtype=object)
	
	
	only_num_regex = "^\d+$"
	class1_ticket_number = ctrd[(ctrd['Pclass'] == 1) & (ctrd.Ticket.str.contains(only_num_regex))][['Survived', 'Ticket']]
	class1_ticket_alpha = ctrd[
		(ctrd['Pclass'] == 1) & (~ctrd.Ticket.str.contains(only_num_regex))][['Survived', 'Ticket']]
	
	class2_ticket_number = ctrd[
		(ctrd['Pclass'] == 2) & (ctrd.Ticket.str.contains(only_num_regex))][['Survived', 'Ticket']]
	class2_ticket_alpha = ctrd[
		(ctrd['Pclass'] == 2) & (~ctrd.Ticket.str.contains(only_num_regex))][['Survived', 'Ticket']]
	
	class3_ticket_number = ctrd[
		(ctrd['Pclass'] == 3) & (ctrd.Ticket.str.contains(only_num_regex))][['Survived', 'Ticket']]
	class3_ticket_alpha = ctrd[
		(ctrd['Pclass'] == 3) & (~ctrd.Ticket.str.contains(only_num_regex))][['Survived', 'Ticket']]
	'''

	# for Pclass in range(1, 4):
	# 	alphaTicket_list = trd[(trd['Pclass'] == Pclass) & (trd.Ticket.str.contains("[a-zA-Z]"))] \
	# 		.Ticket.str.split(' ').str[0].unique()
	#
	# 	for alphaTicket in alphaTicket_list:
	# 		ctrd.loc[(ctrd.Ticket.str.contains(alphaTicket)) & (ctrd.Pclass == Pclass), 'Ticket'] = 1

	# 가설 1. 남성이 여성보다 많이 죽었을것이다.
	# 가설 2. 어린아이, 노인의 생존률이 더 높았을 것이다. (검증 결과 어린아이의 생존률만 높음)
	# 가설 1과 2를 조합하여 나이대별로도 클래스를 구분하고 싶지만
	# 실제로 그러기는 어려우므로 남자와 여자로 구분한다.

	del ctrd['Ticket']

	'''
	Fare - 
	0원인 애들은 각 Pclass의 평균으로 맞춰놓음 
	그 후 평균보다 낮으면 0 , 같거나 높으면 1로 변환 
	'''
	ctrd.loc[((ctrd.Fare.isna()) | (ctrd.Fare == 0)) & (ctrd.Pclass == 1), 'Fare'] = ctrd[ctrd.Pclass == 1].Fare.mean()
	ctrd.loc[((ctrd.Fare.isna()) | (ctrd.Fare == 0)) & (ctrd.Pclass == 2), 'Fare'] = ctrd[ctrd.Pclass == 2].Fare.mean()
	ctrd.loc[((ctrd.Fare.isna()) | (ctrd.Fare == 0)) & (ctrd.Pclass == 3), 'Fare'] = ctrd[ctrd.Pclass == 3].Fare.mean()

	fare_mean = ctrd.Fare.mean()
	ctrd.loc[(ctrd.Fare < fare_mean), 'Fare'] = 0
	ctrd.loc[(ctrd.Fare >= fare_mean), 'Fare'] = 1


	'''
	Cabin 과  Embarked 삭제
	'''
	del ctrd['Cabin']
	del ctrd['Embarked']

	# alpha_ticket_list = trd[trd.Ticket.str.contains("[a-zA-Z]")] \
	# 	.Ticket.str.replace(r' \d{1,}', '', regex=True).str.replace('.', '').unique()
	#
	# ticket_plot_list = []
	# ticket_survived_rate_list = []
	#
	# for index, ticket in enumerate(alpha_ticket_list):
	# 	ticket_plot_list.append(index)
	# 	ticket_survived_rate_list.append(
	# 		trd[trd.Ticket.str.replace('.', '').str.contains(ticket)].Survived.mean())
	#
	# plt.plot(ticket_plot_list, ticket_survived_rate_list)
	# plt.show()

	# 가설 2:
	# 어린아이는 생존률이 높지만 나이든 사람의 생존률은 높지 않다.
	return ctrd

# splitter = '\\' if platform.system() == 'Windows' else '/'
# pwd = os.path.dirname(os.getcwd()) + splitter
#
# if 'titanic' not in pwd:
# 	pwd = pwd + 'titanic_project' + splitter
#
# trd = pd.read_csv(pwd + f'data{splitter}train.csv')
# gsc = pd.read_csv(pwd + f'data{splitter}gender_submission.csv')
# trd.append(gsc)
# ctrd = preprocessing(trd)
