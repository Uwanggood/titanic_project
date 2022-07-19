import pandas as pd

# 판다스에는 두가지 핵심 오브젝트가 있음 .
# 1. DataFrame
# 2. Series


# 1. 데이터프레임
# 데이터프레임은 테이블이다.
# 특정 값이 있는 배열형태의 데이터를 저장하는 객체이다.
pd.DataFrame({'Yes': [50, 21], 'No': [131, 2]})

pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'],
			  'Sue': ['Pretty good.', 'Bland.']})

# 2. 데이터시리즈
# 데이터시리즈는 리스트형태이다.
pd.Series([30, 35, 40], index=['2015 Sales', '2016 Sales', '2017 Sales'],
		  name='Product A')


