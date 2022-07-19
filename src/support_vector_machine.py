from sklearn import svm, datasets

# sklearn에서 학습용 데이터셋을 불러옵니다
digits = datasets.load_digits()

# Support vector machine Classifier 를 생성합니다.
clf = svm.SVC(gamma=0.001, C=100)

# 학습용 데이터셋을 설정합니다.
x, y = digits.data[:-1], digits.target[:-1]

# 학습 모델
clf.fit(x, y)

# 예측
y_pred = clf.predict([digits.data[-1]])
y_true = digits.target[-1]

print(y_pred)
print(y_true)
