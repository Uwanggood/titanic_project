from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet


def pred_linear_regression(x_train, x_test, y_train, y_test):
	std_scale = StandardScaler()
	std_scale.fit(x_train)
	X_tn_std = std_scale.transform(x_train)
	X_te_std = std_scale.transform(x_test)
	clf_lr = LinearRegression()
	clf_lr.fit(X_tn_std, y_train)

	# ridge 적용
	clf_ridge = Ridge(alpha=0.1)
	clf_ridge.fit(X_tn_std, y_train)

	# lasso 적용
	clf_lasso = Lasso(alpha=0.1)
	clf_lasso.fit(X_tn_std, y_train)

	# elastic net 적용
	clf_elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
	clf_elastic_net.fit(X_tn_std, y_train)

	pred_lr = clf_lr.predict(X_te_std)
	pred_ridge = clf_ridge.predict(X_te_std)
	pred_lasso = clf_lasso.predict(X_te_std)
	pred_elastic_net = clf_elastic_net.predict(X_te_std)
	return [1 if i > 0.5 else 0 for i in pred_lr], \
		   [1 if i > 0.5 else 0 for i in pred_ridge], \
		   [1 if i > 0.5 else 0 for i in pred_lasso], \
		   [1 if i > 0.5 else 0 for i in pred_elastic_net]
