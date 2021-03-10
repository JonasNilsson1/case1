import pandas as pd 
import sklearn.linear_model
import sklearn.model_selection
import numpy as np

data = pd.read_csv('case1Data.txt', sep=",")

# x_1 - x_95 are numerical variables
# C_1 - C_5 are categorixal variables

#data = pd.get_dummies(data)

data.columns = data.columns.str.replace('_ ','_').str.strip()

# Convert categorical data to strings, removing spaces

category_cols = ["C_1","C_2","C_3","C_4","C_5"]

for col in category_cols:

	data[col] = data[col].map(str).str.replace(" ","")

data = data.replace("NaN", np.NaN)

NaN_counts = data.isna().sum().to_dict()

print("The following columns have NaN:")
for col in data:
	if NaN_counts[col] > 0:
		print(col)

# Fill missing with most frequent values
data[category_cols] = data[category_cols].apply(lambda x: x.fillna(x.value_counts().index[0]))


# Convert labels to one hot, so we only have numerical

one_hot = pd.get_dummies(data)

Y = np.array(one_hot['y'])
X = np.array(one_hot.drop(['y'], axis=1))



K = 5
kf = sklearn.model_selection.KFold(n_splits=K)


ridge_rmse_vec = np.zeros(K)


def standardize(data):
    
    mu = np.mean(data,axis=0)
    sigma = np.std(data, axis=0)
    data = (data - mu) / sigma
    
    return data, mu, sigma


for i, (train_index, test_index) in enumerate(kf.split(X)):
	X_train = X[train_index]
	X_test = X[test_index]
	
	Y_train = Y[train_index]
	Y_test = Y[test_index]

	X_train[:, :95], mu, sigma = standardize(X_train[:, :95])

	# use the mean and std of the training data to standardize test set
	X_test[:, :95] = (X_test[:, :95]-mu)/sigma  


	# Ridge regression
	ridge = sklearn.linear_model.Ridge()
	ridge = ridge.fit(X_train,Y_train)
	
	Y_hat = ridge.predict(X_test)

	rmse = np.sqrt(np.mean((Y_test - Y_hat)**2))

	ridge_rmse_vec[i] = rmse	


	# Lassoo
	clf = sklearn.linear_model.Lasso(alpha=0.1)

	

print(ridge_rmse.mean())





