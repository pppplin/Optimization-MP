from sklearn.linear_model import LogisticRegression
from sklearn import datasets, preprocessing, metrics
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state

# def load_data():
# 	X, y = datasets.fetch_openml('mnist_784', version=1, return_X_y=True)
# 	random_state = check_random_state(0)
# 	permutation = random_state.permutation(X.shape[0])
# 	X = X[permutation]
# 	y = y[permutation]
# 	X = X.reshape((X.shape[0], -1))
# 	return train_test_split(X, y, test_size=(1 / 7.))

# print("Preparing data...")
# x_train, x_test, y_train, y_test = load_data()


dataset = datasets.load_iris()
data, target = dataset.data, dataset.target
data_scaler = preprocessing.MinMaxScaler()
target_scaler = preprocessing.OneHotEncoder(sparse=False)

x_train, x_test, y_train, y_test = train_test_split(
	data_scaler.fit_transform(data),
	target,
	test_size=0.15
	)

clf = LogisticRegression(solver="newton-cg", warm_start=True, multi_class='multinomial')

epochs = 10
batch_size = y_train.shape[0]
for epoch in range(epochs):
	for i in range(y_train.shape[0]/batch_size):
		data = x_train[i*batch_size:(i+1)*batch_size]
		label = y_train[i*batch_size:(i+1)*batch_size]
		clf.fit(data,label)
	print("training accuracy : {}").format(clf.score(x_train, y_train))
	print("testing accuracy : {}").format(clf.score(x_test, y_test))


# print clf.score(x_test, y_test)

