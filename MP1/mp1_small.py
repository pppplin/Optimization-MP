from sklearn import datasets, preprocessing, metrics
from sklearn.model_selection import train_test_split
from neupy import algorithms, layers

dataset = datasets.load_iris()
data, target = dataset.data, dataset.target
data_scaler = preprocessing.MinMaxScaler()
target_scaler = preprocessing.OneHotEncoder(sparse=False)

x_train, x_test, y_train, y_test = train_test_split(
	data_scaler.fit_transform(data),
	target_scaler.fit_transform(target.reshape(-1, 1)),
	test_size=0.15
	)


x_train, x_test, y_train, y_test = train_test_split(
	data_scaler.fit_transform(data),
	target_scaler.fit_transform(target.reshape(-1, 1)),
	test_size=0.15
	)


# cojugate gradient
# optimizer = algorithms.ConjugateGradient(
# 	network=[
# 	layers.Input(4),
# 	layers.Softmax(3)
# 	],
# 	update_function='polak_ribiere',
# 	loss='categorical_crossentropy',
# 	verbose=False
# 	)

# quasi newton
optimizer = algorithms.QuasiNewton(
	network=[
	layers.Input(4),
	layers.Softmax(3)
	],
	loss='categorical_crossentropy',
	verbose=False,
	show_epoch=10
	)

print("Training...")
optimizer.train(x_train, y_train, epochs=100)
y_predict = optimizer.predict(x_test).argmax(axis=1)
y_test = y_test.argmax(axis=1)
print(metrics.classification_report(y_test, y_predict))
score = metrics.accuracy_score(y_test, y_predict)
print("Validation accuracy: {:.2%}".format(score))
