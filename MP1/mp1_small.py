from sklearn import datasets, preprocessing, metrics
from sklearn.model_selection import train_test_split
from neupy import algorithms, layers
import time

dataset = datasets.load_iris()
data, target = dataset.data, dataset.target
data_scaler = preprocessing.MinMaxScaler()
target_scaler = preprocessing.OneHotEncoder(sparse=False)

x_train, x_test, y_train, y_test = train_test_split(
	data_scaler.fit_transform(data),
	target_scaler.fit_transform(target.reshape(-1, 1)),
	test_size=0.15
	)


# x_train, x_test, y_train, y_test = train_test_split(
# 	data_scaler.fit_transform(data),
# 	target_scaler.fit_transform(target.reshape(-1, 1)),
# 	test_size=0.15
# 	)


# pr gradient
optimizer = algorithms.ConjugateGradient(
	network=[
	layers.Input(4),
	layers.Softmax(3)
	],
	update_function='polak_ribiere',
	loss='categorical_crossentropy',
	verbose=True,
	show_epoch=1
	)

# cg newton
# optimizer = algorithms.QuasiNewton(
# 	network=[
# 	layers.Input(4),
# 	layers.Softmax(3)
# 	],
# 	update_function='dfp',
# 	loss='categorical_crossentropy',
# 	verbose=True,
# 	show_epoch=1
# 	# regularizer=algorithms.l2(10.)
# 	)

start = time.time()
print("Training...")

# i = 13
# batch_size = 100
# print x_train.shape
# print y_train.shape
# print [i*batch_size, (i+1)*batch_size]
# data = x_train[i*batch_size:(i+1)*batch_size]
# label = y_train[i*batch_size:(i+1)*batch_size]
# print data
# print label


epochs = 10
batch_size = 1 #y_train.shape[0]
for epoch in range(epochs):
	# print y_train.shape[0]/batch_size - 1
	for i in range(y_train.shape[0]/batch_size):
		# print i
		data = x_train[i*batch_size:(i+1)*batch_size]
		label = y_train[i*batch_size:(i+1)*batch_size]
		# print data, label
		optimizer.train(data, label, epochs=1)

# print type(optimizer)
# print optimizer.loss
end = time.time()

y_predict = optimizer.predict(x_test).argmax(axis=1)
y_test = y_test.argmax(axis=1)
print(metrics.classification_report(y_test, y_predict))
score = metrics.accuracy_score(y_test, y_predict)
print("Validation accuracy: {:.2%}".format(score))
print("Use time : {}".format((end-start)/60.))
