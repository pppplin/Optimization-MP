import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn import model_selection, metrics, datasets
from neupy import algorithms, layers
from scipy.misc import imresize

def load_data():
    X, y = datasets.fetch_openml('mnist_784', version=1, return_X_y=True)
    X /= 255.
    X -= X.mean(axis=0)
    # X_ = np.empty((0, 196))
    # for i in range(X.shape[0]):
    # 	X_ = np.vstack((X_, imresize(X[i].reshape((28,28)), (14, 14)).reshape(196)))
    # print X_.shape

    target_scaler = OneHotEncoder(sparse=False, categories='auto')
    y = target_scaler.fit_transform(y.reshape(-1, 1))

    return model_selection.train_test_split(
        X.astype(np.float32),
        y.astype(np.float32),
        test_size=(1 / 7.))


# # cojugate gradient
# optimizer = algorithms.ConjugateGradient(
# 	network=[
# 	layers.Input(784),
# 	layers.Softmax(10)
# 	],
# 	update_function='polak_ribiere',
# 	loss='categorical_crossentropy',
# 	verbose=False
# 	)

# quasi newton
optimizer = algorithms.QuasiNewton(
	network=[
	layers.Input(784),
	layers.Softmax(10)
	],
	loss='categorical_crossentropy',
	verbose=False,
	show_epoch=10
	)

print("Preparing data...")
x_train, x_test, y_train, y_test = load_data()


# print("Training...")
# optimizer.train(x_train, y_train, x_test, y_test, epochs=20)

# y_predicted = optimizer.predict(x_test).argmax(axis=1)
# y_test = np.asarray(y_test.argmax(axis=1)).reshape(len(y_test))

# print(metrics.classification_report(y_test, y_predicted))
# score = metrics.accuracy_score(y_test, y_predicted)
# print("Validation accuracy: {:.2%}".format(score))

