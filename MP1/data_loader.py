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


