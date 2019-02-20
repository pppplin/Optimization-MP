from sklearn import datasets, preprocessing
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split

def load_data(larger=False):
    if larger:
        X, y = datasets.fetch_openml('mnist_784', version=1, return_X_y=True)
        random_state = check_random_state(0)
        permutation = random_state.permutation(X.shape[0])
        X = X[permutation]
        y = y[permutation]
        X = X.reshape((X.shape[0], -1))
        #truncate dataset
        X = X[:1000]
        y = y[:1000]
        return train_test_split(X, y, test_size=(1 / 5.))

    dataset = datasets.load_iris()
    data, target = dataset.data, dataset.target
    data_scaler = preprocessing.MinMaxScaler()
    return train_test_split(
            data_scaler.fit_transform(data),
            target,
            test_size=0.15
            )


