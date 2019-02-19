import click
#from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics, datasets, preprocessing
from neupy import algorithms, layers
from neupy.algorithms import ConjugateGradient
from sklearn.model_selection import train_test_split
#from scipy.misc import imresize
from data_loader import load_data

@click.command()
@click.option('--method', type=str) # cg or pr
@click.option('--epochs', type=int)
@click.option('--larger_param', type=bool, default=False)
@click.option('--reg', type=bool, default=False)
@click.option('--no_restart', type=bool, default=False)

def main(method, epochs, larger_param, reg, no_restart):
    if no_restart:
        from conjgrad import ConjugateGradient
    if larger_param:
        x_train, x_test, y_train, y_test = load_data()
        in_num = 784
        softmax_num = 10
    else:
        dataset = datasets.load_iris()
        data, target = dataset.data, dataset.target
        data_scaler = preprocessing.MinMaxScaler()
        target_scaler = preprocessing.OneHotEncoder(sparse=False)
        x_train, x_test, y_train, y_test = train_test_split(
            data_scaler.fit_transform(data),
            target_scaler.fit_transform(target.reshape(-1, 1)),
            test_size=0.15
            )
        in_num = 4
        softmax_num = 3

    if reg:
        regularizer = algorithms.l2(10.)
    else:
        regularizer = None

    if method=='cg':
        optimizer = algorithms.QuasiNewton(
            network=[
            layers.Input(in_num),
            layers.Softmax(softmax_num)
            ],
            update_function='dfp',
            loss='categorical_crossentropy',
            verbose=True,
            show_epoch=1,
            regularizer = regularizer
            )

    elif method=='pr':
        optimizer = ConjugateGradient(
          network=[
          layers.Input(in_num),
          layers.Softmax(softmax_num)
          ],
          update_function='polak_ribiere',
          loss='categorical_crossentropy',
          verbose=True,
          show_epoch=1,
          regularizer = regularizer
          )
    else:
        print("No such method.")
        assert False

    print("Training...")
    optimizer.train(x_train, y_train, epochs=epochs)
    y_predict = optimizer.predict(x_test).argmax(axis=1)
    y_test = y_test.argmax(axis=1)
    print(metrics.classification_report(y_test, y_predict))
    score = metrics.accuracy_score(y_test, y_predict)
    print("Validation accuracy: {:.2%}".format(score))

if __name__ == '__main__':
    main()

