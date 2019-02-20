import click, time
from sklearn.linear_model import LogisticRegression
#from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from neupy import layers
#from scipy.misc import imresize
from data_loader import load_data

@click.command()
@click.option('--method', type=str)
@click.option('--epochs', type=int)
@click.option('--larger_param', type=bool, default=False)
@click.option('--full_batch', type=bool, default=True)
@click.option('--no_restart', type=bool, default=False)

def main(method, epochs, larger_param, full_batch, no_restart):
    if no_restart:
        from conjgrad import ConjugateGradient
    else:
        from neupy.algorithms import ConjugateGradient

    if larger_param:
        in_num = 784
        softmax_num = 10
    else:
        in_num = 4
        softmax_num = 3
    x_train, x_test, y_train, y_test = load_data(larger=larger_param)

    if method=='cg':
        clf = LogisticRegression(solver="newton-cg", multi_class='multinomial', warm_start=True)

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
                )

    full_batch_size = int(y_train.shape[0])
    if full_batch:
        batch_size = full_batch_size
    else:
        batch_size = 10 #size of mini-batch

    print("Training")
    start = time.time()
    for epoch in range(epochs):
        for i in range(int(full_batch_size/batch_size)):
            data = x_train[i*batch_size:(i+1)*batch_size]
            label = y_train[i*batch_size:(i+1)*batch_size]
            if method=='cg':
                clf.fit(data, label)
            else:
                optimizer.train(data, label, epochs=1)
        if method=='cg':
            print("training accuracy : {}".format(clf.score(x_train, y_train)))
            print("testing accuracy : {}".format(clf.score(x_test, y_test)))

    end = time.time()
    if method=='pr':
        y_predict = optimizer.predict(x_test).argmax(axis=1)
        y_test = y_test.argmax(axis=1)
        print(metrics.classification_report(y_test, y_predict))
        score = metrics.accuracy_score(y_test, y_predict)
        print("Validation accuracy: {:.2%}".format(score))
    print("Use time : {}".format((end-start)/60.))

if __name__ == '__main__':
    main()
