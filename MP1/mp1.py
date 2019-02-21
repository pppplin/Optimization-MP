#import click
import time
from sklearn.linear_model import LogisticRegression
#from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from neupy import layers
#from scipy.misc import imresize
from data_loader import load_data
import matplotlib.pyplot as plt
#@click.command()
#@click.option('--method', type=str)
#@click.option('--epochs', type=int)
#@click.option('--larger_param', type=bool, default=False)
#@click.option('--full_batch', type=bool, default=True)
#@click.option('--no_restart', type=bool, default=False)

def main(method, epochs, larger_param, full_batch, no_restart):
    if larger_param:
        in_num = 784
        softmax_num = 10
    else:
        in_num = 4
        softmax_num = 3
    x_train, x_test, y_train, y_test = load_data(larger=larger_param, method=method)
    if method=='cg':
        clf = LogisticRegression(solver="newton-cg", multi_class='multinomial', warm_start=True)
    elif method=='pr':
        y_test_label = y_test.argmax(axis=1)
        y_train_label = y_train.argmax(axis=1)
        if no_restart:
            from conjgrad import ConjugateGradient
        else:
            from neupy.algorithms import ConjugateGradient
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
        batch_size = 100 #size of mini-batch

    print("Training")
    print("Full Batch Size: {}".format(full_batch_size))
    print("Batch Size: {}".format(batch_size))

    time_slot = []
    train_acc = []
    test_acc = []
    time_acc = 0
    for epoch in range(epochs):
        print ("Epoch: {}".format(epoch))
        for i in range(int(full_batch_size/batch_size)):
            print ("Iteration: {}".format(i))
            data = x_train[i*batch_size:(i+1)*batch_size]
            label = y_train[i*batch_size:(i+1)*batch_size]
            if method=='cg':
                start = time.time()
                clf.fit(data, label)
                end = time.time()
                time_acc = end - start + time_acc
                time_slot.append(time_acc)
                train_acc.append(clf.score(x_train, y_train))
                test_acc.append(clf.score(x_test, y_test))
            else:
                start = time.time()
                optimizer.train(data, label, epochs=1)
                end = time.time()
                time_acc = end - start + time_acc
                time_slot.append(time_acc)
                y_predict_test = optimizer.predict(x_test).argmax(axis=1)
                y_predict_train = optimizer.predict(x_train).argmax(axis=1)
                #        print(metrics.classification_report(y_test, y_predict))
                score_test = metrics.accuracy_score(y_test_label, y_predict_test)
                score_train = metrics.accuracy_score(y_train_label, y_predict_train)
                test_acc.append(score_test)
                train_acc.append(score_train)
    return time_slot, train_acc, test_acc

def plot(time_slot, train_acc, test_acc, name):
    plt.figure(figsize=(8,6), dpi=300)
    for i in range(len(time_slot)):
        print("epoch: {}".format(i))
        print("training accuracy : {}".format(train_acc[i]))
        print("testing accuracy : {}".format(test_acc[i]))
        print("time (s) : {}".format(time_slot[i]))
    plt.title("Convergence Time And Accuracy")
    plt.xlabel("Convergence Time (s)")
    plt.ylabel("Accuracy")
    
    
    plt.plot(time_slot, train_acc, color="blue", linewidth=1.0, linestyle="-", label='training_accuracy',)
    plt.plot(time_slot, test_acc, color="green", linewidth=1.0, linestyle="-", label='test_accuracy')
    plt.legend(loc='upper left')
    plt.ylim(0, 1.0)
    plt.savefig("./figures/{}.png".format(name),dpi=300)
    plt.show()

if __name__ == '__main__':
    m='cg'
    e=10
    l=True
    f=False
    nr=True
    time_slot, train_acc, test_acc = main(method=m, epochs=e, larger_param=l, full_batch=f, no_restart=nr)
    plot(time_slot, train_acc, test_acc, "method: {}, epoch: {}, larger: {}, full: {}, nr: {}".format(m, e, l, f, nr))