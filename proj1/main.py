import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import math


# Location = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'


classifier_name = sys.argv[1]
data_file = sys.argv[2]
key = sys.argv[3]
classifier_type = ['perceptron', 'adaline', 'sgd', 'one_vs_rest']
learning_percentage = 0.7
test_percentage = 0.3
start_col = int(sys.argv[4])
len_col = int(sys.argv[5])
y_col = int(sys.argv[6])
header_value = sys.argv[7]



class perceptron(object):

    def __init__(self, eta=0.1, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def predict(self, X):
        z = np.dot(X, self.w_[1:]) + self.w_[0]
        return np.where(z >= 0.0, 1, -1)

    def showplot(self):
        plt.plot(range(1, len(self.errors_) + 1), self.errors_, marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Number of updates')
        plt.show()
        return


class adaline(object):
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = np.dot(X, self.w_[1:]) + self.w_[0]
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def predict(self, X):
        z = np.dot(X, self.w_[1:]) + self.w_[0]
        return np.where(z >= 0.0, 1, -1)

    def showplot(self):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
        ax[0].plot(range(1, len(self.cost_) + 1), np.log10(self.cost_), marker='o')
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('log(Sum-squared-error)')
        ax[0].set_title('Adaline - Learning rate 0.05')
        ada2 = adaline(n_iter=10, eta=0.0001).fit(Xtr, ytr)
        ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Sum-squared-error')
        ax[1].set_title('Adaline - Learning rate 0.0001')
        plt.show()



class sgd(object):
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.rgen = np.random.RandomState(self.random_state)

    def shuffle(self, X, y):
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def fit(self, X, y):
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            X, y = self.shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self.update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
        return self

    def update_weights(self, xi, target):
        output = np.dot(xi, self.w_[1:]) + self.w_[0]
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost

    def predict(self, X):
        z = np.dot(X, self.w_[1:]) + self.w_[0]
        return np.where(z >= 0.0, 1, -1)


       


if classifier_name in classifier_type:
    Location = r'%s' % data_file
    if header_value == 'no' :
       df = pd.read_csv(Location, header=None)
    else:
       df = pd.read_csv(Location, header=0)
        
    num_learning_row = math.floor(df.shape[0] * learning_percentage)
    num_testing_row = df.shape[0] - num_learning_row
    Xtr = df.iloc[0: num_learning_row, [ start_col, len_col]].values
    ytr = df.iloc[0: num_learning_row, y_col].values
    ytr = np.where(ytr == key, 1, -1)
    Xts = df.iloc[num_learning_row + 1: df.shape[0], [start_col, len_col]].values
    yts = df.iloc[num_learning_row + 1: df.shape[0], y_col].values
    yts = np.where(yts == key, 1, -1)
    total_true_value = 0
    total_predict_value = 0
    for x in range(0 , num_testing_row - 1):
        if yts[x] == 1:
          total_true_value =  total_true_value + 1


    if classifier_name == 'perceptron' :
            ppn = perceptron(eta=0.05, n_iter=100)
            ppn.fit(Xtr, ytr)
            ppn.showplot()
            predict_one = ppn.predict(Xts)
    elif classifier_name == 'adaline' :
        ada = adaline(eta=0.05, n_iter=100)
        ada.fit(Xtr, ytr)
        ada.showplot()
        predict_one = ada.predict(Xts)

    elif classifier_name == 'sgd' :
        StockGD = sgd(eta=0.05, n_iter=100)
        StockGD.fit(Xtr, ytr)
        predict_one = StockGD.predict(Xts)



    for x in range(0 , num_testing_row - 1):
        if predict_one[x] == 1:
          total_predict_value =  total_predict_value + 1

    total_error = abs(total_predict_value-total_true_value)
    print('total error is ', total_error )
    if total_predict_value == total_true_value :
        print('Accuracy is 100%')
    else:
        accuracy = (num_testing_row - total_error) / num_testing_row
        print('Accuracy is ', accuracy , '%')
    
            


else:
    print("Wrong arguments!, Please try again.")
