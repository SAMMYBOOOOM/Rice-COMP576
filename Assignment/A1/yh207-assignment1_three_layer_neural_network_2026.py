__author__ = 'tan_nguyen'

import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


def generate_data():
    '''
    generate data
    :return: X: input data, y: given labels
    '''
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)
    return X, y


def plot_decision_boundary(pred_func, X, y):
    '''
    plot the decision boundary
    :param pred_func: function used to predict the label
    :param X: input data
    :param y: given labels
    :return:
    '''
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()


class NeuralNetwork(object):
    """
    This class builds and trains a neural network
    """

    def __init__(self, nn_input_dim, nn_hidden_dim, nn_output_dim,
                 actFun_type='tanh', reg_lambda=0.01, seed=0):
        self.nn_input_dim = nn_input_dim
        self.nn_hidden_dim = nn_hidden_dim
        self.nn_output_dim = nn_output_dim
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda

        np.random.seed(seed)
        self.W1 = np.random.randn(self.nn_input_dim, self.nn_hidden_dim) / np.sqrt(self.nn_input_dim)
        self.b1 = np.zeros((1, self.nn_hidden_dim))
        self.W2 = np.random.randn(self.nn_hidden_dim, self.nn_output_dim) / np.sqrt(self.nn_hidden_dim)
        self.b2 = np.zeros((1, self.nn_output_dim))

    def actFun(self, z, type):
        '''
        actFun computes the activation functions
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: activations
        '''
        type = type.lower()
        if type == 'tanh':
            return np.tanh(z)
        elif type == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-z))
        elif type == 'relu':
            return np.maximum(0, z)
        else:
            raise ValueError("actFun_type must be 'tanh', 'sigmoid', or 'relu'")

    def diff_actFun(self, z, type):
        '''
        diff_actFun computes the derivatives of the activation functions wrt the net input
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: derivatives
        '''
        type = type.lower()
        if type == 'tanh':
            return 1.0 - np.tanh(z) ** 2
        elif type == 'sigmoid':
            s = 1.0 / (1.0 + np.exp(-z))
            return s * (1.0 - s)
        elif type == 'relu':
            return (z > 0).astype(float)
        else:
            raise ValueError("actFun_type must be 'tanh', 'sigmoid', or 'relu'")

    def feedforward(self, X, actFun):
        '''
        feedforward builds a 3-layer neural network and computes the two probabilities
        '''
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = actFun(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2

        # stable softmax
        exp_scores = np.exp(self.z2 - np.max(self.z2, axis=1, keepdims=True))
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return None

    def calculate_loss(self, X, y):
        '''
        calculate_loss computes the loss for prediction
        '''
        num_examples = len(X)
        self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))

        # negative log likelihood
        data_loss = -np.sum(np.log(self.probs[range(num_examples), y])) / num_examples

        # L2 regularization
        reg_loss = 0.5 * self.reg_lambda * (
            np.sum(np.square(self.W1)) + np.sum(np.square(self.W2))
        )

        return data_loss + reg_loss

    def predict(self, X):
        '''
        predict infers the label of a given data point X
        '''
        self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
        return np.argmax(self.probs, axis=1)

    def backprop(self, X, y):
        '''
        backprop runs backpropagation to compute the gradients
        '''
        num_examples = len(X)

        # dL/dz2 for softmax + NLL
        dlogprobs = self.probs.copy()
        dlogprobs[range(num_examples), y] -= 1.0
        dlogprobs /= num_examples

        # gradients for W2, b2
        dW2 = np.dot(self.a1.T, dlogprobs)
        db2 = np.sum(dlogprobs, axis=0, keepdims=True)

        # gradient through hidden layer
        da1 = np.dot(dlogprobs, self.W2.T)
        dz1 = da1 * self.diff_actFun(self.z1, self.actFun_type)

        # gradients for W1, b1
        dW1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        return dW1, dW2, db1, db2

    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
        '''
        fit_model uses backpropagation to train the network
        '''
        for i in range(0, num_passes):
            self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
            dW1, dW2, db1, db2 = self.backprop(X, y)

            # Add regularization derivatives
            dW2 += self.reg_lambda * self.W2
            dW1 += self.reg_lambda * self.W1

            # Gradient descent update
            self.W1 += -epsilon * dW1
            self.b1 += -epsilon * db1
            self.W2 += -epsilon * dW2
            self.b2 += -epsilon * db2

            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, y)))

    def visualize_decision_boundary(self, X, y):
        plot_decision_boundary(lambda x: self.predict(x), X, y)


def main():
    # 1. Generate and visualize Make-Moons
    X, y = generate_data()
    plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    plt.title("Make-Moons Dataset")
    plt.show()

    # 2. Train with different activation functions
    for act in ['tanh', 'sigmoid', 'relu']:
        print("\n" + "=" * 60)
        print("Activation function:", act)
        print("=" * 60)
        model = NeuralNetwork(nn_input_dim=2, nn_hidden_dim=3,
                              nn_output_dim=2, actFun_type=act)
        model.fit_model(X, y, num_passes=20000, print_loss=True)
        model.visualize_decision_boundary(X, y)

    # 3. Increase number of hidden units with tanh
    print("\n" + "=" * 60)
    print("Tanh with more hidden units: 20")
    print("=" * 60)
    model = NeuralNetwork(nn_input_dim=2, nn_hidden_dim=20,
                          nn_output_dim=2, actFun_type='tanh')
    model.fit_model(X, y, num_passes=20000, print_loss=True)
    model.visualize_decision_boundary(X, y)


if __name__ == "__main__":
    main()