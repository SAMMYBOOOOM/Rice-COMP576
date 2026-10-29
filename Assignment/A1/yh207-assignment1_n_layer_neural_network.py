# yh207_n_layer_neural_network.py
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def generate_data():
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)
    return X, y


def plot_decision_boundary(pred_func, X, y):
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


class DeepNeuralNetwork(object):
    """
    Fully-connected neural network with arbitrary depth.
    layer_sizes = [input_dim, hidden1, hidden2, ..., output_dim]
    """

    def __init__(self, layer_sizes, actFun_type='tanh', reg_lambda=0.01, seed=0):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda

        np.random.seed(seed)
        self.W, self.b = [], []
        for i in range(self.num_layers):
            # He-style init works well for tanh/sigmoid too
            W = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i + 1]))
            self.W.append(W)
            self.b.append(b)

    def actFun(self, z, type):
        type = type.lower()
        if type == 'tanh':    return np.tanh(z)
        if type == 'sigmoid': return 1.0 / (1.0 + np.exp(-z))
        if type == 'relu':    return np.maximum(0, z)
        raise ValueError("Unsupported activation: " + type)

    def diff_actFun(self, z, type):
        type = type.lower()
        if type == 'tanh':    return 1.0 - np.tanh(z) ** 2
        if type == 'sigmoid':
            s = 1.0 / (1.0 + np.exp(-z))
            return s * (1.0 - s)
        if type == 'relu':    return (z > 0).astype(float)
        raise ValueError("Unsupported activation: " + type)

    def feedforward(self, X):
        self.z, self.a = [], [X]
        for i in range(self.num_layers):
            z = np.dot(self.a[-1], self.W[i]) + self.b[i]
            self.z.append(z)
            if i == self.num_layers - 1:                 # softmax output
                e = np.exp(z - np.max(z, axis=1, keepdims=True))
                self.a.append(e / np.sum(e, axis=1, keepdims=True))
            else:
                self.a.append(self.actFun(z, self.actFun_type))
        self.probs = self.a[-1]
        return self.probs

    def calculate_loss(self, X, y):
        n = len(X)
        self.feedforward(X)
        data_loss = -np.sum(np.log(self.probs[range(n), y])) / n
        reg_loss = 0.5 * self.reg_lambda * sum(np.sum(np.square(W)) for W in self.W)
        return data_loss + reg_loss

    def predict(self, X):
        self.feedforward(X)
        return np.argmax(self.probs, axis=1)

    def backprop(self, X, y):
        n = len(X)
        self.feedforward(X)
        dW, db = [None] * self.num_layers, [None] * self.num_layers

        delta = self.probs.copy()
        delta[range(n), y] -= 1.0
        delta /= n

        for i in reversed(range(self.num_layers)):
            dW[i] = np.dot(self.a[i].T, delta)
            db[i] = np.sum(delta, axis=0, keepdims=True)
            if i > 0:
                delta = np.dot(delta, self.W[i].T) * \
                        self.diff_actFun(self.z[i - 1], self.actFun_type)
        return dW, db

    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
        for i in range(num_passes):
            dW, db = self.backprop(X, y)
            for j in range(self.num_layers):
                dW[j] += self.reg_lambda * self.W[j]
                self.W[j] += -epsilon * dW[j]
                self.b[j] += -epsilon * db[j]
            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, y)))

    def visualize_decision_boundary(self, X, y):
        plot_decision_boundary(lambda x: self.predict(x), X, y)


def main():
    X, y = generate_data()
    configs = [[2, 3, 2], [2, 10, 2], [2, 10, 10, 2], [2, 20, 20, 2]]
    for layers in configs:
        print("\n" + "=" * 60)
        print("Make-Moons layers:", layers)
        print("=" * 60)
        m = DeepNeuralNetwork(layers, actFun_type='tanh', reg_lambda=0.01)
        m.fit_model(X, y, num_passes=20000, print_loss=True)
        m.visualize_decision_boundary(X, y)

    iris = datasets.load_iris()
    X_tr, X_te, y_tr, y_te = train_test_split(iris.data, iris.target,
                                              test_size=0.3, random_state=0)
    sc = StandardScaler().fit(X_tr)
    X_tr, X_te = sc.transform(X_tr), sc.transform(X_te)

    m = DeepNeuralNetwork([4, 16, 16, 3], actFun_type='tanh', reg_lambda=0.01)
    m.fit_model(X_tr, y_tr, num_passes=5000, print_loss=True)
    print("Iris test accuracy: %.2f%%" % (100.0 * np.mean(m.predict(X_te) == y_te)))


if __name__ == "__main__":
    main()