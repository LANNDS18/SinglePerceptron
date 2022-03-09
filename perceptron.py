"""
Author: Wuwei Zhang
E-mail: sgwhzha23@liverpool.ac.uk
Student ID: 201522671
"""
import numpy as np


def load_data(file_name):
    """
    Load the data from the file
    """
    x = []
    y = []
    with open(file_name, 'r') as f:
        for line in f.readlines():
            line = line.strip().split(',')
            label = 0 if line[4] == 'class-1' else 1 if line[4] == 'class-2' else 2
            x.append([float(line[0]), float(line[1]), float(line[2]), float(line[3])])
            y.append(label)
    return np.array(x), np.array(y)


def single_perceptron(x, weight, bias):
    """
    Thr forward propagation of the perceptron
    """
    return np.dot(weight, x) + bias


def shuffle_data(x, y):
    """
    Shuffle the data and label
    """
    index = np.arange(len(x))
    np.random.shuffle(index)
    return x[index], y[index]


def train_perceptron(_train_x, y, lr=0.01, epoch=20, l2_reg=0.0, shuffle=True):
    """
    The perceptron training algorithm, using y_pred * y_true as loss function
    :param shuffle: Shuffle the data before each epoch or not
    :param lr: Learning rate
    :param epoch: Number of epochs
    :param _train_x: input vector with shape (n, m)
    :param y: input label with shape (n, 1) (Clip into {-1, 1})
    :param l2_reg: L2 regularization
    :return: weight vector with shape (m, 1) and bias
    """
    weight = np.zeros(_train_x.shape[1])
    bias = 0
    update_count = 0
    for i in range(epoch):
        if shuffle:
            _train_x, y = shuffle_data(_train_x, y)
        for j in range(len(_train_x)):
            pred = single_perceptron(_train_x[j], weight, bias)

            if y[j] * pred <= 0:
                update_count += 1
                weight = weight + lr * y[j] * _train_x[j] - lr * l2_reg * weight
                bias = bias + lr * y[j]
    print('The number of weight/bias updates: {}'.format(update_count))
    return weight, bias


def set_to_target(label, target_class=1):
    """
    determine which class to be classified
    """
    return np.array([1 if label == target_class else -1 for label in label])


def predict(x, weight, bias):
    """
    Predict the label of the input vector
    """
    return [1 if single_perceptron(x[i], weight, bias) >= 0 else -1 for i in range(len(x))]


def one_rest_predict(x, perceptron_list):
    """
    predict label by compare and using argmax
    """
    y_pred = []
    for _i in x:
        score = [single_perceptron(_i, weight=j[0], bias=j[1]) for j in perceptron_list]
        y_pred.append(np.argmax(score))
    return y_pred


def accuracy(y_true, y_pred):
    """
    Calculate the accuracy of the prediction
    """
    return sum([1 if y_true[i] == y_pred[i] else 0 for i in range(len(y_true))]) / len(y_true)


def two_class_classification(_train_x, _train_y, _test_x, _test_y, shuffle=True):
    """
    Classification between two specific classes
    """
    class_name = ['class 2 and class 3', 'class 1 and class 3', 'class 1 and class2']
    train_accuracies = []
    test_accuracies = []
    for i in range(3):
        assert i in np.unique(_train_y)
        x_train_class = _train_x[np.where(_train_y != i)]
        y_train_class = _train_y[np.where(_train_y != i)]
        x_test_class = _test_x[np.where(_test_y != i)]
        y_test_class = _test_y[np.where(_test_y != i)]
        class_train = np.unique(y_train_class)

        y_train_class = set_to_target(y_train_class, target_class=class_train[0])
        y_test_class = set_to_target(y_test_class, target_class=class_train[0])
        weight, bias = train_perceptron(x_train_class, y_train_class, lr=LEARNING_RATE, epoch=EPOCH, shuffle=shuffle)
        y_pred_train = predict(x_train_class, weight, bias)
        y_pred_test = predict(x_test_class, weight, bias)
        train_accuracies.append(accuracy(y_train_class, y_pred_train))
        test_accuracies.append(accuracy(y_test_class, y_pred_test))
    [print('Train accuracy between {}: {:.2%}'.format(class_name[i], train_accuracies[i])) for i in
     range(len(train_accuracies))]
    print('-' * 100)
    [print('Test accuracy between {}: {:.2%}'.format(class_name[i], test_accuracies[i])) for i in
     range(len(test_accuracies))]
    print('Overall test accuracy: {:.2%}'.format(sum(train_accuracies) / len(train_accuracies)))


def one_rest_classification(train_x_, train_y_, test_x_, test_y_, shuffle=True, l2_reg=0.0):
    """
    Using one vs rest approach to classify
    """
    class_name = ['class 1 and rest', 'class 2 and rest', 'class 3 and rest']
    perceptron_list = []
    for i in range(len(class_name)):
        assert i in np.unique(train_y_)
        y_train_class = set_to_target(train_y_, target_class=i)
        weight, bias = train_perceptron(train_x_, y_train_class, lr=LEARNING_RATE, epoch=EPOCH, shuffle=shuffle, l2_reg=l2_reg)
        perceptron_list.append((weight, bias))
    y_test_pred = one_rest_predict(test_x_, perceptron_list)
    y_train_pred = one_rest_predict(train_x_, perceptron_list)
    print('Overall train accuracy {:.2%}'.format(accuracy(train_y_, y_train_pred)))
    print('-' * 100)
    print('Overall test accuracy: {:.2%}'.format(accuracy(test_y_, y_test_pred)))


# Hyperparameter tuning
np.random.seed(1)
LEARNING_RATE = 1e-2
EPOCH = 20
L2_REG = [0.01, 0.1, 1.0, 10.0, 100.0]
SHUFFLE = True

if __name__ == '__main__':
    train_x, train_y = load_data('CA1data/train.data')
    test_x, test_y = load_data('CA1data/test.data')

    print('#' * 100)
    print('Question 3: Classification between two classes\n', '-' * 100)
    two_class_classification(train_x, train_y, test_x, test_y, shuffle=SHUFFLE)
    print('#' * 100)

    print('Question 4: Classification between one class and rest\n', '-' * 100)
    one_rest_classification(train_x, train_y, test_x, test_y, shuffle=SHUFFLE, l2_reg=0)
    print('#' * 100)

    print('Question 5: Multi class classifier with L2 regularisation\n', '-' * 100)
    for l2 in L2_REG:
        print('L2_REG = {:.2e}'.format(l2))
        one_rest_classification(train_x, train_y, test_x, test_y, shuffle=SHUFFLE, l2_reg=l2)
        print('-' * 100)
