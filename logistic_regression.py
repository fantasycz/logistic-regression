import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import random


### Helper Functions ###
def logistic_fn(z):
    return 1.0 / (1.0 + np.exp(-z))
    # return .5 * (1 + np.tanh(.5 * z))


def load_data_pairs(type_str):
    return pd.read_csv("mnist_3s_and_7s/" + type_str + "_x.csv").values, pd.read_csv(
        "mnist_3s_and_7s/" + type_str + "_y.csv").values


def run_log_reg_model(x, beta):
    return logistic_fn(np.dot(x, beta))


def calc_log_likelihood(x, y, beta):
    log_likelihood = 0.0
    theta_hats = run_log_reg_model(x, beta)
    k = x.shape[0]
    for i in range(k):
        if y[i] == 1:
            log_likelihood = log_likelihood + np.log(theta_hats[i])
        else:
            log_likelihood = log_likelihood + np.log(1 - theta_hats[i])

    log_likelihood = log_likelihood / k

    return log_likelihood


def calc_accuracy(x, y, beta):
    theta_hats = run_log_reg_model(x, beta)
    k = x.shape[0]
    accuracy = 0.0
    for i in xrange(k):
        if (y[i] == 1 and theta_hats[i] >= 0.5) or (y[i] == 0 and theta_hats[i] < 0.5):
            accuracy += 1
    accuracy = accuracy / k

    return accuracy

### Adam Update

def get_AdaM_update(grad, adam_values, alpha_0=.001, b1=.95, b2=.999, e=1e-8):
    adam_values['t'] += 1

    # update mean
    adam_values['mean'] = b1 * adam_values['mean'] + (1 - b1) * grad
    m_hat = adam_values['mean'] / (1-b1**adam_values['t'])

    # update variance
    adam_values['var'] = b2 * adam_values['var'] + (1 - b2) * grad**2
    v_hat = adam_values['var'] / (1-b2**adam_values['t'])

    return alpha_0 * m_hat/(np.sqrt(v_hat) + e)

### NR Update

def get_NR_update(x, grad, theta_hats):
    a = theta_hats * (1 - theta_hats)
    a.shape = (a.shape[0],)
    a = np.diag(a)
    alpha =  np.dot(np.dot(x.T, a),x)
    if np.linalg.det(alpha) == 0:
        alpha += 0.001 * np.identity(alpha.shape[0])
    alpha = np.linalg.inv(alpha)

    return np.dot(alpha, grad)

def shuffle(x, y):
    x_stochastic = copy.deepcopy(x)
    y_stochastic = copy.deepcopy(y)

    for i in range(x.shape[0])[::-1]:
        index = random.randint(0,i)
        x_stochastic[[index, i], :] = x_stochastic[[i, index], :]
        y_stochastic[[index, i], :] = y_stochastic[[i, index], :]

    return x_stochastic, y_stochastic



### Model Training ###

def train_logistic_regression_model(x, y, beta, learning_rate, batch_size, max_epoch, lr_type):
    beta = copy.deepcopy(beta)
    n_batches = int(round(x.shape[0] / batch_size))
    train_progress = []
    count = 0

    for epoch_idx in xrange(max_epoch):

        if n_batches > 1:
            x_stochastic, y_stochastic = shuffle(x, y)
        else:
            x_stochastic = copy.deepcopy(x)
            y_stochastic = copy.deepcopy(y)

        for batch_idx in xrange(n_batches):
            x_current_batch = x_stochastic[batch_idx * batch_size:(batch_idx + 1) * batch_size, :]
            y_current_batch = y_stochastic[batch_idx * batch_size:(batch_idx + 1) * batch_size, :]
            # x_current_batch = x[batch_idx * batch_size:(batch_idx + 1) * batch_size, :]
            # y_current_batch = y[batch_idx * batch_size:(batch_idx + 1) * batch_size, :]
            count += 1
            theta_hats = run_log_reg_model(x_current_batch, beta)
            adam_values = {'mean': np.zeros(beta.shape), 'var': np.zeros(beta.shape), 't': 0}
            beta_grad = np.dot(x_current_batch.T, y_current_batch - theta_hats)
            # perform updates
            # Fix update
            if lr_type == 0:
                beta_update = learning_rate * beta_grad / batch_size
            # RMS update
            elif lr_type == 1:
                beta_update = 0.1 / count * beta_grad / batch_size
            # Adam update
            elif lr_type == 2:
                beta_update = get_AdaM_update(beta_grad, adam_values)
            # NR update
            else:
                beta_update = get_NR_update(x_current_batch, beta_grad, theta_hats) / batch_size
            beta += beta_update


        train_progress.append(calc_log_likelihood(x, y, beta))
        print "Epoch %d. Train Log Likelihood: %f" %(epoch_idx, train_progress[-1])

    return beta, train_progress


def initialize():
    # Load the data
    train_x, train_y = load_data_pairs("train")
    valid_x, valid_y = load_data_pairs("valid")
    test_x, test_y = load_data_pairs("test")

    # a Add a one for the bias term
    train_x = np.hstack([train_x, np.ones((train_x.shape[0], 1))])
    valid_x = np.hstack([valid_x, np.ones((valid_x.shape[0], 1))])
    test_x = np.hstack([test_x, np.ones((test_x.shape[0], 1))])

    return train_x, train_y, valid_x, valid_y, test_x, test_y


def run_training(train_x, train_y, valid_x, valid_y, test_x, test_y, learning_rates, batch_sizes, lr_type, max_epochs):
    # Initialize model parameters
    beta = np.random.normal(scale=.001, size=(train_x.shape[1], 1))

    # Iterate over training parameters, testing all combinations
    valid_ll = []
    valid_acc = []
    all_params = []
    all_train_logs = []

    for type in lr_type:
        for lr in learning_rates:
            for bs in batch_sizes:
                # train model
                final_params, train_progress = train_logistic_regression_model(train_x, train_y, beta, lr, bs, max_epochs, type)
                all_params.append(final_params)
                # all_train_logs.append((train_progress, "Initial learning rate: %f, Batch size: %d" % (lr, bs)))
                method = ['Robbins-Monro', 'AdaM', 'Newton-Raphson']
                # all_train_logs.append((train_progress, "%s, Batch size: %d" % (method[type-1], bs)))
                all_train_logs.append((train_progress, "Learing rate: %f, Batch size: %d" % (lr, bs)))

                # evaluate model on validation data
                valid_ll.append(calc_log_likelihood(valid_x, valid_y, final_params))
                valid_acc.append(calc_accuracy(valid_x, valid_y, final_params))

    # Get the best model
    best_model_idx = np.argmax(valid_acc)
    best_params = all_params[best_model_idx]
    test_ll = calc_log_likelihood(test_x, test_y, best_params)
    test_acc = calc_accuracy(test_x, test_y, best_params)
    print "Validation Log Likelihood: " + str(valid_ll)
    print "Test Log Likelihood: %f" %(test_ll)
    print "Validation Accuracies: " + str(valid_acc)
    print "Test Accuracy: %f" %(test_acc)

    # Plot
    plt.figure()
    epochs = range(max_epochs)
    for idx, log in enumerate(all_train_logs):
        plt.plot(epochs, log[0], '--', linewidth=3, label="Training, " + log[1])
        plt.plot(epochs, max_epochs * [valid_ll[idx]], '--', linewidth=5, label="Validation, " + log[1])
    plt.plot(epochs, max_epochs * [test_ll], '*', ms=8, label="Testing, " + all_train_logs[best_model_idx][1])

    plt.xlabel(r"Epoch ($t$)")
    plt.ylabel("Log Likelihood")
    plt.ylim([-.8, 0.])
    plt.title("MNIST Results for Various Logistic Regression Models")
    plt.legend(loc=4)
    plt.show()


if __name__ == "__main__":
    train_x, train_y, valid_x, valid_y, test_x, test_y = initialize()
    '''
    # problem 2 part 1 parameters
    learning_rates = [1e-3, 1e-2, 1e-1]
    # learning_rates = [1e-1]
    batch_sizes = [train_x.shape[0]]
    '''
    # problem 2 part 2 parameters
    learning_rates = [1e-3]
    batch_sizes = [1000,100,10]
    '''
    # problem 3 parameters
    learning_rates = [1e-1]
    batch_sizes = [200]
    '''
    # learning rate, 0: fix  1: RMS  2:Adam  3: NR
    lr_type = [0]
    max_epochs = 250

    run_training(train_x, train_y, valid_x, valid_y, test_x, test_y, learning_rates, batch_sizes, lr_type, max_epochs)
