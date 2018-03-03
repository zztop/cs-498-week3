import pandas as pd
from numpy.random import random
import random
from sklearn.preprocessing import scale
import tkinter

import matplotlib.pyplot as plt
import numpy as np

EPOCHS = 50
STEPS = 300

LAMBDAS = [0.001, 0.01, 0.1, 1]

ACCURACY_STEPS = 30


def label_converter(x):
    # print('{}'.format(x))
    return 1 if str(x).strip() != '<=50K' else -1


def evaluate_accuracy(a, b, evaluation_data):
    yay = 0
    for i, row in evaluation_data.iterrows():
        rowT = row.iloc[0:(row.shape[0] - 1)].T.values
        label = np.sign(np.dot(a.T, rowT)[0] + b)
        yay += 1 if label == row[-1] else 0

        # print('label {}'.format(label))
    return yay / evaluation_data.shape[0]


def train_and_validate(train, validate):
    a, b = np.zeros((validate.shape[1] - 1, 1)), 0
    accuracy_per_lambda = {}
    coefficient_per_lambda = {}
    coefficient_vector_per_lambda = {}
    for LAMBDA in LAMBDAS:
        accuracy = []
        coefficient_vector = []
        total_step_count = 1
        final_a, final_b = 0, 0
        for epoch in range(1, EPOCHS + 1):
            sample_train = train.sample(50 if train.shape[0] > 50 else train.shape[0])
            eta = 1 / (0.01 * epoch + 50)
            for step in range(1, STEPS + 1):
                if total_step_count % ACCURACY_STEPS == 0:
                    accuracy.append(evaluate_accuracy(a, b, sample_train))
                    coefficient_vector.append(np.dot(a.T, a)[0][0])
                total_step_count += 1
                idx = np.random.randint(validate.shape[0], size=1)
                validationK = validate.iloc[idx]
                xk = validationK.iloc[:, 0:(validationK.shape[1] - 1)]
                yk = validationK.iloc[0][validationK.shape[1] - 1]
                if (yk * (np.dot(a.T, xk.T)[0][0] + b)) >= 1:
                    a = np.subtract(a, eta * LAMBDA * a)
                    b = b
                else:
                    a = np.subtract(a, eta * np.subtract(LAMBDA * a, yk * xk.T))
                    b = b - (-1 * eta * yk)
                final_a = a
                final_b = b
        # accuracy_frame = pd.DataFrame(accuracy, list(range(1, len(accuracy) + 1)))

        accuracy_per_lambda[str(LAMBDA)] = accuracy
        coefficient_vector_per_lambda[str(LAMBDA)] = coefficient_vector
        coefficient_per_lambda[str(LAMBDA)] = [final_a, final_b]

    # plt.plot(list(range(1, len(accuracy) + 1)), accuracy_per_lambda, 'r--')
    plt.ioff()
    plt.subplot(2, 1, 1)
    for key in accuracy_per_lambda:
        plt.plot(list(range(1, len(accuracy_per_lambda[key]) + 1)), accuracy_per_lambda[key], label=str(key))
    plt.xlabel('Steps')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(2, 1, 2)
    for key in coefficient_vector_per_lambda:
        plt.plot(list(range(1, len(coefficient_vector_per_lambda[key]) + 1)), coefficient_vector_per_lambda[key],
                 label=str(key))
    plt.xlabel('Steps')
    plt.ylabel('Coefficient Vector')
    plt.legend()
    # plt.show()

    plt.savefig('accuracy and coefficient ' + str(random.randint(1, 100)) + '.png')

    for key in coefficient_per_lambda:
        print('LAMBDA = {} a = {} b = {}'.format(key, str(coefficient_per_lambda[key][0]),
                                                 str(coefficient_per_lambda[key][1])))
    return coefficient_per_lambda


if __name__ == "__main__":
    all_data = pd.read_csv('./adult.data',
                           names=None,
                           converters={14: label_converter}, na_values=['?'], sep=',\s')
    all_data = all_data.dropna()
    all_data_labels = all_data.iloc[:, -1:]
    all_data = all_data.drop(all_data.columns[[0, 1, 3, 5, 6, 7, 8, 9, 13, 14]], axis=1)
    all_data = pd.DataFrame(scale(all_data, with_mean=False, with_std=True))
    all_data['label'] = all_data_labels.values
    # fix this

    train_data = all_data.sample(frac=0.8, random_state=200)
    test_validate = all_data.drop(train_data.index)
    test = test_validate.sample(frac=0.5, random_state=200)
    validate_data = test_validate.drop(test.index)

    coeff_per_lambda = train_and_validate(train_data, validate_data)

    print('accuracy on training = ' + str(evaluate_accuracy(
        coeff_per_lambda['0.001'][0], coeff_per_lambda['0.001'][1], test)))

    print('done')

