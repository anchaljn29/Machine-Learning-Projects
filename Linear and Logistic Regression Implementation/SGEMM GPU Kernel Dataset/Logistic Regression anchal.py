import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Defining cost function
def cost_function(X, Y, Beta):
    xB = np.dot(X, Beta)
    predictedy = 1 / (1 + np.exp(-xB))
    identity = np.ones(len(predictedy))
    sum1 = (np.multiply(Y, np.log(predictedy)) + np.multiply((identity - Y), np.log((identity - predictedy))))
    return np.sum(sum1) / (-len(X))


# Gradient Descent function
def gradient_descent(x, y, B, learningrate, iters, exper2=False):
    cost_final = []
    b_list = []
    m = len(y)
    temp_B = B.copy()
    for i in range(iters):
        xB = np.dot(x, B)
        z = 1 / (1 + np.exp(-xB))
        for j in range(x.shape[1]):
            temp_B[j] = B[j] - (learningrate / m) * sum((z - y) * x[:, j])
        B = temp_B.copy()
        cost = cost_function(x, y, B)
        if exper2:
            # threshold for the cost reduction
            if len(cost_final) != 0:
                if abs(cost_final[-1] - cost) <= 0.0001:
                    print("Break happened at iteration = %d and previous cost = %f" % (i, cost_final[-1]))
                    cost_final.extend([0] * (len(iters) - i))
                    break
        b_list.append(B)
        cost_final.append(cost)
    minimum_cost = min(cost_final)
    minimum_index = cost_final.index(minimum_cost)
    bi = b_list[minimum_index]
    iteration = np.arange(iters)
    plt.style.use('seaborn-whitegrid')
    plt.plot(iteration, cost_final, color='blue')
    plt.xlabel("Number of iterations")
    plt.ylabel("Cost")
    plt.show()
    return bi, minimum_cost


def part1(x_train, x_test, y_train, y_test, Beta):
    maximum_iter = 5
    # learningrate_list = [0.00001, 0.0001, 0.001, 0.01, 0.1]
    learningrate_list = [0.00005]
    b_list = []
    cost_train = []
    cost_test = []
    for item in learningrate_list:
        beta, cost = gradient_descent(x_train, y_train, Beta, item, maximum_iter)
        b_list.append(Beta)
        cost_train.append(cost)
        cost_temp = cost_function(x_test, y_test, Beta)
        cost_test.append(cost_temp)

    # plot for training set
    figure, axis = plt.subplots()
    axis.plot(learningrate_list, cost_train)
    axis.set_ylabel('Training Set Cost')
    axis.set_xlabel('learningrate')
    axis.set_title('Training Set Cost vs. learningrate')
    # plot for testing set
    figure, axis = plt.subplots()
    axis.plot(learningrate_list, cost_test)
    axis.set_ylabel('Test Set Cost')
    axis.set_xlabel('learningrate')
    axis.set_title('Test Set Cost vs. learningrate')
    # calculating best parameters for training set
    min_cost_train = min(cost_train)
    min_index_train = cost_train.index(min_cost_train)
    bi = b_list[min_index_train]
    print('For %d iterations, Minimum cost for training set is %f at learningrate = %f.' % (
        maximum_iter, min_cost_train, learningrate_list[min_index_train]))
    min_cost_test = min(cost_test)
    min_index_test = cost_test.index(min_cost_test)
    if min_index_test == min_index_train:
        print('Minimum cost for test set is %f obtained at same set of beta values.' % (min_cost_test))
    else:
        print('Minimum cost for test set is %f obtained at different set of betas at learningrate = %f.' %
              (min_cost_test, learningrate_list[min_index_test]))


def part3(x_train, x_test, y_train, y_test, Beta):
    # randomly generating list of 8 integers to be used as column names.
    random.seed(40)
    random_features = random.sample(range(1, 15), 8)
    random_features.append(0)
    df_x_train = x_train[:, random_features]
    Beta = Beta[random_features]
    initial_Cost = cost_function(df_x_train, y_train, Beta)
    print('Initial Cost: %f' % initial_Cost)
    B, cost_train = gradient_descent(df_x_train, y_train, Beta, 0.0001, 1000, True)
    df_x_test = x_test[:, random_features]
    cost_test = cost_function(df_x_test, y_test, B)
    print("Training cost for 8 random features = %f" % cost_train)
    print("Test cost for 8 random features = %f" % cost_test)


def part4(x_train, x_test, y_train, y_test, Beta):
    selected_features = [0, 6, 7, 8, 9, 10, 11, 12, 13]
    df_x_train = x_train[:, selected_features]
    Beta = Beta[selected_features]
    initial_Cost = cost_function(df_x_train, y_train, Beta)
    print('Initial Cost: %f' % initial_Cost)
    B, cost_train = gradient_descent(df_x_train, y_train, Beta, 0.0001, 1000, True)
    df_x_test = x_test[:, selected_features]
    cost_test = cost_function(df_x_test, y_test, B)
    print("Training cost for 8 random features = %f" % cost_train)
    print("Test cost for 8 random features = %f" % cost_test)


def logisticRegression():
    # loading dataset
    df = pd.read_csv(
        r"C:\Users\ANCHAL\Documents\utd_coursework\machine_learning\sgemm_product.csv")
    # creating new column of average run time
    df['Runs_Aver'] = round((df['Run1 (ms)'] + df['Run2 (ms)'] + df['Run3 (ms)'] + df['Run4 (ms)']) / 4, 2)
    df.drop(['Run1 (ms)', 'Run2 (ms)', 'Run3 (ms)', 'Run4 (ms)'], inplace=True, axis=1)
    X = df.iloc[:, :14]  # X variables from dataset
    x = np.full([X.shape[0], 1], 1)  # X(0) variables added for beta(0) variables as 1
    X = np.hstack((x, X))  # full set of X variables
    Y = df["Runs_Aver"].values  # Y variable
    y_median = np.median(Y)
    Y = np.where(Y < y_median, 0, 1)
    Beta = np.zeros(X.shape[1])  # beta variables
    # splitting data into 70-30 ratio
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    # initial cost calculation
    initial_cost = cost_function(x_train, y_train, Beta)
    print("Initial cost is %f when all betas are 0" % initial_cost)
    part1(x_train, x_test, y_train, y_test, Beta)
    part3(x_train, x_test, y_train, y_test, Beta)
    part4(x_train, x_test, y_train, y_test, Beta)


logisticRegression()


def prebuilt_linear(x, y):
    from sklearn import linear_model
    regression = linear_model.logisticRegression()
    regression.fit(x, y)
    print('Coefficients: \n', regression.coef_)
