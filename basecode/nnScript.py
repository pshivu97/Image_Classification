import time
import json
import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import pickle


def write_config(data):
    with open("output.json", 'w') as json_file:
        json.dump(data, json_file)


def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer

    # Output:
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    return 1 / (1 + np.exp(-z))


def get_all_data_with_labels(mat, key):
    """"
    %Input:
    %   mat - Input object as dictionary
    %   key - Key to get Train and Test data

    %Output:
    %   Input data with labels for all the digits from 0-9
    """

    all_digits_data = None
    for i in range(10):
        train_mat = mat[key + str(i)]
        labels = np.full((train_mat.shape[0], 1), i)
        labeled_train_mat = np.concatenate((train_mat, labels), axis=1)

        if all_digits_data is None:
            all_digits_data = labeled_train_mat
        else:
            all_digits_data = np.concatenate((all_digits_data, labeled_train_mat), axis=0)
    return all_digits_data


def save_used_features(feature_selection_data):
    """
    %Input:
    %   feature_selection_data - Array of True/False indicating whether the feature at that column is
    %                            selected or not. True - Feature Selected, False - Feature eliminated

    Save the selected features in feature_indices
    """
    feature_count = 0
    global feature_indices

    for i in range(len(feature_selection_data)):
        if feature_selection_data[i] == False:
            feature_count += 1
            feature_indices.append(i)


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the
       training set
     test_data: matrix of training set. Each row of test_data contains
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    """Train Data with labels for all digits"""
    all_digits_train = get_all_data_with_labels(mat, 'train')

    np.random.shuffle(all_digits_train)

    train_data = all_digits_train[:, 0:784]
    train_label = all_digits_train[:, 784]

    """Change the range of train data from 0-1"""
    train_data = train_data / 255.0

    """Convert Data with labels for all digits"""
    all_digits_test = get_all_data_with_labels(mat, 'test')

    np.random.shuffle(all_digits_test)

    test_data = all_digits_test[:, 0:784]
    test_label = all_digits_test[:, 784]

    """Convert the range of test data from 0-1"""
    test_data = test_data / 255.0

    """Remove features which have the same data for all the training examples"""
    reference = train_data[0, :]
    feature_selection_data = np.all(train_data == reference, axis=0)

    save_used_features(feature_selection_data)

    """Feature Selection - Remove feature which are not needed"""
    train_data = train_data[:, ~feature_selection_data]
    test_data = test_data[:, ~feature_selection_data]

    """Split train data into train and validation data"""
    validation_data = train_data[50000:60000]
    validation_label = train_label[50000:60000]

    train_data = train_data[0:50000]
    train_label = train_label[0:50000]

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log
    %   likelihood error function with regularization) given the parameters
    %   of Neural Networks, thetraining data, their corresponding training
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.

    % Output:
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    training_sample_size = training_data.shape[0]

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_grad = np.array([])

    """
        FeedForward Propagation
    """
    output_layer, training_layer_bias, hidden_layer_bias = nn_feed_forward_propagation(w1, w2, training_data)

    """
        Error Calculation
    """

    """Initialise a matrix of size (training sample size * n of class)"""
    y_mat = np.full((training_sample_size, n_class), 0)

    """Actual Label column for each training data is set to 1"""
    for i in range(training_sample_size):
        y_mat[i][training_label[i]] = 1

    """Calculating error using negative log-likelihood error function"""
    error = (-1) * (np.sum(
        np.multiply(y_mat, np.log(output_layer)) + np.multiply((1.0 - y_mat), np.log((1.0 - output_layer)))) / (
                        training_sample_size))

    """
        Back Propagation
    """

    delta = output_layer - y_mat
    w2_gradient = np.dot(delta.T, hidden_layer_bias)

    training_layer_delta = np.dot(delta, w2) * (hidden_layer_bias * (1.0 - hidden_layer_bias))

    w1_gradient = np.dot(training_layer_delta.T, training_layer_bias)

    """Removing bias weight from gradient"""
    w1_gradient = w1_gradient[1:, :]

    """
        Regularization
    """
    regularization = (lambdaval / (2 * training_sample_size)) * (np.sum(np.square(w1)) + np.sum(np.square(w2)))
    obj_val = error + regularization

    regularized_gradient_w1 = ((w1_gradient + lambdaval * w1) / training_sample_size).flatten()
    regularized_gradient_w2 = ((w2_gradient + lambdaval * w2) / training_sample_size).flatten()

    obj_grad = np.concatenate((regularized_gradient_w1, regularized_gradient_w2), 0)

    return obj_val, obj_grad


def nn_feed_forward_propagation(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
        % Network.

        % Input:
        % w1: matrix of weights of connections from input layer to hidden layers.
        %     w1(i, j) represents the weight of connection from unit j in input
        %     layer to unit i in hidden layer.
        % w2: matrix of weights of connections from hidden layer to output layers.
        %     w2(i, j) represents the weight of connection from unit j in hidden
        %     layer to unit i in output layer.
        % data: matrix of data. Each row of this matrix represents the feature
        %       vector of a particular image

        % Output:
        % output layer data"""

    """Initialize bias and include it to input layer data"""
    bias1 = np.full((data.shape[0], 1), 1)
    layer1_data = np.concatenate((bias1, data), axis=1)

    """Calculate Input layer's output using sigmoid function"""
    layer1_output = sigmoid(np.dot(layer1_data, w1.T))

    """Initialize bias and include it to hidden layer data"""
    bias2 = np.full((layer1_output.shape[0], 1), 1)
    layer2_data = np.concatenate((bias2, layer1_output), axis=1)

    """Calculate Hidden layer's output using sigmoid function"""
    layer2_output = sigmoid(np.dot(layer2_data, w2.T))

    return layer2_output, layer1_data, layer2_data


def nnPredict(w1, w2, training_data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden
    %     layer to unit i in output layer.
    % data: matrix of data. Each row of this matrix represents the feature
    %       vector of a particular image

    % Output:
    % label: a column vector of predicted labels"""

    output = nn_feed_forward_propagation(w1, w2, training_data)[0]

    """Return the class which has the maximum probability"""
    return np.argmax(output, axis=1)


"""**Neural Network Script Starts here****"""

feature_indices = []

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 12

# set the number of nodes in output unit
n_class = 10

n_hidden_values = np.arange(20, 24, 4)
lambdavalues = np.arange(30, 35, 5)
mydict = {}
accuracyList = []

for lambdaval in lambdavalues:
    for n_hidden in n_hidden_values:
        output = {}
        # initialize the weights into some random matrices
        st_time = time.time()
        initial_w1 = initializeWeights(n_input, n_hidden)
        initial_w2 = initializeWeights(n_hidden, n_class)

        # unroll 2 weight matrices into single column vector
        initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

        # set the regularization hyper-parameter
        # lambdaval = 0

        args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

        # Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

        opts = {'maxiter': 50}  # Preferred value.

        nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

        # In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
        # and nnObjGradient. Check documentation for this function before you proceed.
        # nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)

        # Reshape nnParams from 1D vector into w1 and w2 matrices
        w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
        w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

        # Test the computed parameters

        predicted_label = nnPredict(w1, w2, train_data)

        # find the accuracy on Training Dataset

        print('\n Lamdaval: ' + str(lambdaval) + '  , hidden_val: ' + str(n_hidden))
        accuracy = str(100 * np.mean((predicted_label == train_label).astype(float)))
        print('\n Training set Accuracy:' + accuracy + '%')
        accuracyList.append(accuracy)
        keyName = str(lambdaval) + '_' + str(n_hidden)
        output['Training set Accuracy'] = accuracy
        
        predicted_label = nnPredict(w1, w2, validation_data)

        # find the accuracy on Validation Dataset

        print('\n Validation set Accuracy:' + str(
            100 * np.mean((predicted_label == validation_label).astype(float))) + '%')
        output['Validation set Accuracy'] = str(100 * np.mean((predicted_label == validation_label).astype(float)))
        predicted_label = nnPredict(w1, w2, test_data)

        # find the accuracy on Validation Dataset

        print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
        output['Test set Accuracy:'] = str(100 * np.mean((predicted_label == test_label).astype(float)))

        mydict[keyName] = output

        end_time = time.time()
        print("Total Time: ", end_time - st_time)
        # parameters = [feature_indices, int(n_hidden), w1, w2, int(lambdaval)]
        # pickle.dump(parameters, open('params' + '' + str(lambdaval) + '' + str(n_hidden) + '.pickle', 'wb'))



        pickleData = {
            "selected features": feature_indices,"n hidden": n_hidden,"w1": w1,"w2": w2,"lambda": lambdaval
        }
        pickle.dump(pickleData, open("params_"+str(lambdaval)+"_"+str(n_hidden)+".pickle", "wb"))

        pickleData = pickle.load( open("params_"+str(lambdaval)+"_"+str(n_hidden)+".pickle", "rb") )
print("Max accuracy: " + str(max(accuracyList)))
write_config(mydict)