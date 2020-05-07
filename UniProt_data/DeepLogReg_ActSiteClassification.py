import tensorflow as tf
from tensorflow.python.framework import ops

import os
import sys

import numpy as np
import pandas as pd
from six.moves import urllib
import json
import math
import time
import argparse


def one_hot_matrix(labels, C):
    C = tf.constant(C,name='C')
    
    one_hot_matrix = tf.one_hot(labels,depth=C,axis=0)
    
    sess = tf.Session()
    one_hot = sess.run(one_hot_matrix)
    sess.close()
    
    return one_hot


def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    m = X.shape[1]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

# weights_initializer=tf.contrib.layers.xavier_initializer()
# biases_initializer=tf.zeros_initializer()

def create_placeholders(n_x,n_y):
    X = tf.placeholder(tf.float32, shape=[n_x, None])
    Y = tf.placeholder(tf.float32, shape=[n_y, None])
    return X,Y
def initialize_parameters(n_x, n_y, n_hidden, trained, weight_file='.'):
    n_list = [n_x]+n_hidden+[n_y]
    parameters = {}
    for idx in range(len(n_list)-1):
        W = 'W'+str(idx)
        b = 'b'+str(idx)
        n_curr = n_list[idx]
        n_next = n_list[idx+1]
        if trained:
            if weight_file == '.':
                raise ValueError('Provide weight_file')
            parameters[W] = tf.get_variable(W, [n_next, n_curr], initializer=tf.constant_initializer(np.load(os.path.join(weight_file, "top_model_weights_{}.npy".format(idx)))))
            parameters[b] = tf.get_variable(b, [n_next, 1], initializer=tf.constant_initializer(np.load(os.path.join(weight_file, "top_model_biases_{}.npy".format(idx)))))
        else:
            parameters[W] = tf.get_variable(W, [n_next, n_curr], initializer=tf.contrib.layers.xavier_initializer(seed=1))
            parameters[b] = tf.get_variable(b, [n_next, 1], initializer=tf.zeros_initializer())

    print('parameters: ', list(parameters.keys()))
    return parameters

def forward_prop(x, parameters):
    W0 = parameters["W0"]
    b0 = parameters["b0"]
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    z0 = tf.add(tf.matmul(W0,x),b0)
    z1 = tf.add(tf.matmul(W1,z0),b1)
    return z1
    
def compute_cost(z, y):
    logits = tf.transpose(z)
    labels = tf.transpose(y)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    
    return cost

def model(X_train, Y_train, X_test, Y_test, n_hidden, output_file, learning_rate = 0.0001,
          num_epochs = 1500, minibatch_size = 32, print_cost = True, trained=False, weight_file='.'):
    
    ops.reset_default_graph()                         
    tf.set_random_seed(1)                             
    seed = 3                                          
    (n_x, m) = X_train.shape                          
    n_y = Y_train.shape[0]                            
    costs = []                                        
    train_accuracy = []
    test_accuracy = []
    X, Y = create_placeholders(n_x, n_y)
    parameters = initialize_parameters(n_x, n_y, [n_hidden], trained, weight_file)
    Z = forward_prop(X, parameters)
    cost = compute_cost(Z, Y)
    
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        
        sess.run(init)
        
        for epoch in range(num_epochs):

            epoch_cost = 0.                       
            num_minibatches = int(m / minibatch_size) 
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:

                (minibatch_X, minibatch_Y) = minibatch
                
                _ , minibatch_cost = sess.run([optimizer,cost], feed_dict={X:minibatch_X,Y:minibatch_Y})
                
                epoch_cost += minibatch_cost / minibatch_size

            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost), end='\t')
                # Calculate the correct predictions
                correct_prediction = tf.equal(tf.argmax(Z), tf.argmax(Y))
                # Calculate accuracy on the test set
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                iter_train_accu = accuracy.eval({X: X_train, Y: Y_train})
                iter_test_accu = accuracy.eval({X: X_test, Y: Y_test})
                print ("Train Accuracy: ", iter_train_accu, end='\t')
                print ("Test Accuracy: ", iter_test_accu)
                train_accuracy.append(iter_train_accu)
                test_accuracy.append(iter_test_accu)
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                
        # plt.plot(np.squeeze(costs))
        # plt.ylabel('cost')
        # plt.xlabel('iterations (per fives)')
        # plt.title("Learning rate =" + str(learning_rate))
        # plt.show()

        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        correct_prediction = tf.equal(tf.argmax(Z), tf.argmax(Y))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:" , accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy: ", accuracy.eval({X: X_test, Y: Y_test}))
        
        train_accuracy = np.array(train_accuracy)
        test_accuracy = np.array(test_accuracy)
        costs = np.array(costs)

        if not os.path.isdir(output_file):
            os.mkdir(output_file)
        np.save(output_file+'/train_accuracy.npy', train_accuracy)
        np.save(output_file+'/test_accuracy.npy', test_accuracy)
        np.save(output_file+'/costs.npy', costs)
        
        return parameters


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training top model with fixed hidden representation')
    parser.add_argument('--input_file', type=str, required=True, help='File from which to read input data and labels')
    parser.add_argument('--weight_file', type=str, default='.', help='File from which to read weights and biases')
    parser.add_argument('--trained', action='store_true', default=False, help='Set to True if weights should be read from a file, set --weight_file')
    parser.add_argument('--output_file', type=str, required=True, help='File to write accuracy and costs')
    parser.add_argument('--rnn_size', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--num_epochs', type=int, required=True)
    parser.add_argument('--n_hidden', type=int, required=True)


    args = parser.parse_args()
    print(args)

    # UniRep parameters
    rnn_size = args.rnn_size
    initialize_mLSTM_from_scratch = False 
    initialize_toplayer_from_scratch = True # does not matter for babbler unless doing Evotuning
    batch_size = args.batch_size
    len_unirep = rnn_size

    num_class = 5
    n_hidden = args.n_hidden

    rep_mat = np.load(args.input_file)
    # Using only a small subset for testing
    print('Using only 1024 samples for testing')
    rep_mat = rep_mat[:,:1024]

    len_rep = rep_mat.shape[0]

    train_mask = np.arange(rep_mat.shape[1]) % 10 == 0
    train_set = rep_mat[:, ~train_mask]
    test_set = rep_mat[:, train_mask]
    X_train = train_set[:-1,:]
    Y_train = train_set[len_rep-1:len_rep,:]
    X_test = test_set[:-1,:]
    Y_test = test_set[len_rep-1:len_rep,:]

    Y_train_one_hot = one_hot_matrix(Y_train.T,C=5)
    Y_test_one_hot = one_hot_matrix(Y_test.T, C=5)

    Y_train_one_hot = Y_train_one_hot[:,:,0]
    Y_test_one_hot = Y_test_one_hot[:,:,0]

    print('X_train.shape: ', X_train.shape)
    print('Y_train_one_hot.shape: ', Y_train_one_hot.shape)
    print('Y_train.shape: ', Y_train.shape)
    print('Y_test_one_hot.shape: ', Y_test_one_hot.shape)

    params = model(X_train, Y_train_one_hot, X_test, Y_test_one_hot, n_hidden, args.output_file, 
        learning_rate=0.0001, num_epochs=args.num_epochs, minibatch_size=batch_size, trained=args.trained, 
        weight_file=args.weight_file)
    np.save(os.path.join(args.output_file, 'top_model_weights_0.npy') , params['W0'])
    np.save(os.path.join(args.output_file, 'top_model_biases_0.npy') , params['b0'])
    np.save(os.path.join(args.output_file, 'top_model_weights_1.npy') , params['W1'])
    np.save(os.path.join(args.output_file, 'top_model_biases_1.npy') , params['b1'])
    