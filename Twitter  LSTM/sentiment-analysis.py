# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 18:46:00 2018

@author: Reza Marzban
"""
from string import punctuation
import csv
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from randombatches import random_mini_batches
from keras import backend as K
import keras
import tensorflow as tf
import time

# load doc into memory
def load_doc(filename):
	file = open(filename, 'r', encoding="utf8")
	text = file.read()
	file.close()
	return text

# turn a doc into clean tokens
def clean_doc(line, vocab):
    tokens = line.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [w for w in tokens if w in vocab]
    tokens = ' '.join(tokens)
    return tokens

# load all docs in a directory
def load_sentences(filename, vocab, is_train):
    lines = list()
    file = open(filename, 'r', encoding="utf8")
    doc=csv.reader(file,delimiter=',')
    row_count = sum(1 for row in doc)
    train_size=int(row_count*0.95)
    file = open(filename, 'r', encoding="utf8")
    doc=csv.reader(file,delimiter=',')
    if(is_train):
        for i,row in enumerate(doc):
            if(i>=train_size):
                break
            tweet=row[2]
            line = clean_doc(tweet, vocab)
            lines.append(line)
    else:
        for i,row in enumerate(doc):
            if(i<train_size):
                continue
            tweet=row[2]
            line = clean_doc(tweet, vocab)
            lines.append(line)
    return lines

def load_y(filename, is_train):
    Ys = list()
    file = open(filename, 'r', encoding="utf8")
    doc=csv.reader(file,delimiter=',')
    row_count = sum(1 for row in doc)
    train_size=int(row_count*0.95)
    file = open(filename, 'r', encoding="utf8")
    doc=csv.reader(file,delimiter=',')
    if(is_train):
        for i,row in enumerate(doc):
            if(i>=train_size):
                break
            y=row[1]
            Ys.append(y)
    else:
        for i,row in enumerate(doc):
            if(i<train_size):
                continue
            y=row[1]
            Ys.append(y)
    return Ys

# load embedding as a dict
def load_embedding(filename):
	file = open(filename,'r', encoding="utf8")
	lines = file.readlines()[1:]
	file.close()
	embedding = dict()
	for line in lines:
		parts = line.split()
		# key is string word, value is numpy array for vector
		embedding[parts[0]] = np.asarray(parts[1:], dtype='float32')
	return embedding

# create a weight matrix for the Embedding layer from a loaded embedding
def get_weight_matrix(embedding, vocab):
	vocab_size = len(vocab) + 1
	weight_matrix = np.zeros((vocab_size, 100))
	for word, i in vocab.items():
		weight_matrix[i] = embedding.get(word)
	return weight_matrix

def compute_cost(z, y, parameters, penalty):    
    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits =z,  labels =y)

#    l2_cost = penalty * (tf.nn.l2_loss(parameters['wc1']) + 
#                                       tf.nn.l2_loss(parameters['wc2']) +
#                                       tf.nn.l2_loss(parameters['wd1']) +
#                                       tf.nn.l2_loss(parameters['wout'])) 

#    loss = tf.reduce_mean(tf.add(cost, l2_cost, name='loss'))
    loss = tf.reduce_mean(cost)
    return loss

def create_placeholders(n_x, n_y):

    X = tf.placeholder(tf.float32,shape=(None,n_x),name='X')
    Y = tf.placeholder(tf.float32,shape=(None,n_y),name='Y')
    return X, Y

def initialize_parameters(lstmUnits):

    #Initializes parameters to build a neural network with tensorflow. 
   
    parameters = {
    "w" : tf.Variable(tf.truncated_normal([lstmUnits, 1])),
    "b" : tf.Variable(tf.constant(0.1, shape=[1]))
    }
    return parameters

def forward_propagation(X, parameters, lstmUnits,keep_prob):

    # load embedding from file
    raw_embedding = load_embedding(file+'_embedding_word2vec.txt')
    # get vectors in the right order
    embedding_vectors = get_weight_matrix(raw_embedding, tokenizer.word_index)
    # create the embedding layer
    embedding_layer = keras.layers.Embedding(vocab_size, 100, weights=[embedding_vectors], input_length=max_length, trainable=True)(X)

    lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
    lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=keep_prob)
    value, _ = tf.nn.dynamic_rnn(lstmCell, embedding_layer, dtype=tf.float32)
    weight = parameters['w']
    bias =  parameters['b']
    value = tf.transpose(value, [1, 0, 2])
    last = tf.gather(value, int(value.get_shape()[0]) - 1)
    out = (tf.matmul(last, weight) + bias)
    return out

def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,
          num_epochs = 500, minibatch_size = 32, print_cost = True, keep_prob=0.75, units = 64):

    num_epochs=num_epochs+1
    ops.reset_default_graph() 
    K.clear_session()                     
    (m,n_x) = X_train.shape                        
    n_y = Y_train.shape[1]                     
    costs = []   
    L2penalty=0.7/2                                
    
    X, Y = create_placeholders(n_x, n_y)
    # sets the number of nodes in each hidden layer:
    parameters = initialize_parameters(units)

    out = forward_propagation(X, parameters,units, keep_prob)
    cost = compute_cost(out, Y, parameters,(L2penalty/m))
    
    starter_learning_rate = learning_rate
    global_step = tf.Variable(0, trainable=False)
    end_learning_rate = learning_rate/1000
    decay_steps = 1000
    learn_rate = tf.train.polynomial_decay(starter_learning_rate, global_step,
                                              decay_steps, end_learning_rate,
                                              power=0.5)
    optimizer = tf.train.AdamOptimizer(learning_rate = learn_rate).minimize(cost)
    

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        K.set_session(sess)
        sess.run(init)
        for epoch in range(num_epochs):

            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            minibatches = random_mini_batches(X_train.T, Y_train.T, minibatch_size)
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                minibatch_X=minibatch_X.T
                minibatch_Y=minibatch_Y.T
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 1 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch == (num_epochs-1):
                print ("Final Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(epoch_cost)
                
                
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        #save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.round(tf.sigmoid(out)), Y)
        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 
        
        keep_prob=1.0
        print ("Train Accuracy:%", 100*accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:%", 100*accuracy.eval({X: X_test, Y: Y_test}))
        print(" ")
        
        return parameters

start_time = time.time()
# load the vocabulary
file="test.csv" 
vocab_filename = file+'_vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)

# load all training reviews
train_docs = np.array(load_sentences(file,vocab,True))
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_docs)
encoded_docs = tokenizer.texts_to_sequences(train_docs)
max_length = 25
Xtrain = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
ytrain = np.array(load_y(file,True))
ytrain=ytrain.reshape(ytrain.shape[0],1)
# load all test reviews
test_docs = np.array(load_sentences(file,vocab,False))
encoded_docs = tokenizer.texts_to_sequences(test_docs)
Xtest = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
ytest =  np.array(load_y(file,False))
ytest=ytest.reshape(ytest.shape[0],1)
# define vocabulary size (largest integer value)
vocab_size = len(tokenizer.word_index) + 1

print(" ")
print("Training set is consisted of "+str(ytrain.shape[0])+" Examples and Test set is consisted of "+str(ytest.shape[0])+" Examples. ")
print(" ")
parameters = model(Xtrain, ytrain, Xtest, ytest, learning_rate = 0.001,
          num_epochs = 13, minibatch_size = 32, units = 64)
print("Total time:"+str(time.time()-start_time))
