
# coding: utf-8

# In[20]:

import numpy as np
import numpy.random as npr
from scipy.special import expit

#one hidden layer

class NNet(object):
    def __init__(self,n_features, n_hidden, n_output, iters = 100,                  minibatch = 1, eta = 0.01): 
        '''
        n_features: number of features of the very first layer, constant 
                    not included
        
        n_hidden:   number of features of the hidden layer
        
        n_output:   number of output (classification) of the last layer
        
        iters:      number of passes
        
        minibatch:  number of data points we want to use per round (forward +backward)
        
        eta:        gradient decent step size
        
        '''
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.eta = eta
        self.iters = iters
        self.minibatch = minibatch
        
        #initialize w1 and w2 by utility function initializeWeight()
        self.w1, self.w2 = self.initializeWeight()
    
        
    #####################################################
    #  
    #         Forward computation utility functions
    #
    #####################################################
        
        
    def initializeWeight(self):
        
        #from layer one to the hidden layer
        w1 = npr.uniform(-1.0, 1.0, size = self.n_hidden*(self.n_features + 1))
        w1 = w1.reshape(self.n_hidden, self.n_features + 1)
        
        #from hidden layer to the output layer
        w2 = npr.uniform(-1.0, 1.0, size = self.n_output*(self.n_hidden+1))
        w2 = w2.reshape(self.n_output, self.n_hidden+1)
        
        return w1, w2
        
        
        #tanh function at each cell
    def sigmoid(self, z):
        #return 1.0/(1.0+np.exp(-z))
        return expit(z)
        
    #layer 1        layer 2              layer 3
    #|a| ->      |z->sigmoid ->a| ->    |z|
    def forward(self, X, w1, w2):
        # X is a N x n_features data matrix
        #add constant = 1 to each data, to the first column
        a_input = np.ones([X.shape[0], X.shape[1] + 1])
        a_input[:, 1:] = X
        #a_input is a N x (n_features + 1) data matrix
        
        #compute the output of the hidden layer
        #a_hidden is a n_hidden x N matrix
        z_hidden = w1.dot(a_input.T)
        a_hidden = self.sigmoid(z_hidden)
        
        #add const one to each data
        #new_a_hidden is a (n_hidden+1)x N matrix
        new_a_hidden = np.ones([self.n_hidden + 1, a_hidden.shape[1]])
        new_a_hidden[1: , :] = a_hidden
        
        #compute the output, a (n_output x N) matrix
        z_output = w2.dot(new_a_hidden)
        a_output = self.sigmoid(z_output)
        
        #return result of every layer
        return a_input, z_hidden, new_a_hidden, z_output, a_output 
    
    #######################################################
    #  
    #         Backward computation utility functions
    #
    ########################################################
    
    
    #derivative of tanh
    def sigmoid_gradient(self,z):
        return self.sigmoid(z)*(1-self.sigmoid(z))
    
    #utility function to translate y into the form of a_output
    def encode_label(self, y, k):
        onespot = np.zeros((k, y.shape[0]))
        for idx, val in enumerate(y):
            onespot[val, idx] = 1
        return onespot
    
    
    def w_gradient(self, w1, w2, y_encoded, a_input, new_a_hidden, a_output, z_hidden):
        
        #(n_output x N) matrix
        delta3 = a_output - y_encoded
        
        #new_a_hidden has one more row of one's, so is w2
        #so we also add one row into to z_hidden
        #after computing delta2, we get rid of this row
        #delta2 is a (n_hidden x N) matrix
        new_z_hidden = np.ones([z_hidden.shape[0]+1, z_hidden.shape[1]])
        new_z_hidden [1:, :] = z_hidden
        delta2  = w2.T.dot(delta3)*self.sigmoid_gradient(new_z_hidden)
        delta2 = delta2[1:, :]   
    
        #gradient of w1, a (n_hidden x (n_features + 1)) matrix
        grad1 = delta2.dot(a_input)
        
        #gradient of w2, a (n_output x (n_hidden+1)) matrix
        grad2 = delta3.dot(new_a_hidden.T)
        
        #add regularizer
        grad1[:,1:] += (0.1*w1[:,1:])
        grad2[:,1:] += (0.1*w2[:,1:])
        
        return grad1, grad2
    
    ########################################
    #  
    #         Regularization
    #
    #######################################
    
    #Didn't use this at current point
    
    def L2(self, w1, w2):
        
        return np.sum(w1*w1)+ np.sum(w2*w2)
        
    ########################################
    #  
    #         Fit and Prediction
    #
    ####################################### 
    
    
    def predict(self, X):
        a_input, z_hidden, new_a_hidden, z_output, a_output =                          self.forward(X,self.w1, self.w2)
        #pre_y is a 1 x N matrix
        pre_y = np.argmax(z_output, axis = 0)
        return pre_y
    
    
    def fit(self, X, Y):
        X_data = X.copy()
        Y_data = Y.copy()
        
        
        
        #each iter, we shuffle the data set
        idx = npr.permutation(Y_data.shape[0])
        X_data = X_data[idx]
        Y_data = Y_data[idx]
        
        grad_w1_prev = np.zeros(self.w1.shape)
        grad_w2_prev = np.zeros(self.w2.shape)
        
        #translate the Y_data into the form of a_output
        y_encoded = self.encode_label(Y_data,self.n_output)
        
        
        for i in range(self.iters):
            
            self.eta = self.eta/(1+0.00001*i)
            
            mini = np.array_split(range(Y_data.shape[0]), self.minibatch)
            
            for index in mini:
                a_input, z_hidden, new_a_hidden, z_output, a_output =                 self.forward(X_data[index], self.w1, self.w2)
             
                grad_w1, grad_w2 = self.w_gradient(self.w1, self.w2,                                        y_encoded[:,index], a_input,                                         new_a_hidden, a_output, z_hidden)
                
                #update w1 and w2, also take the grad in the previous 
                #step into consideration
                self.w1 -= (self.eta*grad_w1 + 0.001* grad_w1_prev)
                self.w2 -= (self.eta*grad_w2 + 0.001* grad_w2_prev)
                 
                grad_w1_prev, grad_w2_prev = grad_w1, grad_w2
            
        return self
        



