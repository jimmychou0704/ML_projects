
# coding: utf-8

# In[18]:

import numpy as np
import numpy.random as npr
from scipy.special import expit

#two hidden layers

class NNet(object):
    def __init__(self,n_features, d1, d2, n_output, iters = 100,                  minibatch = 1, eta = 0.01): 
        '''
        n_features: number of features of the very first layer, constant 
                    not included
        
        d1:   number of features of the first hidden layer
        
        d2:   number of features of the first hidden layer
        
        n_output:   number of output (classification) of the last layer
        
        iters:      number of passes
        
        minibatch:  number of data points we want to use per round (forward +backward)
        
        eta:        gradient decent step size
        
        '''
        self.n_features = n_features
        self.d1 = d1
        self.d2 = d2
        self.n_output = n_output
        self.eta = eta
        self.iters = iters
        self.minibatch = minibatch
        
        #####################
        #  Do autoencoding here!!!!!!!!
        #######################
        #initialize w1 and w2 by utility function initializeWeight()
        self.w1, self.w2, self.w3 = self.initializeWeight()
    
        
    #####################################################
    #  
    #         Forward computation utility functions
    #
    #####################################################
        
        
    def initializeWeight(self):
        
        #from input layer  to the first hidden layer
        w1 = npr.uniform(-1.0, 1.0, size = self.d1*(self.n_features + 1))
        w1 = w1.reshape(self.d1, self.n_features + 1)

        #from first hidden layer to the second hidden layer
        w2 = npr.uniform(-1.0, 1.0, size = self.d2*(self.d1+1))
        w2 = w2.reshape(self.d2, self.d1+1)
        
        #from second hidden layer to the output layer
        w3 = npr.uniform(-1.0, 1.0, size = self.n_output*(self.d2+1))
        w3 = w3.reshape(self.n_output, self.d2+1)
        
        
        
        
        return w1, w2, w3
        
        
        #tanh function at each cell
    def sigmoid(self, z):
        #use expit to avoid overfload
        
        return expit(z)
        
   

    #layer 1        layer 2              layer 3
    #|a| ->      |z->sigmoid ->a| ->    |z|
    def forward(self, X, w1, w2, w3):
        # X is a N x n_features data matrix
        #add constant = 1 to each data, to the first column
        a_input = np.ones([X.shape[0], X.shape[1] + 1])
        a_input[:, 1:] = X
        #a_input is a N x (n_features + 1) data matrix
        
        #compute the output of the first hidden layer
        #a1 is a d1 x N matrix
        z1 = w1.dot(a_input.T)
        a1 = self.sigmoid(z1)
        
        #add const one to each data
        #new a1 is a (d1+1)x N matrix
        temp_a1 = np.ones([self.d1 + 1, a1.shape[1]])
        temp_a1[1: , :] = a1
        a1 = temp_a1
  
        #compute the output of the second hidden layer
        #a2 is a d2 x N matrix
        z2 = w2.dot(a1)
        a2 = self.sigmoid(z2)

        #add const one to each data
        #new a2 is a (d2+1)x N matrix
        temp_a2 = np.ones([self.d2 + 1, a2.shape[1]])
        temp_a2[1: , :] = a2
        a2 = temp_a2
        
        
        #compute the output, a (n_output x N) matrix
        z_output = w3.dot(a2)
        a_output = self.sigmoid(z_output)
        
        #return result of every layer
        
        return a_input, z1, a1, z2, a2, z_output, a_output
   

    #######################################################
    #  
    #         Backward computation utility functions
    #
    ########################################################
    
    
    #derivative of sigmoid
    def sigmoid_gradient(self,z):
        return self.sigmoid(z)*(1-self.sigmoid(z))
    
    #utility function to translate y into the form of a_output
    def encode_label(self, y, k):
        onespot = np.zeros((k, y.shape[0]))
        for idx, val in enumerate(y):
            onespot[val, idx] = 1
        return onespot
    
    
    #def w_gradient(self, w1, w2, w3, y_encoded,\
    #               a_input, new_a_hidden, a_output, z_hidden):
    def w_gradient(self, w1, w2, w3, y_encoded,                  a_input, z1, a1, z2, a2, a_output):
        #(n_output x N) matrix
        delta3 = a_output - y_encoded
        
        #a2 has one more row of one's, so is w3
        #so we also add one row into to z2
        #after computing delta2, we get rid of this row
        #delta2 is a (d2 x N) matrix
        temp_z2 = np.ones([z2.shape[0]+1, z2.shape[1]])
        temp_z2 [1:, :] = z2
        z2 = temp_z2
        delta2  = w3.T.dot(delta3)*self.sigmoid_gradient(z2)
        delta2 = delta2[1:, :]  
        
        
        #new_a_hidden has one more row of one's, so is w2
        #so we also add one row into to z1
        #after computing delta1, we get rid of this row
        #delta1 is a (d1 x N) matrix
        temp_z1 = np.ones([z1.shape[0]+1, z1.shape[1]])
        temp_z1[1:, :] = z1
        z1 = temp_z1
        delta1  = w2.T.dot(delta2)*self.sigmoid_gradient(z1)
        delta1 = delta1[1:, :]   
    
        #gradient of w1, a (d1 x (n_features + 1)) matrix
        grad1 = delta1.dot(a_input)
        
        #gradient of w2, a (d2 x (d1+1)) matrix
        grad2 = delta2.dot(a1.T)
        
        #gradient of w3, a (n_output x (d2+1)) matrix
        grad3 = delta3.dot(a2.T)
  




        #add regularizer
        grad1[:,1:] += (0.1*w1[:,1:])
        grad2[:,1:] += (0.1*w2[:,1:])
        grad3[:,1:] += (0.1*w3[:,1:])
        return grad1, grad2, grad3
    
  
        
    ########################################
    #  
    #         Fit and Prediction
    #
    ####################################### 
    
    
    def predict(self, X):
        a_input, z1, a1, z2, a2, z_output, a_output =                          self.forward(X,self.w1, self.w2, self.w3)
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
        grad_w3_prev = np.zeros(self.w3.shape)
        
        #translate the Y_data into the form of a_output
        y_encoded = self.encode_label(Y_data,self.n_output)
        
        
        for i in range(self.iters):
            
            #slow down when we get close to local min
            self.eta = self.eta/(1+0.00001*i)
            
            mini = np.array_split(range(Y_data.shape[0]), self.minibatch)
            
            for index in mini:
                a_input, z1, a1, z2, a2, z_output, a_output =                 self.forward(X_data[index], self.w1, self.w2, self.w3)
             
                grad_w1, grad_w2, grad_w3 = self.w_gradient(self.w1, self.w2,                self.w3, y_encoded[:,index], a_input,z1,a1,z2,a2, a_output)
                
                #update w1 and w2, also take the grad in the previous 
                #step into consideration
                self.w1 -= (self.eta*grad_w1 + 0.001* grad_w1_prev)
                self.w2 -= (self.eta*grad_w2 + 0.001* grad_w2_prev)
                self.w3 -= (self.eta*grad_w3 + 0.001* grad_w3_prev)
                 
                grad_w1_prev, grad_w2_prev, grad_w3_prev = grad_w1, grad_w2, grad_w3
            
        return self
        

