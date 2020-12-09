import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

class UNIPoint(nn.Module):
    def __init__(self, batch_size, seq_len, n_features, n_parameters, n_basis_functions):
      """
      Input parameters:
      n_neurons - number of neurons inside RNN
      n_parameters - expecteed number of parameters in basis function
      n_basis_functions - number of basis functions
      """
      super(UNIPoint, self).__init__()

      self.rnn = nn.RNNCell(n_features, 1)
      self.hx = torch.randn(batch_size, 1) # initialize hidden state 
      self.h2p = nn.Linear(1, n_parameters * n_basis_functions)
      self.basis_res = torch.randn(batch_size, n_basis_functions) #initialize matrix for basis f-s calculations results
      self.Softplus = torch.nn.Softplus(beta = 1)

      self.seq_len = seq_len
      self.n_basis_functions = n_basis_functions

    def ReLU(self, parameter_1, parameter_2, time):
      """Function to apply Rectified Linear Unit (ReLU) as basis function inside network 
        Input parameters:
          parameters - alpha, beta for basis function's value calculation
          time - column-vector with time which had been spent since the begining of 
                  temporal point process (TPP)
      """
      self.output = torch.relu(self.parameters[:,parameter_1] * time + self.parameters[:,parameter_2] ) 
      return self.output
    
    def PowerLaw(self, parameter_1, parameter_2, time): # need to fix (see ReLU parameters and do the same)
      """Function to apply Power Law (PL) as basis function inside network 
        Input parameters:
          parameters - alpha, beta for basis function's value calculation
          time - column-vector with time which had been spent since the begining of 
                  temporal point process (TPP)
      """
      self.output = self.parameters[:,parameter_1] * (1 + time)**( - self.parameters[:,parameter_2])
      return self.output


    def forward(self, X):
      """Input parameters:
          X - batch with data 
          time - column-vector with interarrival time in temporal point process (TPP)
      """
        
      hidden_states, intensity_values = [], []
      
      # for each time step (here X shape is (batch_size, seq_len, n_features) )
      for i in range(self.seq_len):

          self.hx = self.rnn(X[:,i,:], self.hx)
          self.parameters = self.h2p(self.hx)
          
          for function in range(self.n_basis_functions): 
              # calculating numbers of parameters to take for basis function
              par1 = 2 * function
              par2 = 2 * function + 1
              self.basis_res[:, function] = self.ReLU(par1, par2, X[:,i,1]) # here X[:,i,1] - tau
          
          self.sum_res = torch.sum(self.basis_res, 1)

          self.intensity_res = self.Softplus(self.sum_res)

          hidden_states.append(self.hx)
          intensity_values.append(self.intensity_res)
          
          print("Sequence ", i+1, "out of", X.shape[1])
          
      return hidden_states, intensity_values
