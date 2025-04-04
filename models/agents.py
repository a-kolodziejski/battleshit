import torch
import torch.nn as nn
import random
import numpy as np

class preDQNAgent(nn.Module):
    '''
    A simple preDQN (i.e. does not use replay buffer) agent with a feedforward neural network architecture.
    '''
    def __init__(self, model, bootstrap, epsilon, gamma, optimizer):
        '''
        Args:
            model (torch.nn.Module): The neural network model used by the agent.
            bootstrap (str): either 'sarsa' or 'qlearning', indicating the type of bootstrapping used.
            epsilon (float): The exploration rate for the epsilon-greedy policy.
            gamma (float): The discount factor for future rewards.
            optimizer (torch.optim.Optimizer): The optimizer used for training the model.
        '''
        super().__init__(self)
        # Assign attributes
        self.model = model
        self.bootstrap = bootstrap
        self.epsilon = epsilon
        self.gamma = gamma
        self.optimizer = optimizer
        
    def select_greedy_action(self, state):
        '''
        Selects the action with the highest Q-value for the current state.
        
        Args:
            state (torch.Tensor): The current state of the environment.
        
        Returns:
            action (int): The action with the highest Q-value.
        '''
        pass

    def select_exploratory_action(self, state):
        '''
        Selects an action using an epsilon-greedy policy.
        
        Args:
            state (torch.Tensor): The current state of the environment.
        '''
        pass
        

    def train(self, num_epochs):
        '''
        Implements training loop for the agent. During training exploratory actions are selected.
        
        Args:
            num_epochs (int): The number of training epochs.
        '''
        pass