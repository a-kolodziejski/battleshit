import torch
import torch.nn as nn

class preDQNAgent(nn.Module):
    '''
    A simple preDQN (i.e. does not use replay buffer) agent with a feedforward neural network architecture.
    '''
    def __init__(self, epsilon):
        '''
        Args:
            epsilon (float): The exploration rate for the epsilon-greedy policy.
        '''
        super().__init__(self)
        # Assign attributes
        self.epsilon = epsilon
        
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
        Implements training loop for the agent.
        
        Args:
            num_epochs (int): The number of training epochs.
        '''
        pass