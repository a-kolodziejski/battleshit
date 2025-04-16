from collections import deque
import random
import torch
import numpy as np

#####################################################################################
#####################################################################################

# SimpleBuffer - samples uniformly from the buffer.

#####################################################################################
#####################################################################################

class SimpleBuffer:
    '''
    A simple buffer implementation using deque from collections.
    This buffer has a maximum capacity and will overwrite the oldest items when full.
    '''
    def __init__(self, max_capacity, min_capacity):
        '''
        Initialize the buffer with a maximum capacity.
        
        Args:
            max_capacity (int): The maximum number of experience tuples the buffer can hold.
            min_capacity (int): The minimum number of experience tuples the buffer should hold   before sampling.
        '''
        self.max_capacity = max_capacity
        self.min_capacity = min_capacity
        # Initialize a deque with a maximum length to act as the buffer.
        self.buffer = deque(maxlen=max_capacity)

    def store(self, experience):
        '''
        Stores an experience tuple in the buffer. The experience tuple is expected 
        to be of the form (state, action, reward, next_state, done).
        
        Args:
            experience (tuple): The experience tuple to store.
        '''
        self.buffer.append(experience)

    def sample(self, batch_size):
        '''
        Retrieves a batch of experiences from the buffer. If the buffer has fewer
        elements than batch_size, it returns all stored experiences.
        
        Args:
            batch_size (int): The number of experience tuples to retrieve.
        
        Returns:
            list: A list of experience tuples sampled from the buffer.
        '''
        # If the buffer is empty, return an empty list.
        if len(self.buffer) == 0:
            return []
        # Randomly sample experiences from the buffer.
        # If the buffer has fewer elements than batch_size, return all stored experiences.
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        # Retrieve states, actions, rewards, next_states, and dones from the sampled batch.
        states, actions, rewards, next_states, dones = zip(*batch)
        # Convert the lists to tensors of appropriate shape.
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        # Return collected tensors as a tuple.
        return states, actions, rewards, next_states, dones

    def clear(self):
        '''
        Clears the buffer, removing all stored experiences.
        '''
        self.buffer.clear()

    def __len__(self):
        '''
        Returns the current size of the buffer.'''
        return len(self.buffer)
    

#####################################################################################
#####################################################################################

# PrioritizedReplayBuffer - samples from the buffer based on priority.

#####################################################################################
#####################################################################################

class PrioritizedReplayBuffer:
    '''
    A prioritized replay buffer that samples experiences based on their priority.
    Priorities are calculated using the TD errors.
    Proportional or rank-based sampling is used to select experiences
    '''
    def __init__(self, max_capacity, min_capacity, alpha, beta, beta_increment_per_sampling, epsilon, rank_based=False):
        '''
        Initialize the buffer with a maximum capacity.
        
        Args:
            max_capacity (int): The maximum number of experience tuples the buffer can hold.
            min_capacity (int): The minimum number of experience tuples the buffer should hold before sampling.
            alpha (float): The parameter for prioritization. 0 means no prioritization, 1 means full prioritization.
            beta (float): The initial value for importance sampling. 0 means no correction, 1 means full correction.
            beta_increment_per_sampling (float): The increment for beta per sampling.
            epsilon (float): A small value to avoid zero priority.
            rank_based (bool): If True, uses rank-based sampling instead of proportional sampling.
        '''
        # Initialize attributes
        self.max_capacity = max_capacity
        self.min_capacity = min_capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.epsilon = epsilon
        self.rank_based = rank_based
        # Initialize a deque with a maximum length to act as the buffer.
        self.buffer = deque(maxlen=max_capacity)
        # Initialize a list to store priorities for each experience in the buffer.
        self.priorities = np.zeros(max_capacity, dtype=np.float32)

    def _increment_beta(self):
        '''
        Increment the beta value towards 1 for importance sampling.
        '''
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)
    
    def store(self, experience):
        pass


# # x = deque(maxlen=5)
# buf = SimpleBuffer(max_capacity=5)
# buf.store(([1,2,3],1,2,[4,5,6],0))
# buf.store(([10,20,30],10,20,[40,50,60],0))
# buf.store(([1.5,2.5,3.5],3,-4,[4.5,5.5,6.5],1))

# states, actions, rewards, next_states, dones = buf.sample(2)
# print("States:", states)
# print("Actions:", actions)
# print("Rewards:", rewards)
# print("Next States:", next_states)
# print("Dones:", dones)