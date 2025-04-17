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
        # Initialize an epty container of max_capacity size to act as the buffer.
        self.buffer = np.empty(max_capacity, dtype = np.ndarray)
        # Initialize a list to store priorities for each experience in the buffer.
        self.priorities = np.zeros(max_capacity, dtype=np.float32)
        # Initialize a counter to keep track of the number of experiences stored.
        self._counter = 0
        # To keep track of the current size of the buffer.
        self._size = 0

    def _increment_beta(self):
        '''
        Increment the beta value towards 1 for importance sampling.
        '''
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)
    
    def store(self, experience):
        '''
        Stores an experience tuple in the buffer. The experience tuple is expected
        to be of the form (state, action, reward, next_state, done).
        
        Args:
            experience (tuple): The experience tuple to store.
        '''
        # if buffer is empty set priority of the first experience to 1.0 else set it to max priority in priorities list
        priority = self.priorities.max() if self._counter else 1.0
        # Store the experience in the buffer at the current index.
        self.buffer[self._counter] = experience
        # Store the priority in the priorities list at the current index.
        self.priorities[self._counter] = priority
        # Increment the counter and wrap around if it exceeds max_capacity.
        self._counter = (self._counter + 1) % self.max_capacity
        # Increment the size of the buffer, ensuring it does not exceed max_capacity.
        self._size = min(self._size + 1, self.max_capacity)
    
    def update_priorities(self, indices, td_errors):
        '''
        Update the priorities of the experiences at the given indices.
        
        Args:
            indices (np.ndarray): The indices of the experiences to update.
            td_errors (torch.Tensor): batch of TD errors calculated for experiences in the buffer under  indices positions.
        '''
        # Convert the TD errors to absolute values and add epsilon to avoid zero priority.
        new_priorities = np.abs(td_errors.detach().numpy()) + self.epsilon
        # Update the priorities in the buffer at the given indices.
        self.priorities[indices] = new_priorities
    
    def sample(self, batch_size):
        '''
        Retrieves a batch of experiences from the buffer based on their priorities.
        
        Args:
            batch_size (int): The number of experience tuples to retrieve.
        
        Returns:
            batch of experiences and their indices and weights: A tuple containing the sampled experiences and their indices.
        '''
        # If the buffer is empty, return empty lists.
        if len(self.buffer) == 0:
            return [], [], [], [], [], [], []
        # Calculate the sampling probabilities based on priorities.
        if self.rank_based:
            # Rank-based sampling: sort priorities and calculate probabilities based on ranks.
            sorted_indices = np.argsort(self.priorities)[::-1]
            rank_based_priorities = np.zeros_like(self.priorities)
            rank_based_priorities[sorted_indices] = np.arange(1, self.max_capacity + 1)
        else:
            # Proportional sampling: calculate probabilities based on priorities.
            priorities = self.priorities
        # Calculate sampling probsabilities using the priorities and alpha.
        sampling_probs = priorities ** self.alpha
        sampling_probs /= sampling_probs.sum()
        
        # Sample indices based on the calculated probabilities.
        if self.rank_based:
            indices = np.random.choice(self.max_capacity, size=batch_size, replace=False, p=sampling_probs)
        else:
            indices = np.random.choice(self.max_capacity, size=batch_size, replace=False, p=sampling_probs)
        
        # Retrieve the sampled experiences
        experiences = self.buffer[indices]
        
        # Increment beta for importance sampling correction.
        self._increment_beta()
        
        # Calculate the importance sampling weights.
        weights = (self._size * sampling_probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Unzip the experiences into separate components.
        states, actions, rewards, next_states, dones = zip(*experiences)
        
        # Convert the lists to tensors of appropriate shape.
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        
        # Return collected tensors as a tuple along with indices and weights.
        return states, actions, rewards, next_states, dones, indices, weights


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

buffer = np.empty(5, dtype = np.ndarray)
buffer[0] = ([1,2,3],1,2,[4,5,6],0)
buffer[1] = ([10,20,30],10,20,[40,50,60],0)
buffer[2] = ([1.5,2.5,3.5],3,-4,[4.5,5.5,6.5],1)
# print(type(buffer[0]))
priorities = np.zeros(5, dtype=np.float32)
priorities[0] = 0.5
priorities[1] = 0.2
priorities[2] = 0.8
probabilities = priorities ** 0.6/np.sum(priorities ** 0.6)
# print(probabilities)  
# print(priorities.max())
# priorities[np.array([0,2])] = np.array([0.1, 0.1])
# print(priorities)

# print(np.random.choice(5, size=2, replace=False, p =probabilities))
# print(buffer[6])
print(np.argsort(probabilities)[::-1])