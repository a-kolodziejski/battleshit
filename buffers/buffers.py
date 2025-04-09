from collections import deque
import random
import torch

class SimpleBuffer:
    '''
    A simple buffer implementation using deque from collections.
    This buffer has a maximum capacity and will overwrite the oldest items when full.
    '''
    def __init__(self, max_capacity=10):
        '''
        Initialize the buffer with a maximum capacity.
        
        Args:
            max_capacity (int): The maximum number of experience tuples the buffer can hold.
        '''
        self.max_capacity = max_capacity
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
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
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
    
    
# x = deque(maxlen=5)
buf = SimpleBuffer(max_capacity=5)
buf.store(([1,2,3],1,2,[4,5,6],0))
buf.store(([10,20,30],10,20,[40,50,60],0))
buf.store(([1.5,2.5,3.5],3,-4,[4.5,5.5,6.5],1))

states, actions, rewards, next_states, dones = buf.sample(2)
print("States:", states)
print("Actions:", actions)
print("Rewards:", rewards)
print("Next States:", next_states)
print("Dones:", dones)