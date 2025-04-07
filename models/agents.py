import random
from battleshit.models.neural_nets import *
import torch
import torch.nn as nn


class preDQNAgent(nn.Module):
    '''
    A simple preDQN (i.e. does not use replay buffer) agent with a feedforward neural network architecture. It uses either SARSA or Q-learning for bootstrapping. The agent interacts with the environment using an epsilon-greedy policy for exploration and exploitation.
    '''
    def __init__(self, model, env, bootstrap, epsilon, gamma, optimizer):
        '''
        Args:
            model (torch.nn.Module): The neural network model used by the agent.
            env (gym.Env or custom): The environment in which the agent operates.
            bootstrap (str): either 'sarsa' or 'qlearning', indicating the type of bootstrapping used.
            epsilon (float): The exploration rate for the epsilon-greedy policy.
            gamma (float): The discount factor for future rewards.
            optimizer (torch.optim.Optimizer): The optimizer used for training the model.
        '''
        super().__init__()
        # Assign attributes
        self.model = model
        self.env = env
        self.bootstrap = bootstrap
        self.epsilon = epsilon
        self.gamma = gamma
        self.optimizer = optimizer
        
    def select_greedy_action(self, state):
        raise NotImplementedError("preDQN subclasses should implement this method.")

    def select_exploratory_action(self, state):
        raise NotImplementedError("preDQN subclasses should implement this method.")
    
    def calculate_loss(self, state, action, reward, next_state, done):
        raise NotImplementedError("preDQN subclasses should implement this method.")

    def train(self, num_steps, test_freq):
        raise NotImplementedError("preDQN subclasses should implement this method.")
        

    def test(self, num_episodes):
        raise NotImplementedError("preDQN subclasses should implement this method.")
    


class preDQNAgentNoBatch(preDQNAgent):
    '''
    A simple preDQN (i.e. does not use replay buffer) agent with a feedforward neural network architecture. It updates NN generating behavior after single interaction step of the agent with the environment. It uses either SARSA or Q-learning for bootstrapping. The agent interacts with the environment using an epsilon-greedy policy for exploration and exploitation.
    '''
    def __init__(self, model, env, bootstrap, epsilon, gamma, optimizer):
        '''
        Args:
            model (torch.nn.Module): The neural network model used by the agent.
            env (gym.Env or custom): The environment in which the agent operates.
            bootstrap (str): either 'sarsa' or 'qlearning', indicating the type of bootstrapping used.
            epsilon (float): The exploration rate for the epsilon-greedy policy.
            gamma (float): The discount factor for future rewards.
            optimizer (torch.optim.Optimizer): The optimizer used for training the model.
        '''
        super().__init__(model, env, bootstrap, epsilon, gamma, optimizer)
        # Assign attributes
        # self.model = model
        # self.env = env
        # self.bootstrap = bootstrap
        # self.epsilon = epsilon
        # self.gamma = gamma
        # self.optimizer = optimizer
        
    def select_greedy_action(self, state):
        '''
        Selects the action with the highest Q-value for the current state.
        
        Args:
            state (torch.Tensor): The current state of the environment.
        
        Returns:
            action (int): The action with the highest Q-value.
        '''
        # Switch off gradient tracking
        with torch.no_grad():
            # Get the Q-values for the current state
            q_values = self.model(state)
            # Select the action with the highest Q-value
            action = torch.argmax(q_values, dim = -1).item() 
        return action

    def select_exploratory_action(self, state):
        '''
        Selects an action using an epsilon-greedy policy.
        
        Args:
            state (torch.Tensor): The current state of the environment.
        '''
        # Switch off gradient tracking
        with torch.no_grad():
            # Get the Q-values for the current state
            q_values = self.model(state)
            # Select an action using epsilon-greedy policy
            action = torch.argmax(q_values, dim = -1).item() if random.random() < self.epsilon else random.randint(0, q_values.shape[1]-1)
        return action
    
    def calculate_loss(self, state, action, reward, next_state, done):
        '''
        Calculates the loss for the current state-action pair. If bootstrap is 'sarsa', the loss is calculated using the SARSA algorithm. If bootstrap is 'qlearning', the loss is calculated using the Q-learning algorithm.
        
        Args:
            state (torch.Tensor): The current state of the environment.
            action (int): The action taken.
            reward (float): The reward received after taking the action.
            next_state (torch.Tensor): The next state of the environment.
            done (bool): Whether the episode has ended.
        
        Returns:
            loss (torch.Tensor): The calculated loss.
        '''
        # Calculate target
        if self.bootstrap == 'sarsa':
            # Select next action
            next_action = self.select_exploratory_action(next_state)
            # Get the Q-value for the next state and action --> detach to avoid backpropagation through the next action
            next_q_value = self.model(next_state)[0, next_action].detach()
            # Calculate target using SARSA
            target = reward + self.gamma * next_q_value * (1 - done)
        elif self.bootstrap == 'qlearning':
            # Get the maximum Q-value for the next state --> detach to avoid backpropagation through the next action
            max_next_q_value = torch.max(self.model(next_state).detach())
            # Calculate target using Q-learning
            target = reward + self.gamma * max_next_q_value * (1 - done)
            
        # Get the Q-value for the current state and action
        current_q_value = self.model(state)[0, action]
        # Calculate loss
        loss = (reward + self.gamma * target - current_q_value)**2
        # Return loss
        return loss

        

    def train(self, num_steps, test_freq, num_episodes):
        '''
        Implements training loop for the agent. During training exploratory actions are selected.
        
        Args:
            num_steps (int): The number of intaractions of the agent with the environment.
            test_freq (int): The frequency of testing the agent during training. For example, if   set to 10 it means testing will be done every 10 episodes. If set to 0 it means no testing is performed.
            num_episodes (int): The number of testing episodes.
        '''
        # Reset environment
        state, _ = self.env.reset()
        # Put model in training mode
        self.model.train()
        # Set up training loop
        for step in range(num_steps):
            # Select action
            action = self.select_exploratory_action(state)
            # Take action in environment and collect experience
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            # Check if episode is done
            done = terminated or truncated
            # Convert state and next_state to tensors
            state_tensor = torch.tensor(state, dtype = torch.float32).unsqueeze(0)
            next_state_tensor = torch.tensor(next_state, dtype = torch.float32).unsqueeze(0)
            # Calculate loss
            loss = self.calculate_loss(state_tensor, action, reward, next_state_tensor, done)
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # Update state
            state = next_state if not done else self.env.reset()[0]

    def test(self, num_episodes):
        '''
        Implements testing loop for the agent. During testing greedy actions are selected.
        
        Args:
            num_episodes (int): The number of testing episodes.
        '''
        # Put model in evaluation mode
        self.model.eval()
        # Switch off gradient tracking
        with torch.no_grad():
            # Loop over episodes
            for episode in range(num_episodes):
                # Reset environment
                state, _ = self.env.reset()
                done = False
                # Loop until episode is done
                while not done:
                    # Select action
                    action = self.select_greedy_action(state)
                    # Take action in environment and collect experience
                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                    # Check if episode is done
                    done = terminated or truncated
                    # Update state
                    state = next_state
        # Put model back into training mode
        self.model.train()
    
    

# x = torch.tensor([1.,2.,3.])
# # print(x.unsqueeze(0))

# model = SimpleFCN(input_dim = 3, output_dim = 2, hidden_dims = (4,5), hidden_activation = nn.ReLU(), output_activation = nn.Softmax(dim = -1))

# agent = preDQNAgentNoBatch(model = model, env = "lala", bootstrap = 'qlearning', epsilon = 0.1, gamma = 0.9, optimizer = torch.optim.Adam(model.parameters(), lr = 0.001))

# # print(agent.select_exploratory_action(x))
# print(agent.calculate_loss(x, 0, 1, x, False))

