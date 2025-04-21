import random
from battleshit.models.neural_nets import *
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import math


#####################################################################################
#####################################################################################

# Agent class for preDQN agents

#####################################################################################
#####################################################################################


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

    def train(self, num_steps, test_freq, num_trials):
        raise NotImplementedError("preDQN subclasses should implement this method.")
        
    def test(self, num_episodes):
        raise NotImplementedError("preDQN subclasses should implement this method.")
    
    def save_model(self, path):
        '''
        Saves the model to the specified path.
        
        Args:
            path (str): The path where the model will be saved.
        '''
        torch.save(self.model, path)
        print(f"Model saved to {path}.")


    def plot_performance_graph(self, path):
        '''
        Plots and saves the performance graph obtained during training, i.e. it plots
        the average reward obtained during testing episodes. The graph is saved to the specified path.
        '''
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
        # Initialize empty list for storing testing data
        self.testing_data = []
        
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
            action = torch.argmax(q_values, dim = -1).item() if random.random() > self.epsilon else random.randint(0, q_values.shape[1]-1)
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

        

    def train(self, num_steps, test_freq, num_trials):
        '''
        Implements training loop for the agent. During training exploratory actions are selected.
        
        Args:
            num_steps (int): The number of intaractions of the agent with the environment.
            test_freq (int): The frequency of testing the agent during training. For example, if   set to 10 it means testing will be done every 10 episodes. If set to 0 it means no testing is performed.
            num_trials (int): The number of testing episodes.
        '''
        # Reset environment
        state, _ = self.env.reset()
        # Keep track of the number of episodes
        num_episodes = 0
        # Put model in training mode
        self.model.train()
        # Set up training loop
        for step in tqdm(range(num_steps)):
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
            if done:
                # Increment episode count
                num_episodes += 1
                # Check if testing is needed
                if test_freq > 0 and num_episodes % test_freq == 0:
                    # Test the agent
                    mean_reward = self.test(num_trials)
                    # Store testing data
                    self.testing_data.append((num_episodes, mean_reward))
                    # Print testing data
                    print(f"Episode {num_episodes}: Mean reward: {mean_reward:.2f}")
                # Reset environment 
                state, _ = self.env.reset() 
            else:
                state = next_state
            
                

    def test(self, num_episodes):
        '''
        Implements testing loop for the agent. During testing greedy actions are selected.
        
        Args:
            num_episodes (int): The number of testing episodes.
        '''
        # Put model in evaluation mode
        self.model.eval()
        # Initialize rewards list
        episode_rewards = []
        # Switch off gradient tracking
        with torch.no_grad():
            # Loop over episodes
            for _ in range(num_episodes):
                # Reset environment
                state, _ = self.env.reset()
                done = False
                # Loop until episode is done
                while not done:
                    # Select action
                    action = self.select_greedy_action(state)
                    # Take action in environment and collect experience
                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                    # Store reward
                    episode_rewards.append(reward)
                    # Check if episode is done
                    done = terminated or truncated
                    # Update state
                    state = next_state if not done else self.env.reset()[0]
        # Put model back into training mode
        self.model.train()
        # Return average reward over all episodes
        return sum(episode_rewards) / num_episodes
    

    def plot_performance_graph(self, path):
        '''
        Plots and saves the performance graph obtained during training, i.e. it plots
        the average reward obtained during testing episodes. The graph is saved to the specified path.
        
        Specifically it plots self.testing_data which is a list of tuples (episode, mean_reward).
        
        Args:
            path (str): The path where the graph will be saved.
        '''
        # Unpack testing data
        episodes, mean_rewards = zip(*self.testing_data)
        # Plot data
        plt.figure(figsize=(16, 10))
        plt.plot(episodes, mean_rewards, marker='o', linestyle='-', color='b')
        plt.title("Agent Performance During Training")
        plt.xlabel("Episode")
        plt.ylabel("Mean Reward")
        # Save plot to file with the name of the simulation
        plt.tight_layout()
        plt.savefig(f"{path}.png")
        plt.close()


#####################################################################################
#####################################################################################

# Agent class for DQN agent

#####################################################################################
#####################################################################################

class DQNAgent(nn.Module):
    '''
    A DQN agent class. It uses experience replay and a target network for stability. The agent interacts 
    with the environment using an epsilon-greedy policy with exponentially decaying epsilon for exploration and exploitation.
    '''
    def __init__(self, online_model, target_model, env, buffer, 
                 init_epsilon, end_epsilon, decay_ratio, gamma, optimizer, tau):
        '''
        Args:
            online_model (torch.nn.Module): The neural network model used by the agent.
            target_model (torch.nn.Module): The target network used for bootstrapping.
            buffer (battleshit.buffers): The experience replay buffer used for storing experiences.
            env (gym.Env or custom): The environment in which the agent operates.
            init_epsilon (float): The initial value of exploration rate for the epsilon-greedy policy.
            end_epsilon (float): The final value of exploration rate for the epsilon-greedy policy.
            decay_ratio (float): The ratio for decaying epsilon.
            gamma (float): The discount factor for future rewards.
            optimizer (torch.optim.Optimizer): The optimizer used for training the model.
            tau (float): The soft update parameter for the target network.
        '''
        super().__init__()
        # Assign attributes
        self.online_model = online_model
        self.target_model = target_model
        self.env = env
        self.buffer = buffer
        self.init_epsilon = init_epsilon
        self.end_epsilon = end_epsilon
        self.decay_ratio = decay_ratio
        self.gamma = gamma
        self.optimizer = optimizer
        self.tau = tau
        # Set target model to evaluation mode
        self.target_model.eval()
        # Initialize empty list for storing testing data
        self.testing_data = []
        # Initialize time parameter (needed for epsilon decay)
        self._time = 0
        # Make the two networks initially equal (same weights)
        self.target_model.load_state_dict(self.online_model.state_dict())
        
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
            q_values = self.online_model(state)
            # Select the action with the highest Q-value
            action = torch.argmax(q_values, dim = -1).item() 
        return action

    def select_exploratory_action(self, state):
        '''
        Selects an action using an epsilon-greedy policy.
        
        Args:
            state (torch.Tensor): The current state of the environment.
        '''
        # Calculate current epsilon value
        epsilon = self.end_epsilon + (self.init_epsilon - self.end_epsilon) * math.exp(-self._time / self.decay_ratio)
        epsilon = max(epsilon, self.end_epsilon)
        # Increment time parameter 
        self._time += 1
        # Switch off gradient tracking
        with torch.no_grad():
            # Get the Q-values for the current state
            q_values = self.online_model(state)
            # Select an action using epsilon-greedy policy
            action = torch.argmax(q_values, dim = -1).item() if random.random() > epsilon else random.randint(0, q_values.shape[1]-1)
        return action
    
    def calculate_loss(self, batch_size):
        '''
        Calculates the loss for the batch of experiences sampled from the buffer.
        
        Args:
            batch_size (int): The size of the batch sampled from the buffer.
        
        Returns:
            loss (torch.Tensor): The calculated loss.
        '''
        # Sample a batch of experiences from the buffer
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        # Get the maximum Q-value for the next state --> detach to avoid backpropagation through the next action
        max_next_q_values = torch.amax(self.target_model(next_states).detach(), dim = 1)
        # Calculate target using Q-learning
        targets = rewards + self.gamma * max_next_q_values * (1 - dones)
            
        # Get the Q-value for the current state and action
        current_q_values = self.online_model(states)[range(batch_size), actions]
        # Calculate mean loss over the batch
        loss = torch.mean((targets - current_q_values)**2)
        # Return loss
        return loss

    def _update_target_network(self):
        '''
        Updates the target network using soft update.
        '''
        # Get the parameters of the target network
        target_net_state_dict = self.target_model.state_dict()
        # Get the parameters of the online model
        online_net_state_dict = self.online_model.state_dict()
        # Iterate over all parameters of the target network
        for key in target_net_state_dict:
            # Update the parameters of the target network using soft update
            target_net_state_dict[key] = online_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        # Load the updated parameters into the target network
        self.target_model.load_state_dict(target_net_state_dict)
    
    def _prepopulate_buffer(self, size):
        '''
        Prepopulates the buffer with random experiences.
        
        Args:
            size (int): The number of experiences to prepopulate the buffer with.
        '''
        # Reset environment
        state, _ = self.env.reset()
        # Save "size" experienes in the buffer
        for _ in range(size):
            # Select action
            action = random.randint(0, self.online_model.output_dim-1)
            # Take action in environment and collect experience
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            # Check if episode is done
            done = terminated or truncated
            # Store experience in buffer
            self.buffer.store((state, action, reward, next_state, done))
            # Update state
            state = next_state if not done else self.env.reset()[0]
        

    def train(self, num_steps, batch_size, steps_in_env, update_freq, test_freq, num_trials):
        '''
        Implements training loop for the agent. During training exploratory actions are selected.
        
        Args:
            num_steps (int): The number of intaractions of the agent with the environment.
            batch_size (int): The size of the batch sampled from the buffer.
            steps_in_env (int): The number of steps taken in the environment before sampling a batch from the buffer.
            update_freq (int): The frequency of updating the target network. For example, if set to 10 it means the target network will be updated every 10 steps.
            test_freq (int): The frequency of testing the agent during training. For example, if   set to 10 it means testing will be done every 10 episodes. If set to 0 it means no testing is performed.
            num_trials (int): The number of testing episodes.
        '''
        # Prepopulate buffer with random experiences
        self._prepopulate_buffer(self.buffer.min_capacity)
        # Reset environment
        state, _ = self.env.reset()
        # Keep track of the number of episodes
        num_episodes = 0
        # Put model in training mode
        self.online_model.train()
        # Set up training loop
        for step in tqdm(range(1, num_steps)):
            # Sample a batch of experiences from the buffer
            if step % steps_in_env == 0:
                # Calculate loss
                loss = self.calculate_loss(batch_size)
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            # Update target network
            if step % update_freq == 0:
                self._update_target_network()
            # Act for steps_in_env number of steps before sampling a batch from the buffer
            # Select action
            action = self.select_exploratory_action(state)
            # Take action in environment and collect experience
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            # Check if episode is done
            done = terminated or truncated
            # Store experience in buffer
            self.buffer.store((state, action, reward, next_state, done))
            # Update state
            if done:
                # Increment episode count
                num_episodes += 1
                # Check if testing is needed
                if test_freq > 0 and num_episodes % test_freq == 0:
                    # Test the agent
                    mean_reward = self.test(num_trials)
                    # Store testing data
                    self.testing_data.append((num_episodes, mean_reward))
                    # Print testing data
                    print(f"Episode {num_episodes}: Mean reward: {mean_reward:.2f}")
                # Reset environment 
                state, _ = self.env.reset() 
            else:
                state = next_state
            
                

    def test(self, num_episodes):
        '''
        Implements testing loop for the agent. During testing greedy actions are selected.
        
        Args:
            num_episodes (int): The number of testing episodes.
        '''
        # Put model in evaluation mode
        self.online_model.eval()
        # Initialize rewards list
        episode_rewards = []
        # Switch off gradient tracking
        with torch.no_grad():
            # Loop over episodes
            for _ in range(num_episodes):
                # Reset environment
                state, _ = self.env.reset()
                done = False
                # Loop until episode is done
                while not done:
                    # Select action
                    action = self.select_greedy_action(state)
                    # Take action in environment and collect experience
                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                    # Store reward
                    episode_rewards.append(reward)
                    # Check if episode is done
                    done = terminated or truncated
                    # Update state
                    state = next_state if not done else self.env.reset()[0]
        # Put model back into training mode
        self.online_model.train()
        # Return average reward over all episodes
        return sum(episode_rewards) / num_episodes
    
    def save_model(self, path):
        '''
        Saves the model to the specified path.
        
        Args:
            path (str): The path where the model will be saved.
        '''
        torch.save(self.online_model, path)
        print(f"Model saved to {path}.")

    def plot_performance_graph(self, path):
        '''
        Plots and saves the performance graph obtained during training, i.e. it plots
        the average reward obtained during testing episodes. The graph is saved to the specified path.
        
        Specifically it plots self.testing_data which is a list of tuples (episode, mean_reward).
        
        Args:
            path (str): The path where the graph will be saved.
        '''
        # Unpack testing data
        episodes, mean_rewards = zip(*self.testing_data)
        # Plot data
        plt.figure(figsize=(16, 10))
        plt.plot(episodes, mean_rewards, marker='o', linestyle='-', color='b')
        plt.title("Agent Performance During Training")
        plt.xlabel("Episode")
        plt.ylabel("Mean Reward")
        # Save plot to file with the name of the simulation
        plt.tight_layout()
        plt.savefig(f"{path}.png")
        plt.close()
        


#####################################################################################
#####################################################################################

# Agent class for Double DQN agent

#####################################################################################
#####################################################################################


class DoubleDQNAgent(DQNAgent):
    '''
    A Double DQN agent class. It inherits from the DQNAgent class.
    '''
    def __init__(self, online_model, target_model, env, buffer, 
                 init_epsilon, end_epsilon, decay_ratio, gamma, optimizer, tau):
        '''
        Args:
            online_model (torch.nn.Module): The neural network model used by the agent.
            target_model (torch.nn.Module): The target network used for bootstrapping.
            buffer (battleshit.buffers): The experience replay buffer used for storing experiences.
            env (gym.Env or custom): The environment in which the agent operates.
            init_epsilon (float): The initial value of exploration rate for the epsilon-greedy policy.
            end_epsilon (float): The final value of exploration rate for the epsilon-greedy policy.
            decay_ratio (float): The ratio for decaying epsilon.
            gamma (float): The discount factor for future rewards.
            optimizer (torch.optim.Optimizer): The optimizer used for training the model.
            tau (float): The soft update parameter for the target network.
        '''
        super().__init__(online_model, target_model, env, buffer, 
                         init_epsilon, end_epsilon, decay_ratio, gamma, optimizer, tau)
        
    def calculate_loss(self, batch_size):
        '''
        Calculates the loss for the batch of experiences sampled from the buffer.
        
        Args:
            batch_size (int): The size of the batch sampled from the buffer.
        
        Returns:
            loss (torch.Tensor): The calculated loss.
        '''
        # Sample a batch of experiences from the buffer
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        # Choose actions online model thinks are best for next states
        next_best_actions = torch.argmax(self.online_model(next_states).detach(), dim = 1)
        # Let target model score those next_best_actions
        targets = rewards + self.gamma * self.target_model(next_states).detach()[range(batch_size), next_best_actions] * (1 - dones)
        # Get the Q-value for the current state and action
        current_q_values = self.online_model(states)[range(batch_size), actions]
        # Calculate mean loss over the batch
        loss = torch.mean((targets - current_q_values)**2)
        # Return loss
        return loss
    
    
#####################################################################################
#####################################################################################

# Agent class for Double DQN agent using Prioritized Experience Replay

#####################################################################################
#####################################################################################
     
class PERDoubleDQNAgent(DoubleDQNAgent):
    '''
    A Double DQN agent class with Prioritized Experience Replay. It inherits from the DoubleDQNAgent class.
    '''
    def __init__(self, online_model, target_model, env, buffer, init_epsilon, end_epsilon, decay_ratio, gamma, optimizer, tau):
        '''
        Args:
            online_model (torch.nn.Module): The neural network model used by the agent.
            target_model (torch.nn.Module): The target network used for bootstrapping.
            buffer (battleshit.buffers.PrioritizedReplayBuffer): PER buffer used for storing experiences.
            env (gym.Env or custom): The environment in which the agent operates.
            init_epsilon (float): The initial value of exploration rate for the epsilon-greedy policy.
            end_epsilon (float): The final value of exploration rate for the epsilon-greedy policy.
            decay_ratio (float): The ratio for decaying epsilon.
            gamma (float): The discount factor for future rewards.
            optimizer (torch.optim.Optimizer): The optimizer used for training the model.
            tau (float): The soft update parameter for the target network.
        '''
        super().__init__(online_model, target_model, env, buffer, init_epsilon, end_epsilon, decay_ratio, gamma, optimizer, tau)
    
    
    def calculate_loss(self, batch_size):
        '''
        Calculates the loss for the batch of experiences sampled from the buffer.
        
        Args:
            batch_size (int): The size of the batch sampled from the buffer.
        
        Returns:
            loss (torch.Tensor): The calculated loss.
        '''
        # Sample a batch of experiences from the buffer
        states, actions, rewards, next_states, dones, indices, weights = self.buffer.sample(batch_size)
        # Choose actions online model thinks are best for next states
        next_best_actions = torch.argmax(self.online_model(next_states).detach(), dim = 1)
        # Let target model score those next_best_actions
        targets = rewards + self.gamma * self.target_model(next_states).detach()[range(batch_size), next_best_actions] * (1 - dones)
        # Get the Q-value for the current state and action
        current_q_values = self.online_model(states)[range(batch_size), actions]
        # Calculate TD errors
        td_errors = targets - current_q_values
        # Update priorities in the buffer for sampled experiences
        self.buffer.update_priorities(indices, td_errors)
        # Calculate mean loss over the batch
        loss = torch.mean((weights * td_errors)**2)
        # Return loss
        return loss


#####################################################################################
#####################################################################################

# REINFORCE agent with rewards-to-go and baselines calculated by critic network

#####################################################################################
#####################################################################################




online_model = DuellingFCN(input_dim = 4, output_dim = 2, hidden_dims = (32, 64), hidden_activation = torch.nn.ReLU(), output_activation = torch.nn.Identity())    
target_model = DuellingFCN(input_dim = 4, output_dim = 2, hidden_dims = (32, 64), hidden_activation = torch.nn.ReLU(), output_activation = torch.nn.Identity()) 


# agent = DoubleDQNAgent(
#     online_model = online_model, 
#     target_model = target_model, 
#     env = None, # Replace with your environment
#     buffer = None, # Replace with your buffer
#     init_epsilon = 1.0,
#     end_epsilon = 0.1,
#     decay_ratio = 10000,
#     gamma = 0.99,
#     optimizer = torch.optim.Adam(online_model.parameters(), lr=0.001),
#     tau = 0.9
# )


states = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=torch.float32)
# dones = torch.tensor([[0.0], [1.0]], dtype=torch.float32).squeeze()
# rewards = torch.tensor([[1.0], [2.0]], dtype=torch.float32).squeeze()
# actions = torch.tensor([[0], [1]], dtype=torch.long).squeeze()
# # print(torch.amax(target_model(states).detach(), dim=1))
# # print(rewards + torch.amax(target_model(states).detach(), dim=1)*(1-dones))
# print(actions)
# q_values = online_model(states)
# print(q_values)
# print(q_values[range(len(q_values)), actions])

# print(online_model(states))
# print(torch.argmax(online_model(states).detach(), dim = 1))
# next_best_actions = torch.argmax(online_model(states).detach(), dim = 1)

# print(target_model(states).detach()[range(len(states)), next_best_actions])
# print(torch.amax(online_model(states), dim = -1).detach().numpy())