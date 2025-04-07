'''
Perform experiments based on settings in experiments_config.toml file
'''
from battleshit.models.neural_nets import *
from battleshit.models.agents import *
from battleshit.environments.environments import *
import torch
from torch.nn import *
from torch.optim import *
import gymnasium as gym
import random
import numpy as np


if __name__ == "__main__":
  # Read content of experiment_config.toml file to run the experiments
  import toml
  config_file = toml.load("models/experiment_config.toml")
#   Iterate over all active experiments from config file and run them
  for exp in config_file:
    if config_file[exp]["active"]:
      # Retrieve experiment objects and parameters
      # Environment-specific parameters
      env_type = config_file[exp]["environment_type"] 
      env_name = config_file[exp]["environment"]
      # Model-specific parameters
      model_kind = config_file[exp]['model']["model_kind"]
      model_input_dim = config_file[exp]['model']["input_dim"]
      model_output_dim = config_file[exp]['model']["output_dim"]
      model_hidden_dims = config_file[exp]['model']["hidden_dims"]
      model_hidden_activation = config_file[exp]['model']["hidden_activation"]
      model_output_activation = config_file[exp]['model']["output_activation"]
      # Agent-specific parameters
      agent_kind = config_file[exp]['agent']["agent_kind"]
      agent_bootstrap = config_file[exp]['agent']["bootstrap"]
      agent_epsilon = config_file[exp]['agent']["epsilon"]
      agent_gamma = config_file[exp]['agent']["gamma"]
      agent_optimizer = config_file[exp]['agent']["optimizer"]
      agent_learning_rate = config_file[exp]['agent']["learning_rate"]
      agent_num_steps = config_file[exp]['agent']['num_steps']
      
      # Set up appropriate objects
      # Environment
      if env_type == "gym":
        env = gym.make(env_name)
      else:
        env = globals()[env_name]()
    
      # Neural Network (model)
      model = globals()[model_kind](model_input_dim,
                                    model_output_dim,
                                    model_hidden_dims,
                                    globals()[model_hidden_activation](),
                                    globals()[model_output_activation]())
      # Agent
      agent = globals()[agent_kind](model,
                                    env,
                                    agent_bootstrap,
                                    agent_epsilon, 
                                    agent_gamma,
                                    globals()[agent_optimizer](model.parameters(),              agent_learning_rate))

      
