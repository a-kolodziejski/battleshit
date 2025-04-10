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
import os


if __name__ == "__main__":
    
    # Read content of experiment_config.toml file to run the experiments
    import toml
    config_file = toml.load("models/experiment_config.toml")
#   Iterate over all active experiments from config file and run them
    for exp in config_file:
        if config_file[exp]["active"]:
            # Retrieve experiment objects and parameters
            save_model = config_file[exp]["save_model"]
            save_model_path = config_file[exp]["save_model_path"]
            # Performance graph parameters
            save_graph = config_file[exp]["save_graph"]
            save_graph_path = config_file[exp]["save_graph_path"]
            # Environment-specific parameters
            env_type = config_file[exp]["environment_type"] 
            env_name = config_file[exp]["environment"]
            # Model-specific parameters
            model_kind = config_file[exp]['model'].get("model_kind", None)
            model_input_dim = config_file[exp]['model'].get("input_dim", None)
            model_output_dim = config_file[exp]['model'].get("output_dim", None)
            model_hidden_dims = config_file[exp]['model'].get("hidden_dims", None)
            model_hidden_activation = config_file[exp]['model'].get("hidden_activation", None)
            model_output_activation = config_file[exp]['model'].get("output_activation", None)
            # Buffer-specific parameters
            buffer_kind = config_file[exp]['buffer'].get("buffer_kind", None)
            buffer_size = config_file[exp]['buffer'].get("buffer_size", None)
            # Agent-specific parameters
            agent_kind = config_file[exp]['agent'].get("agent_kind", None)
            agent_bootstrap = config_file[exp]['agent'].get("bootstrap", None)
            agent_epsilon = config_file[exp]['agent'].get("epsilon", None)
            agent_init_epsilon = config_file[exp]['agent'].get("init_epsilon", None)
            agent_end_epsilon = config_file[exp]['agent'].get("end_epsilon", None)
            agent_decay_ration = config_file[exp]['agent'].get("decay_ratio", None)
            agent_tau = config_file[exp]['agent'].get("tau", None)
            agent_gamma = config_file[exp]['agent'].get("gamma", None)
            agent_optimizer = config_file[exp]['agent'].get("optimizer", None)
            agent_learning_rate = config_file[exp]['agent'].get("learning_rate", None)
            agent_num_steps = config_file[exp]['agent'].get('num_steps', None)
            agent_batch_size = config_file[exp]['agent'].get('batch_size', None)
            agent_steps_in_env = config_file[exp]['agent'].get('steps_in_env', None)
            agent_update_freq = config_file[exp]['agent'].get('update_freq', None)
            agent_num_samples = config_file[exp]['agent'].get('num_samples', None)
            agent_num_trials = config_file[exp]['agent'].get('num_trials', None)
            agent_test_freq = config_file[exp]['agent'].get('test_freq', None)
        
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
            # Buffer
            if buffer_kind:
                buffer = globals()[buffer_kind](buffer_size)
            
            # Agent
            if agent_kind == "preDQNAgentNoBatch":
                agent = globals()[agent_kind](model,
                                        env,
                                        agent_bootstrap,
                                        agent_epsilon, 
                                        agent_gamma,
                                        globals()[agent_optimizer](model.parameters(),              agent_learning_rate))
                # Train agent
                agent.train(agent_num_steps, agent_test_freq, agent_num_trials)
                # Save model
                if save_model:
                    agent.save_model(save_model_path + f"{exp}.pt")
                # Plot performance graph during training
                if save_graph:
                    if not os.path.exists(save_graph_path):
                        os.makedirs(save_graph_path)
                    agent.plot_performance_graph(save_graph_path + f"{exp}.png")
            
            elif agent_kind == "DQNAgent":
                # Create target model for DQN agent
                target_model = globals()[model_kind](model_input_dim,
                                        model_output_dim,
                                        model_hidden_dims,
                                        globals()[model_hidden_activation](),
                                        globals()[model_output_activation]())
                # Initialize DQN agent
                agent = globals()[agent_kind](model, target_model, env, buffer,
                                              agent_init_epsilon, agent_end_epsilon,
                                              agent_decay_ration, agent_tau, agent_gamma,
                                              globals()[agent_optimizer](model.parameters(), agent_learning_rate))
                # Train agent
                agent.train(agent_num_steps, agent_batch_size, agent_steps_in_env,  agent_update_freq, agent_test_freq, agent_num_trials)
                
                # Save model
                if save_model:
                    agent.save_model(save_model_path + f"{exp}.pt")
                # Plot performance graph during training
                if save_graph:
                    if not os.path.exists(save_graph_path):
                        os.makedirs(save_graph_path)
                    agent.plot_performance_graph(save_graph_path + f"{exp}.png")
      
