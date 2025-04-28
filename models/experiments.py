'''
Perform experiments based on settings in experiments_config.toml file
'''
from battleshit.models.neural_nets import *
from battleshit.models.agents import *
from battleshit.environments.environments import *
from battleshit.buffers.buffers import *
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
            actor_input_dim = config_file[exp]['model'].get("actor_input_dim", None)
            actor_output_dim = config_file[exp]['model'].get("actor_output_dim", None)
            actor_hidden_dims = config_file[exp]['model'].get("actor_hidden_dims", None)
            critic_input_dim = config_file[exp]['model'].get("critic_input_dim", None)
            critic_output_dim = config_file[exp]['model'].get("critic_output_dim", None)
            critic_hidden_dims = config_file[exp]['model'].get("critic_hidden_dims", None)
            # Buffer-specific parameters
            buffer_kind = config_file[exp]['buffer'].get("buffer_kind", None)
            buffer_size = config_file[exp]['buffer'].get("buffer_size", None)
            buffer_min_capacity = config_file[exp]['buffer'].get("buffer_min_capacity", None)
            buffer_alpha = config_file[exp]['buffer'].get("alpha", None)
            buffer_beta = config_file[exp]['buffer'].get("beta", None)
            buffer_beta_increment = config_file[exp]['buffer'].get("beta_increment_per_sampling", None)
            buffer_epsilon = config_file[exp]['buffer'].get("epsilon", None)
            buffer_rank_based = config_file[exp]['buffer'].get("rank_based", None)
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
            agent_beta = config_file[exp]['agent'].get('beta', None)
            agent_greedy = config_file[exp]['agent'].get('greedy', None)
            agent_num_trajectories = config_file[exp]['agent'].get('num_trajectories', None)
            agent_numupdates = config_file[exp]['agent'].get('num_updates', None)
        
            # Set up appropriate objects
            # Environment
            if env_type == "gym":
                env = gym.make(env_name)
            else:
                env = globals()[env_name]()
        
            # Buffer
            if buffer_kind == "SimpleBuffer":
                buffer = globals()[buffer_kind](buffer_size, buffer_min_capacity)
            elif buffer_kind == "PrioritizedReplayBuffer":
                buffer = globals()[buffer_kind](buffer_size, buffer_min_capacity, buffer_alpha, buffer_beta, buffer_beta_increment, buffer_epsilon, buffer_rank_based)
            
            # Agent
            if agent_kind == "preDQNAgentNoBatch":
                # Neural Network (model)
                model = globals()[model_kind](model_input_dim,
                                        model_output_dim,
                                        model_hidden_dims,
                                        globals()[model_hidden_activation](),
                                        globals()[model_output_activation]())
                # Initialize agent
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
            
            elif agent_kind == "DQNAgent" or agent_kind == "DoubleDQNAgent" or agent_kind == "PERDoubleDQNAgent":
                # Initialize online model for DQN agent
                model = globals()[model_kind](model_input_dim,
                                        model_output_dim,
                                        model_hidden_dims,
                                        globals()[model_hidden_activation](),
                                        globals()[model_output_activation]())
                # Create target model for DQN agent
                target_model = globals()[model_kind](model_input_dim,
                                        model_output_dim,
                                        model_hidden_dims,
                                        globals()[model_hidden_activation](),
                                        globals()[model_output_activation]())
                # Initialize DQN agent
                agent = globals()[agent_kind](model, target_model, env, buffer,
                                              agent_init_epsilon, agent_end_epsilon,
                                              agent_decay_ration, agent_gamma,
                                              globals()[agent_optimizer](model.parameters(), agent_learning_rate), agent_tau)
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
            
            elif agent_kind == "REINFORCE":
                # Initialize actor and critic
                actor = globals()[model_kind](actor_input_dim,
                                        actor_output_dim,
                                        actor_hidden_dims,
                                        globals()[model_hidden_activation](),
                                        globals()[model_output_activation]())
                critic = globals()[model_kind](critic_input_dim,
                                        critic_output_dim,
                                        critic_hidden_dims,
                                        globals()[model_hidden_activation](),
                                        globals()[model_output_activation]())
                # Initialize optimizers for actor and critic
                actor_optimizer = globals()[agent_optimizer](actor.parameters(), agent_learning_rate)
                critic_optimizer = globals()[agent_optimizer](critic.parameters(), agent_learning_rate)
                # Initialize REINFORCE agent
                agent = globals()[agent_kind](actor, critic, env, agent_gamma,
                                              actor_optimizer, critic_optimizer, agent_beta)
                # Train agent
                agent.train(agent_num_trajectories, agent_numupdates, agent_test_freq, agent_num_trials, agent_greedy)
                # Save model
                if save_model:
                    agent.save_model(save_model_path + f"{exp}.pt")
                # Plot performance graph during training
                if save_graph:
                    if not os.path.exists(save_graph_path):
                        os.makedirs(save_graph_path)
                    agent.plot_performance_graph(save_graph_path + f"{exp}.png")
                

      
