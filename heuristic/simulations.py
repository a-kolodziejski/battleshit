import numpy as np
from tqdm import tqdm
from battleshit.heuristic.strategies import *
from battleshit.environments.environments import *


def simulation(env, pi, num_games):
  """
  Helper function to run policy pi for num_games in a given environment.

  Args:
    env (battleship environment)
    pi (dict or callable): policy to evaluate
    num_games (int): number of games to play (number of episodes)
  Returns:
    N_moves (ndarray of ints): number of moves to end the game
  """
  # Initialize N_moves array
  N_moves = np.zeros(num_games, dtype = np.int32)
  # Main loop
  for game in tqdm(range(num_games)):
    # Reset environment
    state, _ = env.reset()
    # Get generated board
    board = env._board_generated
    # Reset policy
    if isinstance(pi, RandomPolicy):
      pi.reset()
    elif isinstance(pi, HeatMapPolicy):
      board = np.reshape(board, (10, 10))
      pi.reset(board)
    # Set flag to False
    done = False
    # Initialize count
    count = 0
    while not done:
      # Select next action
      action = pi(state)
      # Heatmap policy returns action in the form (row, column).
      # It needs to be converted into single int
      if type(action) == tuple:
        action = 10*action[0] + action[1]
      # Perform selected action
      next_state, reward, done, _, _ = env.step(action)
      # Increase counter variable
      count += 1
    # Save number of moves to end this game
    N_moves[game] = count
  return N_moves


if __name__ == "__main__":
  # Import necessary plotting libraries
  import matplotlib.pyplot as plt
  import seaborn as sns
  # Read content of hsim_config.toml file to run the simulations
  import toml
  config_file = toml.load("heuristic/hsim_config.toml")
  # Iterate over all active simulations from config file and run them
  for sim in config_file:
    if config_file[sim]["active"] == True:
      # Get simulation parameters 
      num_games = config_file[sim]["num_games"]
      strategy = config_file[sim]["strategy"]
      environment = config_file[sim]["environment"]
      
      # Instantiate environment and policy
      policy = globals()[strategy]()
      env = globals()[environment]()
      
      # Run simulation to obtain array of number of moves to end the game
      N_moves = simulation(env, policy, num_games)
      
      #Calulate statistics
      mean = np.mean(N_moves)
      std = np.std(N_moves)
      median = np.median(N_moves)
      min = np.min(N_moves)
      max = np.max(N_moves)
      
      # Make histogram if save_file is set to True
      if config_file[sim]["save_file"] == True:
        sns.displot(N_moves)
        plt.xlabel("# of moves")
        plt.title(f"Number of moves to complete the game for {strategy}")
        
        # Annotate plot with statistics
        plt.annotate(f"Mean: {mean:.2f}", xy=(0.05, 0.95), xycoords="axes fraction", fontsize=12, color="black")
        plt.annotate(f"Median: {median:.2f}", xy=(0.05, 0.90), xycoords="axes fraction", fontsize=12, color="black")
        plt.annotate(f"Std: {std:.2f}", xy=(0.05, 0.85), xycoords="axes fraction", fontsize=12, color="black")
        plt.annotate(f"Min: {min:.2f}", xy=(0.05, 0.80), xycoords="axes fraction", fontsize=12, color="black")
        plt.annotate(f"Max: {max:.2f}", xy=(0.05, 0.75), xycoords="axes fraction", fontsize=12, color="black")
        
        # Save plot to file
        # Create directory if it does not exist
        import os
        if not os.path.exists("heuristic/simulations"):
          os.makedirs("heuristic/simulations")
        # Save plot to file with the name of the simulation
        plt.tight_layout()
        plt.savefig(f"heuristic/simulations/{sim}.png")
      
      
