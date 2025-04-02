import numpy as np
from tqdm import tqdm
from battleshit.heuristic.strategies import RandomPolicy, HeatMapPolicy


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
      if len(action) == 2:
        action = 10*action[0] + action[1]
      # Perform selected action
      next_state, reward, done, _, _ = env.step(action)
      # Increase counter variable
      count += 1
    # Save number of moves to end this game
    N_moves[game] = count
  return N_moves