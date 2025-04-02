import numpy as np
import random

class RandomPolicy:
  """
  Implements random guessing startegy but with no repeated actions (board moves)
  """
  def __init__(self):
    # Define available action space
    self.action_space = [i for i in range(100)]

  def __call__(self, state):
    """
    Generates action for a given environment state. Since it is random guess
    startegy, it ignores state variable completely

    Args:
      state (ndarray): observed configuration of the board
    Returns:
      action (int): action to be taken i.e. board position to uncover
    """
    action = random.choice(self.action_space)
    self.action_space.remove(action)
    return action

  def reset(self):
    """
    Resets policy to its initial state (useful in simulations)
    """
    self.__init__()
    

class HeatMapPolicy:
  """
  Implements probabilistic guessing startegy (no repeated actions).
  Exploits the fact that with random ship placement some board positions
  have greater probability of being occupied. 
  Based on  https://towardsdatascience.com/coding-an-intelligent-battleship-agent-bf0064a4b319
  """
  def __init__(self, board):
    """
    Args:
      board (env.board_generated): board filled with ships, needed
                                  to keep track of hits and misses
    """
    # Assign board variable
    self.board = board
    # Initialize heatmap
    self.heatmap = np.zeros((10, 10))
    self.ship_sizes = [5, 4, 3, 3, 2]
    self._initialize_heatmap()
    # Define dictionary of proposed guesses along with information if they were hit or miss
    self.guess_dict = {}

  def _initialize_heatmap(self):
    for size in self.ship_sizes:
      rem_size = size - 1
      # Check for possible placements of a ship of given size
      for row in range(10):
        for col in range(10):
          # Check if ship can be placed to the up from given row/col position
          if row - rem_size >= 0:
            self.heatmap[(row-rem_size):(row+1), col] += 1
          # Check if ship can be placed to the left from given row/col position
          if col - rem_size >= 0:
            self.heatmap[row, (col - rem_size):(col+1)] += 1

  def _recalculate_index(self, index):
    """
    Helper function to express the given index as a tuple,
    for example, 54 = (5, 4)

    Args:
      index (int): index to recalculate
    Returns:
      (tuple of ints): recalculated index
    """
    return (index//10, index%10)

  def _recalculate_heatmap(self):
    """
    Function that updates heatmap using information about hit/miss positions.
    If some position was a hit it increases probability of picking up neighboring
    fields. In addition it sets probability to 0 at used guesses
    """
    # Initialize heatmap
    self.heatmap = np.zeros((10, 10))
    # Get all guessed positions
    guessed = [key for key,value in self.guess_dict.items()]
    # First, set probability to 0 to already made guesses
    for size in self.ship_sizes:
      rem_size = size - 1
      # Check for possible placements of a ship of given size
      for row in range(10):
        for col in range(10):
          # Check if ship can be placed to the up from given row/col position
          if row - rem_size >= 0:
            possible_to_place = True
            # Generate all row numbers ship is going to occupy
            possible_rows = [row - i for i in range(rem_size + 1)]
            for possible_row in possible_rows:
              if (possible_row, col) in guessed:
                possible_to_place = False
                break
            if possible_to_place:
              self.heatmap[(row-rem_size):(row+1), col] += 1
          # Check if ship can be placed to the left from given row/col position
          if col - rem_size >= 0:
            possible_to_place = True
            # Generate all column numbers ship is going to occupy
            possible_cols = [col - i for i in range(rem_size + 1)]
            for possible_col in possible_cols:
              if (row, possible_col) in guessed:
                possible_to_place = False
                break
            if possible_to_place:
              self.heatmap[row, (col - rem_size):(col+1)] += 1
     # Now increase probability of positions near hits
    hits = [key for key,value in self.guess_dict.items() if value == 1]
    for hit in hits:
      row = hit[0]
      col = hit[1]
      # Check if we are still on the board
      if row - 1 >= 0:
      # If new position not guessed increase prob
        if (row - 1, col) not in guessed:
          self.heatmap[row - 1, col] += 15
      # Same for other neighbors
      if row + 1 < 10:
        if (row + 1, col) not in guessed:
          self.heatmap[row + 1, col] += 15
      if col - 1 >= 0:
        if (row, col - 1) not in guessed:
          self.heatmap[row, col - 1] += 15
      if col + 1 < 10:
        if (row, col + 1) not in guessed:
          self.heatmap[row, col + 1] += 15


  def __call__(self, state):
    """
    Generates action for a given environment state.

    Args:
      state (ndarray): observed configuration of the board (ignored, but
      parameter kept for consistency)
    Returns:
      action (int): action to be taken i.e. board position to uncover
    """
    # Get action with highest heatmap value
    action = np.argmax(self.heatmap)
    # Convert it int (row, col) format
    row, col = self._recalculate_index(action)
    # Check if hit or miss and update guess_dict dictionary
    self.guess_dict[(row, col)] = int(self.board[row, col])
    # Recalculate heatmap
    self._recalculate_heatmap()
    # Return action
    return (row, col)

  def reset(self, board):
    """
    Resets policy to its initial state (useful in simulations)
    """
    self.__init__(board)