
"""Definition of Battleship Environment"""

import numpy as np

class CustomBattleshipEnv():
    """Custom Battleship Environment that follows gym interface"""
    def __init__(self, good_guess = 5.0, bad_guess = -1.0, repeated_guess = - 100,
                 game_completed = 100):
        '''
        Implements constructor for CustomBattleship Environment.

        Args:
            good_guess (float): reward for guessing right ship position
            bad_guess (float): penalty for not guessing the right ship position
            repeated_guess (float): penalty for making repeated move
            game_completed (float): reward for completing the game
        '''
        # Attributes assignment
        self._good_guess = good_guess
        self._bad_guess = bad_guess
        self._repeated_guess = repeated_guess
        self._game_completed = game_completed
        self._ship_sizes = [5, 4, 3, 3, 2]
        # Define generated board attribute (board to make guesses on)
        self._board_generated = None
        # Define board attribute
        self._board = None
        # Define rewards dictionary for taking specific actions
        self.rewards = {
            'good_guess': self._good_guess,
            'bad_guess': self._bad_guess,
            'repetaed_guess': self._repeated_guess,
            'game_completed':self._game_completed
                        }

    def step(self, action):
        '''
        Implements taking action in Battleship environment. Each board position takes one of
        three possible values:
        0: uncovered square on the board
        1: correct guess
        -1: guess not correct

        Args:
            action (int): number between 0 and 99 signifying ship position
        '''
        # Check if this was the rigth guess and update board appropriately
        if int(self._board_generated[action]) == 1 and int(self._board[action]) == 0:
            reward = self.rewards['good_guess']
            # Update board
            self._board[action] = 1.0
        # Wrong guess
        elif int(self._board_generated[action]) == 0 and int(self._board[action]) == 0:
            reward = self.rewards['bad_guess']
            # Update board
            self._board[action] = -1.0
        # Repeated guess
        else:
            reward = self.rewards['repetaed_guess']
        # Check if we are done (all ships were sunk) i.e. board and generated board sums are equal
        done = int(self._board[self._board == 1].sum()) == int(np.sum(self._board_generated))
        # If game is completed add reward for game completion
        if done:
          reward += self.rewards['game_completed']
        # Set info to be empty dictionary
        info = {}
        # Observation = current board situation
        observation = self._board.copy()
        # Return values follow Gymnasium convention
        return observation, reward, done, False, info

    def reset(self):
        '''
        Sets up new board filled with ships and empty board to make guesses about
        ship positions
        '''
        # Get new generated board filled with ships
        self._board_generated = self._setup_board()
        # Create empty board (filled with 0) to make guesses about ship positions
        self._board = np.zeros(shape = (100,), dtype = np.float32)
        # Return values follow Gymnasium convention
        return self._board, {}

    def _setup_board(self):
        '''
        Helper function for randomly setting up ships on the board.
        '''
        # Create empty board
        self._board_generated = np.zeros((10, 10), dtype = np.float32)
        for size in self._ship_sizes:
          # Try to place ship on the board
          while True:
            # Select random row
            row = np.random.randint(0, 10)
            # Select random column
            col = np.random.randint(0, 10)
            # Select ship orientation (vertical or horizontal)
            orientation = np.random.choice(['H', 'V'])
            # Check if ship of given size and orientation can be placed on the board
            if orientation == 'H':
              # Ship must fit in the board and must not cross another ship
              if (col + size <= 9) and int(self._board_generated[row, col:col+size].sum()) == 0:
                self._board_generated[row, col:col+size] = 1.0
                break
              elif (col - size >= 0) and int(self._board_generated[row, col-size:col].sum()) == 0:
                self._board_generated[row, col-size:col] = 1.0
                break
              else:
                continue
            # Vertical orientation case
            else:
              # Ship must fit in the board and must not cross another ship
              if (row + size <= 9) and int(self._board_generated[row : row+size, col].sum()) == 0:
                self._board_generated[row:row+size, col] = 1.0
                break
              elif (row-size >= 0) and int(self._board_generated[row-size:row, col].sum()) == 0:
                self._board_generated[row-size:row, col] = 1.0
                break
              else:
                continue
        # Return flattened generated board
        return self._board_generated.flatten()

"""Definition of helper class, for experimentation purposes (always generates the same board)"""

class FixedBattleshipEnv(CustomBattleshipEnv):
  def __init__(self):
    # Call parent contructor
    super().__init__(self)
    # Get one fixed generated board
    aux_env = CustomBattleshipEnv()
    _, _ = aux_env.reset()
    self._board_generated = aux_env._board_generated
    self.rewards = aux_env.rewards

  def reset(self):
    '''
      Sets up empty board to make guesses about ship positions
    '''
    # Create empty board (filled with 0) to make guesses about ship positions
    self._board = np.zeros(shape = (100,), dtype = np.float32)
    # Return values follow Gymnasium convention
    return self._board, {}

if __name__ == '__main__':
    
    """Testing Custom Battleship Environment"""

    # env = CustomBattleshipEnv()
    # state, _ = env.reset()
    # print(env._board_generated)
    # print(env._board_generated.reshape(10,10))

    # done = False

    # while True:
    #     action = np.random.randint(0, 100)
    #     observation, reward, done, _, info = env.step(action)
    #     print(f"Action: {action}, reward = {reward}")
    #     print(observation)
    #     if done:
    #         obs, _ = env.reset()
    #         # print(obs)
    #         action = np.random.randint(0, 100)
    #         observation, reward, done, _, info = env.step(action)
    #         # print(observation)
    #         action = np.random.randint(0, 100)
    #         observation, reward, done, _, info = env.step(action)
    #         # print(observation)
    #         break


