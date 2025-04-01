'''
Test the FixedBattleshipEnv class.
'''
import numpy as np
from battleshit.environments import FixedBattleshipEnv

def test_fixedbattleshipenv():
    env = FixedBattleshipEnv()
    state, _ = env.reset()
    board_generated1 = env._board_generated.copy()

    done = False
    # Play 2 times to see if the same board is generated
    while True:
        action = np.random.randint(0, 100)
        observation, reward, done, _, info = env.step(action)
        if done:
            _, _ = env.reset()
            board_generated2 = env._board_generated.copy()
            assert board_generated1.all() == board_generated2.all()
            break

