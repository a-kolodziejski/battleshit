import torch
import time
import gymnasium as gym
'''
Perform visualization (animation) of the agent behavior in the environment
'''

def visualize(env, model_path, num_animations):
    '''
    Performs visualization of the agent's performance. Should be used with
    gym types of environments.
        
    Args:
        env (gym.Env): Environment to be used for visualization.
        model_path (str): Path to the model file to be loaded.
        num_animations (int): The number of animations to be performed.
    '''
    # Load model
    model = torch.load(model_path, weights_only=False)
    # Put model in evaluation mode
    model.eval()
    for _ in range(num_animations):
        # Reset environment
        state, _ = env.reset()
        done = False
        # Switch off gradient tracking
        with torch.no_grad():            
            # Loop until episode is done
            while not done:
                # Select action
                action = torch.argmax(model(state), dim = -1).item()
                # Take action in environment and collect experience
                next_state, reward, terminated, truncated, _ = env.step(action)
                # Check if episode is done
                done = terminated or truncated
                # Slow down animation
                time.sleep(0.1)
                # Update state
                state = next_state if not done else env.reset()[0]


# Load environment
env = gym.make("CartPole-v1", render_mode = 'human')
# Run visualization
visualize(env, "models/saved_models/experiment_1.pt", 5)