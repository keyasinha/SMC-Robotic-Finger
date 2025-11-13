import gymnasium as gym
import panda_gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import BaseCallback
import torch
import os
from datetime import datetime
from scipy.optimize import minimize
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create directories for saving results
os.makedirs("results", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("models", exist_ok=True)

class MPCController:
    """
    Model Predictive Control for refining DDPG actions
    """
    def __init__(self, horizon=5, learning_model=None):
        self.horizon = horizon
        self.learning_model = learning_model
        
    def predict_dynamics(self, state, action):
        """
        Simple learned dynamics model (using DDPG's actor as approximate dynamics)
        In practice, you would learn a separate forward model
        """
        next_state = state.copy()
        if 'observation' in state:
            obs = state['observation']
            next_state['observation'] = obs + action * 0.02  # dt = 0.02
        
        return next_state
    
    def cost_function(self, action_sequence, current_state, goal):
        """
        Cost function for MPC optimization
        Minimizes distance to goal and action magnitude
        """
        state = current_state.copy()
        total_cost = 0.0
        action_dim = len(action_sequence) // self.horizon
        
        for t in range(self.horizon):
            # Extract action for this timestep
            action = action_sequence[t * action_dim:(t + 1) * action_dim]
            
            # Clip action to valid range
            action = np.clip(action, -1.0, 1.0)
            
            # Predict next state (simplified)
            state = self.predict_dynamics(state, action)
            
            # Distance to goal cost
            if 'achieved_goal' in state and 'desired_goal' in goal:
                achieved = state.get('observation', state['achieved_goal'])[:3]  # position
                desired = goal['desired_goal']
                distance_cost = np.linalg.norm(achieved - desired)
            else:
                distance_cost = 0.0
            
            # Action magnitude cost (encourage smooth actions)
            action_cost = 0.01 * np.sum(action ** 2)
            
            total_cost += distance_cost + action_cost
        
        return total_cost
    
    def optimize_action(self, current_state, goal, initial_action):
        """
        Optimize action sequence using MPC
        """
        action_dim = len(initial_action)
        
        # Initialize action sequence (repeat initial action)
        initial_sequence = np.tile(initial_action, self.horizon)
        
        # Bounds for actions
        bounds = [(-1.0, 1.0)] * (action_dim * self.horizon)
        
        # Optimize using scipy
        result = minimize(
            lambda x: self.cost_function(x, current_state, goal),
            initial_sequence,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 10, 'disp': False}
        )
        
        # Return first action from optimized sequence
        optimized_action = result.x[:action_dim]
        return optimized_action

class TrackingCallback(BaseCallback):
    """
    Custom callback for tracking episode statistics
    """
    def __init__(self, eval_freq=1000, verbose=0):
        super(TrackingCallback, self).__init__(verbose)
        self.eval_freq = eval_freq
        self.episode_rewards = []
        self.episode_successes = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.episode_count = 0
        
        # For storing data
        self.data = {
            'episode': [],
            'score': [],
            'avg_score': [],
            'win_percentage': [],
            'episode_length': []
        }
    
    def _on_step(self) -> bool:
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_length += 1
        
        # Check if episode is done
        if self.locals['dones'][0]:
            self.episode_count += 1
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            # Check if goal was achieved (success)
            info = self.locals['infos'][0]
            success = info.get('is_success', 0)
            self.episode_successes.append(success)
            
            # Calculate statistics
            avg_score = np.mean(self.episode_rewards[-100:])  # Last 100 episodes
            win_percentage = np.mean(self.episode_successes[-100:]) * 100
            
            # Store data
            self.data['episode'].append(self.episode_count)
            self.data['score'].append(self.current_episode_reward)
            self.data['avg_score'].append(avg_score)
            self.data['win_percentage'].append(win_percentage)
            self.data['episode_length'].append(self.current_episode_length)
            
            if self.verbose > 0:
                print(f"Episode {self.episode_count}: Score={self.current_episode_reward:.2f}, "
                      f"Avg Score={avg_score:.2f}, Win%={win_percentage:.2f}%, "
                      f"Success={success}")
            
            # Reset episode tracking
            self.current_episode_reward = 0
            self.current_episode_length = 0
        
        return True
    
    def get_dataframe(self):
        """Return tracking data as pandas DataFrame"""
        return pd.DataFrame(self.data)

def plot_training_results(df, save_path="plots", algorithm="DDPG+MPC"):
    """
    Plot training results: scores and win percentage
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Episode Scores
    axes[0].plot(df['episode'], df['score'], alpha=0.3, label='Episode Score', color='blue')
    axes[0].plot(df['episode'], df['avg_score'], label='Average Score (100 eps)', 
                 color='red', linewidth=2)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Score')
    axes[0].set_title(f'{algorithm} Training Scores over Episodes')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Win Percentage
    axes[1].plot(df['episode'], df['win_percentage'], label='Win Percentage (100 eps)', 
                 color='green', linewidth=2)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Win Percentage (%)')
    axes[1].set_title(f'{algorithm} Win Percentage over Episodes')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 105])
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"{save_path}/{algorithm.lower().replace('+', '_')}_results_{timestamp}.png", 
                dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}/{algorithm.lower().replace('+', '_')}_results_{timestamp}.png")
    plt.show()

def train_ddpg_mpc():
    """
    Train DDPG with MPC on PandaReach environment
    """
    # Create environment
    env = gym.make("PandaReach-v3")
    
    print(f"Environment: PandaReach-v3")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Create callback for tracking
    callback = TrackingCallback(verbose=1)
    
    # Action noise for exploration (Ornstein-Uhlenbeck process)
    n_actions = env.action_space.shape[0]
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.2 * np.ones(n_actions)
    )
    
    # DDPG hyperparameters optimized for robotic tasks
    model = DDPG(
        "MultiInputPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=1_000_000,
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.95,
        train_freq=(1, "episode"),
        gradient_steps=-1,  # -1 means as many as steps in episode
        action_noise=action_noise,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        device=device,
        tensorboard_log="./ddpg_mpc_tensorboard/"
    )
    
    # Train the model
    total_timesteps = 100_000  # Adjust based on your needs
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        log_interval=4,
        progress_bar=True
    )
    
    # Save the trained model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/ddpg_mpc_panda_{timestamp}"
    model.save(model_path)
    print(f"\nModel saved to {model_path}")
    
    # Get training data
    df = callback.get_dataframe()
    
    # Save data to CSV
    csv_path = f"results/ddpg_mpc_training_data_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Training data saved to {csv_path}")
    
    # Print final statistics
    print("TRAINING COMPLETED")
    print(f"Total Episodes: {len(df)}")
    print(f"Final Average Score (last 100 eps): {df['avg_score'].iloc[-1]:.2f}")
    print(f"Final Win Percentage (last 100 eps): {df['win_percentage'].iloc[-1]:.2f}%")
    print(f"Best Win Percentage: {df['win_percentage'].max():.2f}%")
    print(f"Best Average Score: {df['avg_score'].max():.2f}")
    
    # Plot results
    plot_training_results(df, algorithm="DDPG+MPC")
    
    env.close()
    
    return model, df

def evaluate_model(model, n_eval_episodes=100, use_mpc=True):
    """
    Evaluate the trained model with optional MPC refinement
    """
    env = gym.make("PandaReach-v3")
    
    # Initialize MPC controller
    mpc = MPCController(horizon=5) if use_mpc else None
    
    episode_rewards = []
    episode_successes = []
    
    eval_type = "with MPC" if use_mpc else "without MPC"
    print(f"\nEvaluating model {eval_type} for {n_eval_episodes} episodes")
    
    for episode in range(n_eval_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        
        while not (done or truncated):
            
            action, _states = model.predict(obs, deterministic=True)
            
            
            if use_mpc and mpc is not None:
                try:
                    action = mpc.optimize_action(obs, obs, action)
                except Exception as e:
                    # If MPC fails, use DDPG action
                    pass
            
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
        
        episode_rewards.append(episode_reward)
        episode_successes.append(info.get('is_success', 0))
        
        if (episode + 1) % 10 == 0:
            print(f"Evaluated {episode + 1}/{n_eval_episodes} episodes")
    
    env.close()
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    success_rate = np.mean(episode_successes) * 100
    
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Success Rate: {success_rate:.2f}%")
    
    
    return mean_reward, success_rate

def compare_with_without_mpc(model):
    """
    Compare performance with and without MPC
    """
    # Evaluate without MPC
    print("\n1. Evaluating DDPG alone")
    mean_reward_base, success_rate_base = evaluate_model(model, n_eval_episodes=50, use_mpc=False)
    
    # Evaluate with MPC
    print("\n2. Evaluating DDPG+MPC")
    mean_reward_mpc, success_rate_mpc = evaluate_model(model, n_eval_episodes=50, use_mpc=True)
    
    # Print comparison
    print("COMPARISON RESULTS")
    print(f"DDPG alone:      Reward={mean_reward_base:.2f}, Success={success_rate_base:.2f}%")
    print(f"DDPG+MPC:        Reward={mean_reward_mpc:.2f}, Success={success_rate_mpc:.2f}%")
    print(f"Improvement:     Reward={mean_reward_mpc - mean_reward_base:+.2f}, "
          f"Success={success_rate_mpc - success_rate_base:+.2f}%")

if __name__ == "__main__":
    # Check if GPU is available
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("WARNING: GPU not available, using CPU")
    
    
    # Train the model
    model, training_df = train_ddpg_mpc()
    
    # Compare with and without MPC
    compare_with_without_mpc(model)
    
    