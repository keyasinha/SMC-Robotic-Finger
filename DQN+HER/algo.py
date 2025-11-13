import gymnasium as gym
import panda_gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
import os
from datetime import datetime

# check for gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

os.makedirs("results", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("models", exist_ok=True)

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

def plot_training_results(df, save_path="plots", algorithm="DQN"):
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
    axes[0].set_title(f'{algorithm}+HER Training Scores over Episodes')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Win Percentage
    axes[1].plot(df['episode'], df['win_percentage'], label='Win Percentage (100 eps)', 
                 color='green', linewidth=2)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Win Percentage (%)')
    axes[1].set_title(f'{algorithm}+HER Win Percentage over Episodes')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 105])
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"{save_path}/{algorithm.lower()}_her_results_{timestamp}.png", dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}/{algorithm.lower()}_her_results_{timestamp}.png")
    plt.show()

class DiscretizedActionWrapper(gym.Wrapper):
    """
    Wrapper to discretize continuous actions for DQN
    Creates a discrete action space from continuous space
    """
    def __init__(self, env, n_bins=11):
        super().__init__(env)
        self.n_bins = n_bins
        
        # Get original action space bounds
        low = env.action_space.low
        high = env.action_space.high
        self.action_dim = env.action_space.shape[0]
        
        # Create discrete actions by binning each dimension
        self.action_values = []
        for i in range(self.action_dim):
            self.action_values.append(np.linspace(low[i], high[i], n_bins))
        
        # Total number of discrete actions (combinations)
        # Each action dimension gets n_bins options
        self.n_actions = n_bins ** self.action_dim
        
        # If action space is too large, use a more practical approach
        # Sample representative actions instead
        if self.n_actions > 1000:
            print(f"Warning: Full discretization would create {self.n_actions} actions.")
            print(f"Using simplified discretization with {n_bins * self.action_dim} actions instead.")
            self.simplified = True
            self.action_space = gym.spaces.Discrete(n_bins * self.action_dim)
        else:
            self.simplified = False
            self.action_space = gym.spaces.Discrete(self.n_actions)
    
    def _discrete_to_continuous(self, discrete_action):
        """Convert discrete action to continuous action"""
        if self.simplified:
            # Simplified: each action changes one dimension
            dim = discrete_action // self.n_bins
            bin_idx = discrete_action % self.n_bins
            
            continuous_action = np.zeros(self.action_dim)
            continuous_action[dim] = self.action_values[dim][bin_idx]
            return continuous_action
        else:
            # Full discretization
            continuous_action = np.zeros(self.action_dim)
            temp = discrete_action
            for i in range(self.action_dim - 1, -1, -1):
                bin_idx = temp % self.n_bins
                continuous_action[i] = self.action_values[i][bin_idx]
                temp //= self.n_bins
            return continuous_action
    
    def step(self, action):
        continuous_action = self._discrete_to_continuous(action)
        return self.env.step(continuous_action)

def train_dqn_her():
    """
    Train DQN with HER on PandaReach environment
    """
    # Create environment with discretized actions
    env = gym.make("PandaReach-v3")
    env = DiscretizedActionWrapper(env, n_bins=7)  # 7 bins per dimension
    
    print(f"Environment: PandaReach-v3 (Discretized)")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space} (Discrete: {env.action_space.n} actions)")
    
    callback = TrackingCallback(verbose=1)
    
    # DQN hyperparameters
    model = DQN(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,  
            goal_selection_strategy="future",  
            online_sampling=True,
            max_episode_length=50,  
        ),
        learning_rate=1e-3,
        buffer_size=1_000_000,
        learning_starts=1000,
        batch_size=256,
        tau=1.0,  # Hard update for DQN
        gamma=0.95,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.2,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        device=device,
        tensorboard_log="./dqn_her_tensorboard/"
    )
    
    print("\nStarting training...")
    print("=" * 50)
    
    # model
    total_timesteps = 150_000  
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        log_interval=4,
        progress_bar=True
    )
    
    # Save the trained model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/dqn_her_panda_{timestamp}"
    model.save(model_path)
    print(f"\nModel saved to {model_path}")
    
    # training data
    df = callback.get_dataframe()
    
    # Save data to CSV
    csv_path = f"results/dqn_training_data_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Training data saved to {csv_path}")
    
    # final statistics
    print(f"Total Episodes: {len(df)}")
    print(f"Final Average Score (last 100 eps): {df['avg_score'].iloc[-1]:.2f}")
    print(f"Final Win Percentage (last 100 eps): {df['win_percentage'].iloc[-1]:.2f}%")
    print(f"Best Win Percentage: {df['win_percentage'].max():.2f}%")
    print(f"Best Average Score: {df['avg_score'].max():.2f}")
    
    # Plot results
    plot_training_results(df, algorithm="DQN")
    
    env.close()
    
    return model, df

def evaluate_model(model, n_eval_episodes=100):
    """
    Evaluate the trained model
    """
    env = gym.make("PandaReach-v3")
    env = DiscretizedActionWrapper(env, n_bins=7)
    
    episode_rewards = []
    episode_successes = []
    
    print(f"\nEvaluating model for {n_eval_episodes} episodes...")
    
    for episode in range(n_eval_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
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

if __name__ == "__main__":
    # Check if GPU is available
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("WARNING: GPU not available, using CPU")

    # Train the model
    model, training_df = train_dqn_her()
    
    # Evaluate the model
    evaluate_model(model, n_eval_episodes=100)
    
    print("\nTraining and evaluation complete.")