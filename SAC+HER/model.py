import gymnasium as gym
import panda_gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
import torch
import os
from datetime import datetime

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create directories for saving results
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

def plot_training_results(df, save_path="plots"):
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
    axes[0].set_title('Training Scores over Episodes')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Win Percentage
    axes[1].plot(df['episode'], df['win_percentage'], label='Win Percentage (100 eps)', 
                 color='green', linewidth=2)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Win Percentage (%)')
    axes[1].set_title('Win Percentage over Episodes')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 105])
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"{save_path}/training_results_{timestamp}.png", dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}/training_results_{timestamp}.png")
    plt.show()

def train_sac_her():
    """
    Train SAC with HER on PandaReach environment
    """
    # Create environment
    env = gym.make("PandaReach-v3")
    
    print(f"Environment: PandaReach-v3")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Create callback for tracking
    callback = TrackingCallback(verbose=1)
    
    # SAC hyperparameters optimized for robotic tasks
    model = SAC(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,  # Number of virtual transitions per real transition
            goal_selection_strategy="future",  # Use future strategy for HER
            online_sampling=True,
            max_episode_length=50,  # PandaReach episode length
        ),
        learning_rate=1e-3,
        buffer_size=1_000_000,
        learning_starts=1000,
        batch_size=256,
        tau=0.05,
        gamma=0.95,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs=dict(net_arch=[256, 256, 256]),
        verbose=1,
        device=device,
        tensorboard_log="./sac_her_tensorboard/"
    )
    
    print("\nStarting training...")
    print("=" * 50)
    
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
    model_path = f"models/sac_her_panda_{timestamp}"
    model.save(model_path)
    print(f"\nModel saved to {model_path}")
    
    # Get training data
    df = callback.get_dataframe()
    
    # Save data to CSV
    csv_path = f"results/training_data_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Training data saved to {csv_path}")
    
    # Print final statistics
    print("\n" + "=" * 50)
    print("TRAINING COMPLETED")
    print("=" * 50)
    print(f"Total Episodes: {len(df)}")
    print(f"Final Average Score (last 100 eps): {df['avg_score'].iloc[-1]:.2f}")
    print(f"Final Win Percentage (last 100 eps): {df['win_percentage'].iloc[-1]:.2f}%")
    print(f"Best Win Percentage: {df['win_percentage'].max():.2f}%")
    print(f"Best Average Score: {df['avg_score'].max():.2f}")
    
    # Plot results
    plot_training_results(df)
    
    env.close()
    
    return model, df

def evaluate_model(model, n_eval_episodes=100):
    """
    Evaluate the trained model
    """
    env = gym.make("PandaReach-v3")
    
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
    
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Success Rate: {success_rate:.2f}%")
    print("=" * 50)
    
    return mean_reward, success_rate

if __name__ == "__main__":
    # Check if GPU is available
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("WARNING: GPU not available, using CPU")
    
    # Train the model
    model, training_df = train_sac_her()
    
    # Evaluate the model
    evaluate_model(model, n_eval_episodes=100)
    
    print("\nAll results saved in 'results/', 'plots/', and 'models/' directories")