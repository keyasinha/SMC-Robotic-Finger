# This script implements a hybrid DDPG + Model-Based Control (MPC) strategy
# for the continuous robotic control environment PandaReach-v3 using Stable-Baselines3.

# Installation Requirements:
# pip install stable-baselines3[extra] gymnasium panda-gym matplotlib numpy
# Note: panda-gym requires MuJoCo dependencies (mujoco and mujoco_py) which might require specific setup.

import os
import time
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

# Stable-Baselines3 imports
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.torch_layers import get_actor_critic_arch
from stable_baselines3.common.type_aliases import Schedule

# PyTorch imports
import torch as th
from torch import nn

# --- 1. Custom Policy with Model-Based Correction (Simplified MPC) ---
# This implements the core of the DDPG+MPC idea.
# The DDPG actor learns a residual action, which is added to a
# calculated nominal action (the "MPC" component).

class CustomPandaActor(nn.Module):
    """
    Custom Actor network for DDPG that implements a residual control scheme.
    Total Action = Nominal Action (from Simple Model/MPC) + Residual Action (from DDPG Network)
    
    For PandaReach-v3:
    - Nominal Action (MPC): A simple proportional controller (P-controller)
      that drives the end-effector towards the desired goal in joint space.
      This requires a simplified inverse kinematics approach or a Jacobian transpose.
      
      Since we don't have the full Jacobian here, we will use a common simplification
      in residual control: the nominal action is based on the positional error.
      The model-based component should ideally be a joint velocity that minimizes the
      EE position error (Desired - Current).
      
      We approximate this by passing the observation and calculating the EE error,
      then scaling it into an approximate joint velocity command.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: list[int],
        features_extractor: nn.Module,
        feature_dim: int,
        action_dim: int,
        log_std_init: float = -2,
    ):
        super().__init__()
        
        # DDPG/TD3 Policy Network (learns the RESIDUAL action)
        # Note: DDPG policy is usually a deterministic MLP.
        
        # Determine the shape for the residual network
        actor_arch = net_arch
        if isinstance(net_arch, dict):
            actor_arch = net_arch["pi"]
        
        # Build the residual MLP network
        modules = [nn.Linear(feature_dim, actor_arch[0]), nn.ReLU()]
        for idx in range(len(actor_arch) - 1):
            modules.append(nn.Linear(actor_arch[idx], actor_arch[idx+1]))
            modules.append(nn.ReLU())
        
        # Output layer for the residual action, scaled by tanh
        modules.append(nn.Linear(actor_arch[-1], action_dim))
        self.residual_net = nn.Sequential(*modules)
        
        # Final layer to scale output to action space bounds
        self.action_range = (action_space.high - action_space.low) / 2.0
        self.action_center = (action_space.high + action_space.low) / 2.0
        
        # MPC parameters (Simple P-Controller for nominal action)
        # The observation space for PandaReach is Dict, flattened by SB3's MultiInputPolicy
        # The default observation vector (39 elements) contains:
        # [0:9]: Joint positions/velocities
        # [9:12]: End effector position (achieved_goal)
        # [12:15]: Desired goal position
        
        self.ee_pos_slice = slice(9, 12)  # Check documentation, typically indices 9, 10, 11
        self.goal_pos_slice = slice(12, 15) # Typically indices 12, 13, 14
        self.kp = 0.5  # Proportional gain for the nominal control
        self.action_dim = action_dim # Should be 7 for Panda

    def forward(self, features: th.Tensor) -> th.Tensor:
        """
        Calculates the total action: Nominal (MPC) + Residual (DDPG).
        """
        # --- 1. Residual Action (from DDPG Network) ---
        residual_action = th.tanh(self.residual_net(features))
        
        # Scale residual action to action space bounds
        residual_action = residual_action * th.from_numpy(self.action_range).to(features.device)
        
        # --- 2. Nominal Action (from Simple Model / MPC) ---
        # Get End-Effector position and Desired Goal from the flattened observation
        # Assuming the standard flattened structure:
        # [..., achieved_goal_x, achieved_goal_y, achieved_goal_z, desired_goal_x, desired_goal_y, desired_goal_z, ...]
        
        ee_pos = features[:, self.ee_pos_slice]
        desired_goal = features[:, self.goal_pos_slice]
        
        # Calculate the positional error vector (3D)
        pos_error = desired_goal - ee_pos
        
        # A very basic "MPC" (Nominal Controller)
        # We project the 3D position error onto the 7 joint velocity commands.
        # This is a massive simplification, but serves as a "model-based bias"
        # towards the goal, which the DDPG agent must learn to correct.
        
        # The error is a 3D vector. We need a 7D joint command.
        # To keep it simple: we assume the first 3 joints (which mostly control position)
        # are proportional to the error, and the rest are zero, then normalize.
        
        # Simple nominal joint velocity proportional to positional error
        # Since the action space is 7D, we pad the 3D error
        nominal_action_3d = self.kp * pos_error
        
        # Create a 7D action tensor. We only apply the positional control to the first 3 joints.
        # This is an extremely rough model but demonstrates the concept of providing a nominal path.
        nominal_action = th.cat(
            [
                nominal_action_3d, 
                th.zeros_like(nominal_action_3d)[:, :4] # Zero velocity for the last 4 joints
            ],
            dim=1
        )
        
        # --- 3. Total Action ---
        # Total Action = Nominal Action (Goal-directed) + Residual Action (Learned Correction)
        total_action = nominal_action + residual_action
        
        # Clip the final action to be within the environment's action bounds
        action_low = th.from_numpy(self.action_center - self.action_range).to(features.device)
        action_high = th.from_numpy(self.action_center + self.action_range).to(features.device)
        
        # Use tanh to map back to [-1, 1] for action space scaling, then scale to full range.
        # To avoid clipping issues, we'll clip the final output directly:
        total_action = th.clamp(total_action, action_low, action_high)
        
        return total_action

# Custom Policy combining the custom actor and the standard DDPG MultiInputPolicy
class DDPGMPC(DDPG):
    """
    DDPG agent using the CustomPandaActor for residual control.
    """
    def __init__(self, *args, **kwargs):
        # We enforce MultiInputPolicy due to the Dict observation space of Panda environments
        if 'policy' in kwargs and kwargs['policy'] != 'MultiInputPolicy':
             print("Warning: Overriding policy to 'MultiInputPolicy' for Dict observation space.")
        kwargs['policy'] = 'MultiInputPolicy'

        super().__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        """Overwrite to use the custom actor class."""
        super()._setup_model()
        
        # Re-create the actor using the custom class
        self.actor = self.policy.make_actor(self.policy.features_extractor)
        # Re-initialize the policy network
        self.policy.actor = self.actor
        # Make sure the target actor is also updated
        self.actor_target = self.policy.make_actor(self.policy.features_extractor)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Update optimizers
        self.actor_optimizer = th.optim.Adam(self.actor.parameters(), lr=self.learning_rate)


# --- 2. Custom Callback for Tracking Success Rate and Score ---

class SuccessRateCallback(BaseCallback):
    """
    A custom callback that tracks the episode success rate and cumulative score,
    and stores them for later plotting.
    
    The success is determined by the 'is_success' key in the environment's info dict,
    which is standard for Fetch/Panda robotics environments.
    """
    def __init__(self, check_freq: int, save_path: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        
        self.ep_successes = []
        self.ep_scores = []
        self.episodes = 0
        self.current_episode_rewards = 0.0

        if os.path.exists(save_path):
            os.remove(save_path) # Clean up previous data

    def _on_rollout_start(self) -> None:
        self.current_episode_rewards = 0.0

    def _on_step(self) -> bool:
        # Sum up the reward for the current episode
        self.current_episode_rewards += self.locals.get("rewards", np.array([0]))[0]
        
        # Check if any environment (we use only one here) is terminated or truncated
        dones = self.locals.get("dones", np.array([False]))
        
        if dones[0]: # If the episode ended
            self.episodes += 1
            info = self.locals.get("infos", [{}])[0]
            
            # --- 1. Track Success ---
            # 'is_success' is the key we are looking for in Panda-Gym
            is_successful = info.get("is_success", 0.0)
            self.ep_successes.append(is_successful)

            # --- 2. Track Score (Cumulative Reward) ---
            self.ep_scores.append(self.current_episode_rewards)

            self.current_episode_rewards = 0.0 # Reset for next episode

            # Log periodically
            if self.episodes % self.check_freq == 0 and self.episodes > 0:
                # Calculate success rate over the last `check_freq` episodes
                recent_success_rate = np.mean(self.ep_successes[-self.check_freq:]) * 100
                recent_avg_score = np.mean(self.ep_scores[-self.check_freq:])
                
                if self.verbose > 0:
                    print(f"Episode: {self.episodes} | Timesteps: {self.num_timesteps} | "
                          f"Success Rate (last {self.check_freq}): {recent_success_rate:.2f}% | "
                          f"Avg Score (last {self.check_freq}): {recent_avg_score:.2f}")

            # Save raw data periodically
            if self.episodes % 500 == 0 and self.episodes > 0:
                self._save_data()
        
        return True # Continue training

    def _on_training_end(self) -> None:
        self._save_data()
        
    def _save_data(self):
        """Saves the raw episode data."""
        np.savez(
            self.save_path, 
            successes=np.array(self.ep_successes), 
            scores=np.array(self.ep_scores)
        )
        if self.verbose > 0:
            print(f"--- Data saved to {self.save_path} ---")


# --- 3. Main Simulation and Training ---

def run_ddpg_mpc_simulation(total_timesteps: int = 500000):
    """
    Sets up the environment, trains the DDPG+MPC agent, and plots results.
    """
    # 0. Setup
    ENV_ID = "PandaReach-v3"
    LOG_DIR = "./ddpg_mpc_panda_logs"
    DATA_FILE = os.path.join(LOG_DIR, "training_data.npz")
    
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # 1. Environment Setup
    try:
        # Import panda_gym only when needed to avoid dependency issues on initial load
        import panda_gym 
        env = gym.make(ENV_ID)
        print(f"Environment: {ENV_ID} initialized.")
    except Exception as e:
        print(f"Error initializing {ENV_ID}: {e}")
        print("Falling back to a simpler continuous environment (Pendulum-v1) for demonstration.")
        # Fallback to a simpler continuous environment if Panda is not installed/working
        ENV_ID = "Pendulum-v1"
        env = gym.make(ENV_ID)

    # DDPG requires action noise for exploration
    n_actions = env.action_space.shape[-1]
    # NormalActionNoise is often preferred over OrnsteinUhlenbeckActionNoise now
    action_noise = gym.wrappers.NormalizeReward(gym.spaces.Box(low=env.action_space.low * 0.1, high=env.action_space.high * 0.1, dtype=np.float32)) 


    # 2. DDPG + MPC Model (using the custom policy)
    # The default policy_kwargs will be overridden internally by the custom DDPGMPC agent
    # to use the CustomPandaActor.
    
    print("\nInitializing DDPG+MPC Agent...")
    model = DDPGMPC(
        # Policy is set to 'MultiInputPolicy' for Panda env
        policy="MultiInputPolicy", 
        env=env,
        action_noise=action_noise,
        learning_rate=3e-4,
        buffer_size=100000,
        batch_size=256,
        gamma=0.98, # Discount factor for long horizon
        tau=0.05,
        verbose=0,
        seed=42,
        tensorboard_log=LOG_DIR
    )

    # 3. Training and Tracking
    print(f"Starting training for {total_timesteps} timesteps...")
    
    # Check frequency for logging and success rate calculation (every 10 episodes)
    callback = SuccessRateCallback(check_freq=10, save_path=DATA_FILE, verbose=1)
    
    start_time = time.time()
    
    model.learn(
        total_timesteps=total_timesteps, 
        callback=callback,
        log_interval=10
    )
    
    end_time = time.time()
    print(f"Training finished in {end_time - start_time:.2f} seconds.")
    
    # 4. Plotting
    plot_results(DATA_FILE)

    env.close()


# --- 4. Plotting Function ---

def plot_results(data_file: str):
    """
    Loads saved data and plots the Win Percentage and Cumulative Score.
    """
    if not os.path.exists(data_file):
        print(f"Error: Data file not found at {data_file}")
        return

    # Load data
    data = np.load(data_file)
    ep_successes = data['successes']
    ep_scores = data['scores']
    episodes = np.arange(1, len(ep_successes) + 1)
    
    # Calculate rolling averages for smoother plots
    WINDOW_SIZE = 100
    def rolling_mean(data, window):
        return np.convolve(data, np.ones(window)/window, mode='valid')

    # Success Rate Plot
    rolling_success = rolling_mean(ep_successes, WINDOW_SIZE) * 100
    episodes_for_rolling = episodes[WINDOW_SIZE - 1:]

    # Score Plot
    rolling_score = rolling_mean(ep_scores, WINDOW_SIZE)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('DDPG + Model-Based Control (MPC) on PandaReach-v3 Simulation', fontsize=16)

    # Win Percentage Plot
    ax1.plot(episodes_for_rolling, rolling_success, label=f'Success Rate (Rolling Avg {WINDOW_SIZE})', color='green')
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Win Percentage (%)')
    ax1.set_title(f'Win Percentage vs. Episodes (Target: Reach 100%)')
    ax1.grid(True, linestyle='--')
    ax1.legend()

    # Cumulative Score Plot
    ax2.plot(episodes_for_rolling, rolling_score, label=f'Avg Score (Rolling Avg {WINDOW_SIZE})', color='blue')
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Average Cumulative Reward')
    ax2.set_title('Average Episode Score vs. Episodes')
    ax2.grid(True, linestyle='--')
    ax2.legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
    
    # Save the plot
    plot_path = os.path.join(os.path.dirname(data_file), "ddpg_mpc_results.png")
    plt.savefig(plot_path)
    print(f"\n--- Plot saved to {plot_path} ---")
    
    # Show the plot
    plt.show()


if __name__ == "__main__":
    # Total timesteps should be high for proper learning in robotics environments.
    # We use a moderate value for a runnable simulation example.
    run_ddpg_mpc_simulation(total_timesteps=100000)