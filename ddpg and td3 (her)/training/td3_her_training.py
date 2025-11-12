import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import gymnasium as gym
import panda_gym
from agents.td3 import TD3Agent
from utils.HER import her_augmentation
import matplotlib.pyplot as plt

if __name__ == "__main__":

    n_games = 1500
    opt_steps = 64
    best_score = -np.inf
    score_history = []
    avg_score_history = []
    success_history = []  # Track success rate
    win_percentage_history = []  # Track win percentage
    
    env = gym.make('PandaReach-v3')
    obs_shape = env.observation_space['observation'].shape[0] + \
                env.observation_space['achieved_goal'].shape[0] + \
                env.observation_space['desired_goal'].shape[0]

    agent = TD3Agent(env=env, input_dims=obs_shape)

    for i in range(n_games):
        done = False
        truncated = False
        score = 0
        step = 0
        success = 0

        obs_array = []
        actions_array = []
        new_obs_array = []

        observation, info = env.reset()

        while not (done or truncated):
            curr_obs, curr_achgoal, curr_desgoal = observation.values()
            state = np.concatenate((curr_obs, curr_achgoal, curr_desgoal))

            # Choose an action
            action = agent.choose_action(state)

            # Excute the choosen action in the environement
            new_observation, reward, done, truncated, _ = env.step(np.array(action))
            next_obs, next_achgoal, next_desgoal = new_observation.values()
            new_state = np.concatenate((next_obs, next_achgoal, next_desgoal))

            # Store experience in the replay buffer
            agent.remember(state, action, reward, new_state, done)
        
            obs_array.append(observation)
            actions_array.append(action)
            new_obs_array.append(new_observation)

            observation = new_observation
            score += reward
            step += 1
        
        if score >= -2:  # Successfully reached goal (allows tiny margin)
            success = 1
        # Augmente replay buffer with HER
        her_augmentation(agent, obs_array, actions_array, new_obs_array)

        # train the agent in multiple optimization steps
        for _ in range(opt_steps):
          agent.learn()
            
        score_history.append(score)
        success_history.append(success)
        avg_score = np.mean(score_history[-100:])
        avg_score_history.append(avg_score)

        win_percentage = np.mean(success_history[-100:]) * 100
        win_percentage_history.append(win_percentage)

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()
            print(f"*** New best score: {best_score:.1f} - Models saved ***")
        
        print(f"Episode {i} steps {step} score {score:.1f} avg score {avg_score:.1f} win% {win_percentage:.1f}%")
        

        if (i + 1) % 500 == 0:
            agent.save_models()
            print(f"--- Checkpoint saved at episode {i + 1} ---")
            
            # Plot progress every 500 episodes
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Plot 1: Average Score
            ax1.plot(avg_score_history, label='Avg Score (last 100 episodes)', color='blue')
            ax1.axhline(y=0, color='r', linestyle='--', label='Success threshold')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Average Score')
            ax1.set_title('TD3 Training Progress - Average Score')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Win Percentage
            ax2.plot(win_percentage_history, label='Win % (last 100 episodes)', color='green')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Win Percentage (%)')
            ax2.set_title('TD3 Training Progress - Success Rate')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'td3_training_progress_episode_{i+1}.png', dpi=150)
            plt.close()
            print(f"Progress plot saved as td3_training_progress_episode_{i+1}.png")

    # Final save
    agent.save_models()
    print("\n=== TD3 Training Complete ===")
    print(f"Final avg score: {avg_score:.1f}")
    print(f"Final win percentage: {win_percentage:.1f}%")
    print(f"Best score achieved: {best_score:.1f}")
    
    # Final plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Average Score
    ax1.plot(avg_score_history, label='Avg Score (last 100 episodes)', color='blue')
    ax1.axhline(y=0, color='r', linestyle='--', label='Success threshold')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Score')
    ax1.set_title('TD3 Final Training Progress - Average Score')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Win Percentage
    ax2.plot(win_percentage_history, label='Win % (last 100 episodes)', color='green')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Win Percentage (%)')
    ax2.set_title('TD3 Final Training Progress - Success Rate')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('td3_final_training_progress.png', dpi=150)
    print("Final plot saved as td3_final_training_progress.png")
    plt.show()
    
    # Save training data
    np.save('td3_score_history.npy', score_history)
    np.save('td3_avg_score_history.npy', avg_score_history)
    np.save('td3_success_history.npy', success_history)
    np.save('td3_win_percentage_history.npy', win_percentage_history)
    print("Training data saved as td3_*.npy files")