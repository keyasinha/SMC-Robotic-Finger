import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import gymnasium as gym
import panda_gym
from numpngw import write_apng
from IPython.display import Image
from agents.ddpg import DDPGAgent

env = gym.make("PandaReach-v3", render_mode="rgb_array")
obs_shape = env.observation_space['observation'].shape[0] + \
                env.observation_space['achieved_goal'].shape[0] + \
                env.observation_space['desired_goal'].shape[0]

# Choose your trained agent: DDPG or TD3
agent = DDPGAgent(env=env, input_dims=obs_shape)

# Get initial observation to build networks
observation, info = env.reset()
curr_obs, curr_achgoal, curr_desgoal = observation.values()
state = np.concatenate((curr_obs, curr_achgoal, curr_desgoal))

# Build networks before loading weights
agent.build_networks(state)

# Load pre-trained networks weights
agent.load_models()

# Reset for clean start
observation, info = env.reset()

# Stores frames of robot arm moving in Reacher env
images = [env.render()]
done = False
truncated = False

for i in range(200):
    curr_obs, curr_achgoal, curr_desgoal = observation.values()
    state = np.concatenate((curr_obs, curr_achgoal, curr_desgoal))
    
    # Choose an action using pre-trained RL agent (evaluate=True means no exploration noise)
    action = agent.choose_action(state, evaluate=True)
    
    # Execute the chosen action in the environment
    new_observation, reward, done, truncated, _ = env.step(np.array(action))
    images.append(env.render())
    observation = new_observation
    
    if done or truncated:
        observation, info = env.reset()
        images.append(env.render())

env.close()

import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots(figsize=(8, 6))
ax.axis('off')

# Display first frame
im = ax.imshow(images[0])

def update(frame):
    im.set_array(images[frame])
    return [im]

ani = animation.FuncAnimation(fig, update, frames=len(images), 
                              interval=60, blit=True, repeat=True)

# Save as MP4 (requires ffmpeg) or GIF
ani.save('robot_demo.gif', writer='pillow', fps=16)
plt.close()

print("Animation saved as robot_demo.gif")
print("Opening animation...")

# Display the animation
fig, ax = plt.subplots(figsize=(8, 6))
ax.axis('off')
im = ax.imshow(images[0])

ani = animation.FuncAnimation(fig, update, frames=len(images), 
                              interval=60, blit=True, repeat=True)
plt.show()

# Save frames: real-time rendering = 40 ms between frames
write_apng("anim.png", images, delay=60)
print("Animation saved as anim.png")

# Show movements (this works in Jupyter notebooks)
Image(filename="anim.png")