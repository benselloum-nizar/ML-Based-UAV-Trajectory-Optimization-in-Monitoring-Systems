'''
This program uses the UAV environment with a chosen policy
to test the performance of the policy. The aim of the policy
is to find a trajectory that maximizes the reward during the
flight time of the UAV.

        +---------------+  state,reward   +-----------+
        |               |---------------->|           |
        |      UAV      |                 |    ML     |
        |  Environment  |     action      |   Agent   |
        |               |<----------------|           |
        +---------------+                 +-----------+ 

To code run an episode can be as simple as:

```python
    env = UAVEnv()                    # instantiate a UAV environment
    state, info = env.reset()         # reset the environment
    reward, episode_reward = None, 0  # initialize rewards
    while True:
        action = ai.get_action(state,reward)
        state, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        if terminated or truncated: break
    print(f"The reward for this episode is {episode_reward}")
```
'''

from uavenv.uav2dgrid import UAVEnv
from rl import RandomAction, Q_Learning, SARSA
import json
import os
import numpy as np
from time import strftime, localtime

###########################################
## switches
###########################################
SHOW_ANIMATION_FLAG = False   # to show animation?
LOAD_DATA_FLAG      = False   # to load the model: f"{ai.name}-load"
SAVE_DATA_FLAG      = False   # to save the model: f"{ai.name}-[{date}][{time}]"
EXPLORATION_FLAG    = False   # to perform exploration?
SAMPLE_FLAG         = False   # to sample the model and show result on terminal
SAMPLE_INTERVAL     = 10000   # specify how many episodes before a sample is taken
FPS                 = 10      # initial frame-per-second setting

###########################################
## Run mode, pick one
###########################################

## stateless mode with animation
#SHOW_ANIMATION_FLAG  = True  # show animation
#LOAD_DATA_FLAG       = False # don't load 
#SAVE_DATA_FLAG       = True # don't save
#EXPLORATION_FLAG     = True  # do exploration
#SAMPLE_FLAG          = False # don't show results on the terminal

## training mode from existing model if any
#SHOW_ANIMATION_FLAG  = False # no animation
#LOAD_DATA_FLAG       = True  # load existing model to continue training
#SAVE_DATA_FLAG       = True  # save
#EXPLORATION_FLAG     = True  # do exploration
#SAMPLE_FLAG          = True  # sample results & show on the terminal
#SAMPLE_INTERVAL      = 10000 # show results once every 10000

## testing mode
SHOW_ANIMATION_FLAG  = False  # show animation
FPS                  = 120     # with slow motion
LOAD_DATA_FLAG       = True   # load existing model for testing
SAVE_DATA_FLAG       = True    # save
EXPLORATION_FLAG     = False  # don't explore anymore, only exploit
SAMPLE_FLAG          = True   # sample results & show on the terminal
SAMPLE_INTERVAL      = 100000  # show results every time


###########################################
## outcome collector data structure
###########################################
class Episode:
    reward = []         # to keep reward for each episode
    flight_time = []    # to store the flight time of the UAV
    terminated = []     # to indicate whether the UAV has returned on time
    trajectories = []   # to store the trajectory (list of [col, row] positions) for each episode
episode = Episode()

###########################################
## initialize the UAV environment
###########################################

env = UAVEnv()
state, info = env.reset()

print(f"Simulation info:")
for description in info["description"].values():
    print(f"- {description}")

## pick a policy to run
ai = RandomAction()
#ai = Q_Learning(len(env.action_space), exploration=EXPLORATION_FLAG)
#ai = SARSA(len(env.action_space), exploration=EXPLORATION_FLAG)
print(f"Running simulation using {ai.name} algorithm.")

###########################################
## main loop
###########################################

## initialize episode variables
running_flag = True
pause_flag = False
episode_id = 1
episode_reward = 0
reward = None # set to None to skip reward update for the first run
current_trajectory = [[state.col, state.row]]  # to track the current episode trajectory, initialize with starting position

## load existing data to resume progress?
if LOAD_DATA_FLAG:
    print("- Load data requested")
    progress = ai.load_data()
    if progress!=-1:
        episode_id = progress+1
    else:
        print("- Failed to load data")

## enter the loop
try:
    while running_flag:

        ## render on screen & process events (for animation)
        if SHOW_ANIMATION_FLAG:
            env.render(FPS)
            for event in env.get_event():
                if event.type == env.EVENT_QUIT:
                    running_flag = False
                elif event.type == env.EVENT_MOUSEBUTTONDOWN:
                    if event.button==env.MOUSE_LEFT_BUTTON:
                        pause_flag = not pause_flag
                        if pause_flag: env.set_message("Pause...")
                        else: env.set_message("")
                    elif event.button==env.MOUSE_SCROLL_UP:
                        FPS += 1
                        if FPS>120: FPS = 120
                    elif event.button==env.MOUSE_SCROLL_DOWN:
                        FPS -= 1
                        if FPS<1: FPS = 1

        ## in pause state, skip simulation
        if pause_flag: continue

        ## execute an action
        action = ai.get_action(state,reward,done=False)
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        ## track trajectory (add current position)
        current_trajectory.append([state.col, state.row])

        ## accumulate rewards
        episode_reward += reward
    
        ## end of episode?
        if terminated or truncated:

            ## update the AI with the reward (episode is done)
            action = ai.get_action(state,reward,done=True)

            ## store episode outcome
            episode.reward.append(episode_reward)
            episode.terminated.append(terminated)
            episode.flight_time.append(state.step)
            episode.trajectories.append(current_trajectory.copy())  # store trajectory for this episode

            ## show statistics
            if SAMPLE_FLAG:
                if episode_id==1 or episode_id%SAMPLE_INTERVAL==0:
                    ## re-run the model to show results with full exploitation
                    episode_reward = 0
                    original_value = ai.set_exploration(False) # temporary stop exploration
                    state, info = env.reset()
                    while True:
                        action = ai.get_action(state,reward)
                        state, reward, terminated, truncated, info = env.step(action)
                        episode_reward += reward
                        if terminated or truncated:
                            epsilon = f"{float(ai.epsilon):.4f}" if hasattr(ai, "epsilon") else "N/A"
                            print(f"Episode {episode_id}: reward = {episode_reward:.2f}, "
                                f"epsilon = {epsilon}, "
                                f"flight time = {state.step}"
                                f"{' (returned on time)' if terminated else ''}")
                            ai.set_exploration(original_value) # resume original setting
                            break

            ## reset to start a new episode
            state, info = env.reset()
            episode_reward = 0
            current_trajectory = [[state.col, state.row]]  # initialize with starting position
            episode_id += 1


except KeyboardInterrupt:
    print(" [Interrupted] Program stopped.")

## close the program
env.close()

## save existing data for later use?
print(f"Stopping {ai.name} algorithm...")
if SAVE_DATA_FLAG: 
    print("- Save data requested")
    if ai.save_data(episode_id-1):
        print(f"- Progress saved, number of rounds = {episode_id-1}")
    else:
        print("- Failed to save data")

## save training results to results directory
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(results_dir, exist_ok=True)

# Find the best trajectory (episode with highest reward)
best_trajectory = None
best_episode_idx = None
best_reward = None
if episode.reward and len(episode.reward) > 0:
    best_episode_idx = np.argmax(episode.reward)
    best_reward = episode.reward[best_episode_idx]
    if best_episode_idx < len(episode.trajectories):
        best_trajectory = episode.trajectories[best_episode_idx]

# Prepare training data
training_data = {
    "algorithm": ai.name,
    "episodes": list(range(1, len(episode.reward) + 1)),
    "rewards": episode.reward,
    "flight_times": episode.flight_time,
    "terminated": episode.terminated,
    "num_episodes": len(episode.reward),
    "avg_reward": float(np.mean(episode.reward)) if episode.reward else 0.0,
    "std_reward": float(np.std(episode.reward)) if episode.reward else 0.0,
    "min_reward": float(np.min(episode.reward)) if episode.reward else 0.0,
    "max_reward": float(np.max(episode.reward)) if episode.reward else 0.0,
    "success_rate": float(sum(episode.terminated) / len(episode.terminated) * 100) if episode.terminated else 0.0,
    "avg_flight_time": float(np.mean(episode.flight_time)) if episode.flight_time else 0.0,
    "final_episode": episode_id - 1,
    "best_trajectory": {
        "episode": int(best_episode_idx + 1) if best_episode_idx is not None else None,
        "reward": float(best_reward) if best_reward is not None else None,
        "trajectory": best_trajectory if best_trajectory is not None else []
    }
}

# Save to results directory
algorithm_name_safe = ai.name.replace(" ", "_").replace("-", "_")
results_filename = os.path.join(results_dir, f"{algorithm_name_safe}_results.json")
with open(results_filename, 'w') as f:
    json.dump(training_data, f, indent=2)

print(f"- Training results saved to {results_filename}")
