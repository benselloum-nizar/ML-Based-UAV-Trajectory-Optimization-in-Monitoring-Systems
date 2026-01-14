import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from uavenv.uav2dgrid import UAVEnv, Frame, OBSTACLE, SN, UAV, CELL, State, Action
from rl import Q_Learning, SARSA, RandomAction

def get_next_position(col, row, action):
    '''Get the next position after taking an action.'''
    if action == Action.UP and row > 0:
        return col, row - 1
    elif action == Action.DOWN and row < Frame.ROWS - 1:
        return col, row + 1
    elif action == Action.LEFT and col > 0:
        return col - 1, row
    elif action == Action.RIGHT and col < Frame.COLS - 1:
        return col + 1, row
    return col, row  # No movement if invalid

def get_valid_actions(col, row, step=0):
    '''Get valid actions from a position using State class.'''
    state = State(col, row, step)
    return state.valid_actions()

def reconstruct_trajectory_from_qtable(q_table, max_steps=50):
    '''
    Reconstruct the optimal trajectory by following the greedy policy from Q-table.
    Returns the trajectory as a list of (col, row) tuples.
    '''
    trajectory = []
    col, row = UAV.START_POS.col, UAV.START_POS.row
    step = 0
    
    trajectory.append((col, row))
    
    while step < max_steps:
        # Create state string
        state_str = f"({col},{row},{step})"
        
        # Get valid actions
        valid_actions = get_valid_actions(col, row, step)
        
        if not valid_actions:
            break
        
        # Get Q-values for this state
        if state_str in q_table:
            q_values = np.array(q_table[state_str])
        else:
            # If state not in Q-table, break
            break
        
        # Choose action with highest Q-value (greedy policy)
        # Only consider valid actions
        valid_q_values = [q_values[a] for a in valid_actions]
        max_q = max(valid_q_values)
        
        # Get all actions with max Q-value
        best_actions = [a for a in valid_actions if q_values[a] == max_q]
        
        # If multiple best actions, prefer actions that don't go back
        if len(best_actions) > 1 and len(trajectory) > 1:
            prev_col, prev_row = trajectory[-2]
            # Avoid going back to previous position
            best_actions = [a for a in best_actions 
                          if get_next_position(col, row, a) != (prev_col, prev_row)]
            if not best_actions:
                best_actions = [a for a in valid_actions if q_values[a] == max_q]
        
        action = best_actions[0] if best_actions else valid_actions[0]
        
        # Move to next position
        next_col, next_row = get_next_position(col, row, action)
        col, row = next_col, next_row
        step += 1
        
        trajectory.append((col, row))
        
        # Check if reached end position
        if col == UAV.END_POS.col and row == UAV.END_POS.row:
            break
    
    return trajectory

def find_qtable_files(project_root):
    '''
    Search for Q-table files in results directory and other locations.
    Returns a dict mapping algorithm names to file paths.
    '''
    import glob
    qtable_files = {}
    
    # Search patterns for each algorithm
    search_patterns = {
        'Q-Learning': ['**/Q-Learning-load.json', '**/Q_Learning-load.json', 
                      '**/Q-Learning*.json', '**/Q_Learning*.json'],
        'SARSA': ['**/SARSA-load.json', '**/SARSA*.json']
    }
    
    print("Searching for Q-table files...")
    
    for algo_name, patterns in search_patterns.items():
        for pattern in patterns:
            matches = glob.glob(os.path.join(project_root, pattern), recursive=True)
            # Filter out results files and __pycache__
            matches = [m for m in matches 
                      if 'results.json' not in m 
                      and '__pycache__' not in m
                      and os.path.isfile(m)]
            
            if matches:
                # Prefer -load.json files
                load_files = [m for m in matches if 'load.json' in m]
                if load_files:
                    qtable_files[algo_name] = load_files[0]
                    print(f"  Found {algo_name}: {os.path.relpath(load_files[0], project_root)}")
                    break
                elif algo_name not in qtable_files:
                    qtable_files[algo_name] = matches[0]
                    print(f"  Found {algo_name}: {os.path.relpath(matches[0], project_root)}")
    
    return qtable_files

def run_episode_with_trajectory(env, ai, num_episodes=100):
    '''
    Run multiple episodes and collect trajectories.
    Returns the best trajectory (highest reward) and its reward.
    '''
    best_trajectory = None
    best_reward = float('-inf')
    best_episode_info = None
    
    for episode in range(num_episodes):
        trajectory = []
        episode_reward = 0
        state, info = env.reset()
        reward = None
        
        # Reset agent state for new episode (if applicable)
        if hasattr(ai, 'current_state'):
            ai.current_state = None
            ai.current_action = None
        
        # Track starting position
        trajectory.append((state.col, state.row))
        
        while True:
            action = ai.get_action(state, reward)
            state, reward, terminated, truncated, info = env.step(action)
            # Track position
            trajectory.append((state.col, state.row))
            episode_reward += reward
            
            if terminated or truncated:
                # Final action update
                ai.get_action(state, reward)
                break
        
        # Update best if this episode is better
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_trajectory = trajectory
            best_episode_info = {
                'reward': episode_reward,
                'terminated': terminated,
                'flight_time': state.step
            }
    
    return best_trajectory, best_reward, best_episode_info

def visualize_trajectory(trajectory, algorithm_name, reward, flight_time, terminated, ax):
    '''
    Visualize a trajectory on the grid with arrows showing direction.
    '''
    # Draw grid
    for row in range(Frame.ROWS):
        for col in range(Frame.COLS):
            # Check if obstacle
            is_obstacle = any(pos.col == col and pos.row == row for pos in OBSTACLE.POS_LIST)
            # Check if start/end position
            is_start = (col == UAV.START_POS.col and row == UAV.START_POS.row)
            
            if is_obstacle:
                color = OBSTACLE.COLOR
            elif is_start:
                color = CELL.START_COLOR
            else:
                color = CELL.BACKGROUND
            
            rect = plt.Rectangle((col, row), 1, 1, 
                               facecolor=[c/255.0 for c in color], 
                               edgecolor=[c/255.0 for c in CELL.GRID_COLOR],
                               linewidth=0.5)
            ax.add_patch(rect)
    
    # Mark end position with X
    end_col, end_row = UAV.END_POS.col, UAV.END_POS.row
    ax.plot([end_col + 0.2, end_col + 0.8], [end_row + 0.2, end_row + 0.8], 
            'k-', linewidth=3)
    ax.plot([end_col + 0.8, end_col + 0.2], [end_row + 0.2, end_row + 0.8], 
            'k-', linewidth=3)
    
    # Draw SNs
    for sn_id, sn_pos in SN.POS.items():
        ax.plot(sn_pos.col, sn_pos.row, 'o', color=[c/255.0 for c in SN.COLOR], 
                markersize=15, markeredgecolor='black', markeredgewidth=1)
        ax.text(sn_pos.col, sn_pos.row - 0.3, f'SN{sn_id}', 
                ha='center', va='top', fontsize=8, fontweight='bold')
    
    # Draw trajectory with arrows
    if trajectory and len(trajectory) > 1:
        # Draw arrows between consecutive points
        for i in range(len(trajectory) - 1):
            start_col, start_row = trajectory[i]
            end_col, end_row = trajectory[i + 1]
            
            # Calculate center positions
            start_x = start_col + 0.5
            start_y = start_row + 0.5
            end_x = end_col + 0.5
            end_y = end_row + 0.5
            
            # Create arrow
            arrow = FancyArrowPatch(
                (start_x, start_y), (end_x, end_y),
                arrowstyle='->', mutation_scale=20,
                color='red', linewidth=2, alpha=0.7,
                zorder=10
            )
            ax.add_patch(arrow)
        
        # Mark start position
        start_col, start_row = trajectory[0]
        ax.plot(start_col + 0.5, start_row + 0.5, 's', 
                color='green', markersize=12, markeredgecolor='black', 
                markeredgewidth=1, zorder=11, label='Start')
        
        # Mark end position of trajectory
        end_col, end_row = trajectory[-1]
        ax.plot(end_col + 0.5, end_row + 0.5, 's', 
                color='red', markersize=12, markeredgecolor='black', 
                markeredgewidth=1, zorder=11, label='End')
    
    # Set axis properties
    ax.set_xlim(-0.5, Frame.COLS - 0.5)
    ax.set_ylim(-0.5, Frame.ROWS - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()  # Invert y-axis to match grid layout (row 0 at top)
    ax.set_xlabel('Column', fontsize=10)
    ax.set_ylabel('Row', fontsize=10)
    
    # Title with episode info
    status = "✓ Returned on time" if terminated else "✗ Failed to return"
    title = f'{algorithm_name}\nReward: {reward:.2f} | Steps: {flight_time} | {status}'
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(range(Frame.COLS))
    ax.set_yticks(range(Frame.ROWS))

def main():
    '''
    Main function to load Q-tables from results, reconstruct trajectories, and visualize them.
    '''
    print("="*70)
    print("Trajectory Visualization - Optimal Paths from Q-Tables")
    print("="*70)
    
    # Get project directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = script_dir
    results_dir = os.path.join(project_root, "results")
    
    # Initialize environment once to set up valid_action_matrix
    print("\nInitializing environment...")
    env = UAVEnv()
    env.close()  # Close immediately, we just need the valid_action_matrix
    
    # Find Q-table files
    qtable_files = find_qtable_files(project_root)
    
    if not qtable_files:
        print("\nError: No Q-table files found!")
        print("Expected files:")
        print("  - Q-Learning-load.json or Q_Learning-load.json")
        print("  - SARSA-load.json")
        return
    
    # Algorithm configurations (only Q-Learning and SARSA, no Random Action)
    algorithms = ['Q-Learning', 'SARSA']
    
    # Collect trajectories for each algorithm
    trajectories_data = []
    
    for algo_name in algorithms:
        if algo_name not in qtable_files:
            print(f"\nSkipping {algo_name}: Q-table file not found")
            continue
            
        print(f"\nProcessing {algo_name}...")
        load_file = qtable_files[algo_name]
        print(f"  - Loading Q-table from {os.path.relpath(load_file, project_root)}")
        
        try:
            with open(load_file, 'r') as f:
                q_table = json.load(f)
            
            # Remove metadata entries
            q_table_clean = {k: np.array(v) if isinstance(v, list) else v 
                           for k, v in q_table.items() 
                           if k not in ['round', 'epsilon']}
            
            print(f"  - Reconstructing optimal trajectory from Q-table...")
            trajectory = reconstruct_trajectory_from_qtable(q_table_clean)
            
            if trajectory:
                # Get episode info from results JSON if available
                results_file = None
                possible_results = [
                    os.path.join(results_dir, f"{algo_name.replace(' ', '_')}_results.json"),
                    os.path.join(results_dir, f"{algo_name.replace('-', '_')}_results.json"),
                    os.path.join(results_dir, "plots", f"{algo_name.replace(' ', '_')}_results.json"),
                    os.path.join(results_dir, "plots", f"{algo_name.replace('-', '_')}_results.json"),
                ]
                
                for rf in possible_results:
                    if os.path.exists(rf):
                        results_file = rf
                        break
                
                flight_time = len(trajectory) - 1
                terminated = (trajectory[-1][0] == UAV.END_POS.col and 
                            trajectory[-1][1] == UAV.END_POS.row)
                reward = 0.0
                
                if results_file:
                    try:
                        with open(results_file, 'r') as f:
                            results_data = json.load(f)
                            # Get best reward from results
                            if 'rewards' in results_data and results_data['rewards']:
                                reward = max(results_data['rewards'])
                            if 'flight_times' in results_data and results_data['flight_times']:
                                flight_time = results_data['flight_times'][0] if results_data['flight_times'] else flight_time
                            if 'terminated' in results_data and results_data['terminated']:
                                terminated = any(results_data['terminated'])
                    except:
                        pass
                
                # Print trajectory
                print(f"  - Trajectory ({len(trajectory)} steps):")
                for step, (col, row) in enumerate(trajectory):
                    print(f"    Step {step}: ({col}, {row})")
                
                trajectories_data.append({
                    'name': algo_name,
                    'trajectory': trajectory,
                    'reward': reward,
                    'flight_time': flight_time,
                    'terminated': terminated
                })
                print(f"  - Trajectory reconstructed: {len(trajectory)} steps, "
                      f"Reward = {reward:.2f}, Returned = {terminated}")
            else:
                print(f"  - Warning: Could not reconstruct trajectory")
                
        except Exception as e:
            print(f"  - Error loading {algo_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create visualization
    if not trajectories_data:
        print("\nError: No trajectories to visualize!")
        return
    
    print(f"\n{'='*70}")
    print("Creating visualization...")
    print(f"{'='*70}")
    
    # Create figure with subplots
    num_algorithms = len(trajectories_data)
    fig, axes = plt.subplots(1, num_algorithms, figsize=(6*num_algorithms, 6))
    
    if num_algorithms == 1:
        axes = [axes]
    
    for idx, data in enumerate(trajectories_data):
        visualize_trajectory(
            data['trajectory'],
            data['name'],
            data['reward'],
            data['flight_time'],
            data['terminated'],
            axes[idx]
        )
        plt.tight_layout()
    
    # Save figure
    output_dir = os.path.join(script_dir, "results", "plots")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "best_trajectories.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_file}")
    
    # Also show the plot
    plt.show()
    
    print(f"\n{'='*70}")
    print("Visualization complete!")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
