import os
import csv
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import pickle
import logging

# Step 1: Establish path to SUMO (SUMO_HOME)
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# Step 2: Add TraCI module to provide access to SUMO simulation
import traci

# Step 3: Define SUMO configuration
def get_sumo_config(gui=False):
    """Returns SUMO configuration for GUI or headless mode."""
    base = [
        'sumo-gui' if gui else 'sumo',
        '-c', 'osm.sumocfg',
        '--step-length', '0.10',
        '--lateral-resolution', '0'
    ]
    if gui:
        base.extend(['--delay', '1'])
    return base

# Step 4: Setup logging
logging.basicConfig(filename='dqn_training.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# -------------------------
# Step 5: Define Variables and Hyperparameters
# -------------------------

# ---- Reinforcement Learning Hyperparameters ----
NUM_EPISODES = 10           # Number of simulation episodes
TOTAL_STEPS_PER_EPISODE = 36000  # Steps per episode
TEST_STEPS = 36000           # Steps for test phase after training
ALPHA = 0.0003             # Learning rate
GAMMA = 0.99               # Discount factor
EPSILON_START = 0.5        # Starting exploration rate
EPSILON_END = 0.05         # Final exploration rate
EPSILON_DECAY = 0.9999     # Decay rate for epsilon
ACTIONS = [0, 1]           # 0 = keep current duration, 1 = change to new duration
DURATIONS = list(range(5, 91, 5))  # Duration options for green phases: 5 to 90 seconds in 5-second increments
YELLOW_DURATION = 50       # 5 seconds for yellow phase (50 steps at step-length=0.10)
MAX_WAITING_TIME_THRESHOLD = 600  # 60 seconds (600 steps at step-length=0.10)
WAITING_TIME_PENALTY = -10.0  # Penalty for exceeding waiting time threshold

# ---- Additional Stability Parameters ----
current_phase_duration = 30  # Initial phase duration (in steps)
phase_start_step = 0         # Step when current phase started
TARGET_UPDATE = 1000         # Update target network every 1000 steps
MAX_QUEUE_THRESHOLD = 20     # Threshold for "long congestion"

# ---- File paths for saving/loading ----
MODEL_PATH = "models/dqn_policy_net.pth"
TARGET_MODEL_PATH = "models/dqn_target_net.pth"
REPLAY_BUFFER_PATH = "models/replay_buffer.pkl"
REWARD_BUFFER_PATH = "models/reward_buffer.pkl"
STATE_PATH = "models/training_state.pkl"

# ---- DQN Model Setup ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(14, 256)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, len(ACTIONS))  # Output size matches number of actions (0 or 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

# Initialize networks
policy_net = DQN().to(device)
target_net = DQN().to(device)
optimizer = optim.Adam(policy_net.parameters(), lr=ALPHA)
replay_buffer = deque(maxlen=20000)
reward_buffer = deque(maxlen=1000)
reward_mean = 0.0
reward_std = 1.0
epsilon = EPSILON_START
cumulative_reward = 0.0
total_steps = 0
last_duration = DURATIONS[0]  # Track the last selected duration

def save_simulation_stats_to_csv(stats_dict, filename="plots/simulation_stats.csv"):
    """Saves simulation statistics to a CSV file."""
    file_exists = os.path.isfile(filename)
    
    with open(filename, 'a', newline='') as csvfile:
        fieldnames = list(stats_dict.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        writer.writerow(stats_dict)
    
    logging.info(f"Simulation statistics saved to {filename}")

# -------------------------
# Step 6: Functions for Saving and Loading
# -------------------------

def save_training_state():
    """Saves the training state, including models, optimizer, buffers, and hyperparameters."""
    torch.save(policy_net.state_dict(), MODEL_PATH)
    torch.save(target_net.state_dict(), TARGET_MODEL_PATH)
    torch.save(optimizer.state_dict(), "models/optimizer.pth")
    
    with open(REPLAY_BUFFER_PATH, 'wb') as f:
        pickle.dump(replay_buffer, f)
    with open(REWARD_BUFFER_PATH, 'wb') as f:
        pickle.dump(reward_buffer, f)
    
    state = {
        'epsilon': epsilon,
        'cumulative_reward': cumulative_reward,
        'total_steps': total_steps,
        'reward_mean': reward_mean,
        'reward_std': reward_std,
        'last_duration': last_duration
    }
    with open(STATE_PATH, 'wb') as f:
        pickle.dump(state, f)
    logging.info("Training state saved.")

def load_training_state():
    """Loads the training state if available, otherwise initializes from scratch."""
    global epsilon, cumulative_reward, total_steps, reward_mean, reward_std, replay_buffer, reward_buffer, last_duration
    
    try:
        policy_net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        target_net.load_state_dict(torch.load(TARGET_MODEL_PATH, map_location=device))
        optimizer.load_state_dict(torch.load("models/optimizer.pth", map_location=device))
        
        with open(REPLAY_BUFFER_PATH, 'rb') as f:
            replay_buffer = pickle.load(f)
        with open(REWARD_BUFFER_PATH, 'rb') as f:
            reward_buffer = pickle.load(f)
        
        with open(STATE_PATH, 'rb') as f:
            state = pickle.load(f)
            epsilon = state['epsilon']
            cumulative_reward = state['cumulative_reward']
            total_steps = state['total_steps']
            reward_mean = state['reward_mean']
            reward_std = state['reward_std']
            last_duration = state.get('last_duration', DURATIONS[0])
        
        logging.info("Training state loaded successfully.")
    except FileNotFoundError:
        logging.info("No previous training state found. Starting from scratch.")
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

# -------------------------
# Step 7: Define Functions
# -------------------------

def normalize_state(state):
    """Normalizes queue lengths to [0, 1]."""
    max_queue = 40.0
    normalized_queues = [min(q / max_queue, 1.0) for q in state[:-1]]
    return tuple(normalized_queues + [state[-1]])

def normalize_reward(reward):
    """Normalizes rewards using running mean and std."""
    global reward_mean, reward_std
    reward_buffer.append(reward)
    reward_mean = np.mean(reward_buffer)
    reward_std = np.std(reward_buffer) if np.std(reward_buffer) > 0 else 1.0
    return (reward - reward_mean) / reward_std

def get_state():
    """Collects state: queue lengths from detectors (aggregated per lane) and current traffic light phase."""
    detector_ids = [
        "1.1", "1.2", "1.3",
        "1.4", "1.5", "1.6", "1.7", "2.4", "2.5", "2.6", "2.7", "3.4", "3.5", "3.6", "3.7", "4.4", "4.5", "4.6", "4.7",
        "1.8", "1.9", "2.8", "2.9",
        "1.10", "1.11", "1.12", "1.13", "2.10", "2.11", "2.12", "2.13"
    ]
    
    # Group detectors by lane (based on the Y part of X.Y)
    lanes = {}
    for det in detector_ids:
        try:
            lane = det.split('.')[1]  # Get the lane number (Y)
            if lane not in lanes:
                lanes[lane] = []
            lanes[lane].append(det)
        except IndexError:
            logging.warning(f"Invalid detector ID format: {det}")
            continue
    
    # Aggregate queue lengths per lane (sum the number of vehicles across detectors on the same lane)
    queue_lengths = []
    for lane in sorted(lanes.keys(), key=int):  # Sort by lane number for consistency
        total_vehicles = 0
        for det in lanes[lane]:
            try:
                total_vehicles += traci.lanearea.getLastStepVehicleNumber(det)
            except traci.TraCIException:
                continue
        queue_lengths.append(total_vehicles)
    
    traffic_light_id = "cluster_244495414_244495415"
    try:
        current_phase = traci.trafficlight.getPhase(traffic_light_id)
    except traci.TraCIException:
        current_phase = 0
    
    return tuple(queue_lengths + [current_phase])

def get_max_waiting_time_per_road():
    """Calculates the maximum waiting time for vehicles on each road (based on detectors)."""
    detector_ids = [
        "1.1", "1.2", "1.3",
        "1.4", "1.5", "1.6", "1.7", "2.4", "2.5", "2.6", "2.7", "3.4", "3.5", "3.6", "3.7", "4.4", "4.5", "4.6", "4.7",
        "1.8", "1.9", "2.8", "2.9",
        "1.10", "1.11", "1.12", "1.13", "2.10", "2.11", "2.12", "2.13"
    ]
    
    # Group detectors by lane
    lanes = {}
    for det in detector_ids:
        try:
            lane = det.split('.')[1]  # Get the lane number (Y)
            if lane not in lanes:
                lanes[lane] = []
            lanes[lane].append(det)
        except IndexError:
            logging.warning(f"Invalid detector ID format: {det}")
            continue
    
    # Group lanes by road (based on your network setup)
    roads_detectors = {
        "Almatinka_EB": ["1", "2", "3"],  # Lanes 1, 2, 3 (Node1_2_EB)
        "Gorky_SB": ["4", "5", "6", "7"],  # Lanes 4, 5, 6, 7 (Node2_7_SB)
        "Almatinka_WB": ["8", "9"],  # Lanes 8, 9 (Node2_3_WB)
        "Gorky_NB": ["10", "11", "12", "13"]  # Lanes 10, 11, 12, 13 (Node2_5_NB)
    }
    
    max_waiting_times = {}
    for road, lane_ids in roads_detectors.items():
        max_wait = 0.0
        for lane in lane_ids:
            if lane in lanes:
                for det in lanes[lane]:
                    try:
                        vehicle_ids = traci.lanearea.getLastStepVehicleIDs(det)
                        for veh_id in vehicle_ids:
                            wait_time = traci.vehicle.getWaitingTime(veh_id)
                            max_wait = max(max_wait, wait_time)
                    except traci.TraCIException:
                        continue
        max_waiting_times[road] = max_wait
    return max_waiting_times

def get_reward(state, passed_vehicles, action, prev_action):
    """Enhanced reward function with penalty for long waiting times."""
    total_queue = sum(state[:-1])
    queue_balance = -np.std(state[:-1])
    switch_penalty = -5.0 if action == 1 and prev_action == 1 else 0.0
    reward = -total_queue + 0.5 * passed_vehicles + 0.1 * queue_balance + switch_penalty
    
    # Bonus for no long congestion on all roads
    no_congestion_bonus = 10.0 if all(queue <= MAX_QUEUE_THRESHOLD for queue in state[:-1]) else 0.0
    reward += no_congestion_bonus
    
    # Penalty for long waiting times on any road
    max_waiting_times = get_max_waiting_time_per_road()
    max_wait = max(max_waiting_times.values()) if max_waiting_times else 0.0
    if max_wait * 0.1 > MAX_WAITING_TIME_THRESHOLD / 10:  # Convert steps to seconds
        reward += WAITING_TIME_PENALTY
        logging.info(f"Penalty applied: Max waiting time {max_wait * 0.1:.2f} seconds exceeds threshold {MAX_WAITING_TIME_THRESHOLD / 10} seconds")
    
    return normalize_reward(reward)

def get_passed_vehicles():
    """Placeholder: Add detectors for outgoing lanes in deepLearning.add.xml."""
    return 0

def get_waiting_time():
    """Calculates total waiting time of all vehicles currently in the simulation."""
    total_waiting_time = 0.0
    vehicle_ids = traci.vehicle.getIDList()
    for veh_id in vehicle_ids:
        try:
            total_waiting_time += traci.vehicle.getWaitingTime(veh_id)
        except traci.TraCIException:
            continue
    return total_waiting_time

def apply_action(phase, duration, current_step, tls_id="cluster_244495414_244495415"):
    """Sets the phase and its duration."""
    global current_phase_duration, phase_start_step
    current_phase_duration = duration
    phase_start_step = current_step
    try:
        traci.trafficlight.setPhase(tls_id, phase)
        traci.trafficlight.setPhaseDuration(tls_id, duration / 10.0)  # Convert steps to seconds
    except traci.TraCIException as e:
        logging.warning(f"Error setting phase {phase} with duration {duration}: {e}")

def should_change_phase(current_step):
    """Checks if current phase duration has elapsed."""
    return (current_step - phase_start_step) >= current_phase_duration

def get_action_from_policy(state, epsilon=0.0):
    """Selects action using epsilon-greedy policy with Double DQN."""
    global last_duration
    if random.random() < epsilon:
        action = random.choice(ACTIONS)
    else:
        normalized_state = normalize_state(state)
        state_tensor = torch.tensor(normalized_state, dtype=torch.float32).to(device)
        with torch.no_grad():
            q_values = policy_net(state_tensor)
        action = torch.argmax(q_values).item()  # 0 or 1
    
    if action == 0:
        return last_duration  # Keep the current duration
    else:
        new_duration = random.choice(DURATIONS)  # Choose a new duration
        last_duration = new_duration
        return new_duration

def get_roads_with_green_light(phase):
    """Maps the phase to roads that get green light."""
    phase_to_roads = {
        0: "Gorky Northbound (Green)",
        1: "Gorky Northbound (Yellow)",
        2: "Gorky Southbound (Green)", 
        3: "Gorky Southbound (Yellow)",
        4: "Almatinka Eastbound (Green)",
        5: "Almatinka Eastbound (Yellow)",
        6: "Almatinka Westbound (Green)",
        7: "Almatinka Westbound (Yellow)"
    }
    return phase_to_roads.get(phase, "Unknown Phase")

def update_model(old_state, action, reward, new_state):
    """Updates DQN model using Double DQN and target network."""
    experience = (old_state, action, reward, new_state)
    replay_buffer.append(experience)
    
    if len(replay_buffer) > 128:
        batch = random.sample(replay_buffer, 128)
        states, actions, rewards, next_states = zip(*batch)
        
        states = torch.tensor([normalize_state(s) for s in states], dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.tensor([normalize_state(s) for s in next_states], dtype=torch.float32).to(device)
        
        q_values = policy_net(states)
        next_q_values_policy = policy_net(next_states)
        next_actions = torch.argmax(next_q_values_policy, dim=1)
        next_q_values_target = target_net(next_states).detach()
        targets = rewards + GAMMA * next_q_values_target[range(len(next_actions)), next_actions]
        q_values = q_values[range(len(actions)), actions]
        
        loss = nn.MSELoss()(q_values, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# -------------------------
# Step 8: Command Line Arguments and Main Loop
# -------------------------

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--gui', action='store_true', help='Run with GUI (default is headless)')
    parser.add_argument('--test-only', action='store_true', help='Run only test phase without training')
    return parser.parse_args()

def run_test_only(gui=True):
    """Run only the test phase without training"""
    load_training_state()  # Load trained model
    global epsilon
    epsilon = 0.0  # Disable exploration
    
    print("\n=== Running Test Phase Only ===")
    logging.info("Running test phase only")
    
    # Setup phase logger for test phase
    phase_logger = logging.getLogger('phase_logger_test')
    phase_handler = logging.FileHandler('phase_log_test.txt')
    phase_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    phase_logger.addHandler(phase_handler)
    phase_logger.setLevel(logging.INFO)

    # Start a new SUMO simulation for testing
    try:
        traci.start(get_sumo_config(gui))
        if gui:
            traci.gui.setSchema("View #0", "real world")
        for _ in range(10):
            traci.simulationStep()
    except traci.FatalTraCIError as e:
        sys.exit(f"Error starting SUMO for test phase: {e}")
    except traci.TraCIException as e:
        logging.warning(f"Could not set GUI schema: {e}")

    # Reset variables for test phase
    current_phase_duration = 30
    phase_start_step = 0
    prev_action = 0
    duration_counts = {d: 0 for d in DURATIONS}  # To count how often each duration is chosen
    test_step = 0
    test_queue_history = []  # To track queue lengths during test phase
    test_step_history = []   # To track steps during test phase
    test_waiting_time_history = []  # To track waiting time during test phase
    current_phase = 0  # Start with phase 0
    current_simulation_step = 0  # Initialize for test phase
    apply_action(current_phase, current_phase_duration, current_simulation_step)

    while test_step < TEST_STEPS:
        # Check if simulation has ended
        try:
            if traci.simulation.getMinExpectedNumber() <= 0:
                print(f"Test phase ended early at step {test_step} (no more vehicles).")
                logging.info(f"Test phase ended early at step {test_step} (no more vehicles).")
                break
        except traci.TraCIException as e:
            print(f"TraCI error in test phase at step {test_step}: {e}")
            logging.error(f"TraCI error in test phase at step {test_step}: {e}")
            break

        current_simulation_step = test_step
        state = get_state()
        current_phase = state[-1]  # Last element of state is the current phase
        
        if should_change_phase(current_simulation_step):
            if current_phase % 2 == 0:  # After green phase, go to yellow
                next_phase = current_phase + 1
                duration = YELLOW_DURATION
                action = 0
            else:  # After yellow phase, go to next green
                next_phase = (current_phase + 1) % 8
                action = get_action_from_policy(state, epsilon=0.0)
                duration = action * 10  # Convert seconds to steps
                duration_counts[duration // 10] += 1  # Only count green phase durations
            
            apply_action(next_phase, duration, current_simulation_step)
            prev_action = action
            roads_with_green = get_roads_with_green_light(next_phase)
            phase_logger.info(f"Step {test_step}, Green Light for {roads_with_green}, Duration {duration // 10} seconds")
            print(f"Step {test_step}, Green Light for {roads_with_green}, Duration {duration // 10} seconds")

        try:
            traci.simulationStep()
        except traci.TraCIException as e:
            print(f"Test phase simulation step failed at step {test_step}: {e}")
            logging.error(f"Test phase simulation step failed at step {test_step}: {e}")
            break

        new_state = get_state()
        total_waiting_time = get_waiting_time()
        test_queue_history.append(sum(new_state[:-1]))
        test_step_history.append(test_step)
        test_waiting_time_history.append(total_waiting_time)
        test_step += 1

    # Close SUMO connection after test phase
    try:
        traci.close()
    except traci.TraCIException as e:
        logging.warning(f"Error closing TraCI connection in test phase: {e}")

    # Print statistics on duration usage for test phase
    print("\n=== Duration Usage Statistics for Test Phase ===")
    logging.info("Duration Usage Statistics for Test Phase:")
    for duration, count in duration_counts.items():
        percentage = (count / sum(duration_counts.values())) * 100 if sum(duration_counts.values()) > 0 else 0
        print(f"Duration {duration} seconds: {count} times ({percentage:.2f}%)")
        logging.info(f"Duration {duration} seconds: {count} times ({percentage:.2f}%)")

    # Plot queue length during test phase
    plt.figure(figsize=(10, 6))
    plt.plot(test_step_history, test_queue_history, marker='o', linestyle='-', label="Total Queue Length")
    plt.xlabel("Test Simulation Steps")
    plt.ylabel("Total Queue Length")
    plt.title("Double DQN: Queue Length During Test Phase")
    plt.legend()
    plt.grid(True)
    if os.path.exists("plots/test_queue_length.png"):
        os.remove("plots/test_queue_length.png")
    plt.savefig("plots/test_queue_length.png", bbox_inches='tight', dpi=300)
    plt.close()

    # Plot waiting time during test phase
    plt.figure(figsize=(10, 6))
    plt.plot(test_step_history, test_waiting_time_history, marker='o', linestyle='-', label="Total Waiting Time")
    plt.xlabel("Test Simulation Steps")
    plt.ylabel("Total Waiting Time (seconds)")
    plt.title("Double DQN: Waiting Time During Test Phase")
    plt.legend()
    plt.grid(True)
    if os.path.exists("plots/test_waiting_time.png"):
        os.remove("plots/test_waiting_time.png")
    plt.savefig("plots/test_waiting_time.png", bbox_inches='tight', dpi=300)
    plt.close()

    logging.info("Test phase plots saved: test_queue_length.png, test_waiting_time.png")
    
    # Get simulation statistics
    sim_stats = {
        'Duration': traci.simulation.getTime() / 10.0,  # Convert steps to seconds
        'TraCI-Duration': traci.simulation.getCurrentTime() / 1000.0,  # Convert ms to seconds
        'Real_time_factor': traci.simulation.getTime() / (time.time() - traci.simulation.getStartTime()),
        'UPS': traci.simulation.getUPS(),
        'Vehicles_Inserted': traci.simulation.getDepartedNumber(),
        'Vehicles_Loaded': traci.simulation.getLoadedNumber(),
        'Vehicles_Running': traci.simulation.getMinExpectedNumber(),
        'Vehicles_Waiting': traci.simulation.getStartingTeleportNumber(),
        'Teleports': traci.simulation.getEndingTeleportNumber(),
        'Emergency_Stops': traci.simulation.getEmergencyStoppingNumber(),
        'Emergency_Braking': traci.simulation.getEmergencyBrakingNumber(),
        'Avg_RouteLength': traci.simulation.getRouteLengthMean(),
        'Avg_Speed': traci.simulation.getSpeedMean(),
        'Avg_Duration': traci.simulation.getDurationMean(),
        'Avg_WaitingTime': traci.simulation.getWaitingTimeMean(),
        'Avg_TimeLoss': traci.simulation.getTimeLossMean(),
        'Avg_DepartDelay': traci.simulation.getDepartDelayMean()
    }
    
    # Save statistics to CSV
    save_simulation_stats_to_csv(sim_stats)

from multiprocessing import Pool, cpu_count
import time

def run_simulation(args):
    """Wrapper function for parallel simulation execution"""
    episode, gui = args
    try:
        # Initialize for each process
        traci.start(get_sumo_config(gui))
        if gui:
            traci.gui.setSchema("View #0", "real world")
        
        # Run simulation steps
        state = get_state()
        current_phase = state[-1]
        current_phase_duration = 30
        phase_start_step = 0
        
        for step in range(TOTAL_STEPS_PER_EPISODE):
            if should_change_phase(step):
                next_phase = (current_phase + 1) % 8
                if next_phase % 2 == 1:
                    duration = YELLOW_DURATION
                else:
                    duration = get_action_from_policy(state, 0.0) * 10
                apply_action(next_phase, duration, step)
            
            traci.simulationStep()
            state = get_state()
            current_phase = state[-1]
        
        traci.close()
        return episode, True
    except Exception as e:
        logging.error(f"Error in episode {episode}: {str(e)}")
        return episode, False

def train_model(epochs, gui=False):
    global NUM_EPISODES
    NUM_EPISODES = epochs
    
    # Load previous training state if available
    load_training_state()

    # Create plots directory if not exists
    if not os.path.exists("plots"):
        os.makedirs("plots")

    print(f"\n=== Starting Training with {NUM_EPISODES} epochs ({'GUI' if gui else 'headless'} mode) ===")
    logging.info(f"Starting training loop with {NUM_EPISODES} epochs in {'GUI' if gui else 'headless'} mode")
    
    # Use all available cores
    num_workers = cpu_count()
    print(f"Using {num_workers} parallel workers")

    # Run episodes in parallel
    with Pool(processes=num_workers) as pool:
        results = pool.map(run_simulation, [(episode, gui) for episode in range(NUM_EPISODES)])
    
    # Process results
    successful_episodes = sum(1 for _, success in results if success)
    print(f"Completed {successful_episodes}/{NUM_EPISODES} episodes successfully")

    # Save training state after all episodes
    save_training_state()
    print("\nTraining completed.")
    logging.info("Training completed.")

def main():
    args = parse_args()
    if args.test_only:
        run_test_only(args.gui)
    else:
        train_model(args.epochs, args.gui)

if __name__ == "__main__":
    main()

print("\nDouble DQN Training and Testing completed.")
print(f"Total training steps: {total_steps}")
print(f"Final cumulative reward: {cumulative_reward:.2f}")
print(f"Number of experiences in replay buffer: {len(replay_buffer)}")
logging.info(f"Training and Testing completed. Total steps: {total_steps}, Final cumulative reward: {cumulative_reward:.2f}")
