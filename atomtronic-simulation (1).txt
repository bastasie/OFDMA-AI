import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.fft import fft, ifft
import time

class AtomtronicQPSK:
    """
    Simulation of Atomtronic QPSK processor that implements trigonometric computing
    """
    def __init__(self, dimensions=10, channels=100):
        """
        Initialize the atomtronic processor
        
        Parameters:
        -----------
        dimensions : int
            Number of dimensions in the quantum state
        channels : int
            Number of OFDMA channels to simulate
        """
        self.dimensions = dimensions
        self.channels = channels
        
        # Initialize BEC state in a ring trap
        self.psi_0 = self._initialize_bec_state()
        
        # Initialize phase values for QPSK modulation
        self.phases = np.random.uniform(0, 2*np.pi, (channels, dimensions))
        
        # Initialize quantum gate parameters
        self.gate_phases = np.random.uniform(0, 2*np.pi, channels)
        
    def _initialize_bec_state(self):
        """Initialize the Bose-Einstein condensate ground state"""
        # Simple normalized wavefunction
        psi_0 = np.ones(self.dimensions) / np.sqrt(self.dimensions)
        return psi_0
    
    def phase_modulation(self, input_data, channel_idx=0):
        """
        Apply phase modulation based on input data
        
        Parameters:
        -----------
        input_data : ndarray
            Input data to modulate
        channel_idx : int
            OFDMA channel to use
            
        Returns:
        -----------
        psi_mod : ndarray
            Modulated quantum state
        """
        # Scale input to [0, 1]
        if np.max(input_data) > 1 or np.min(input_data) < 0:
            scaled_input = (input_data - np.min(input_data)) / (np.max(input_data) - np.min(input_data))
        else:
            scaled_input = input_data
            
        # Apply phase modulation (θ(x) = πq(x))
        theta = np.pi * scaled_input
        
        # Modulate the quantum state
        psi_mod = np.exp(1j * theta) * self.psi_0
        
        # Apply channel-specific phase
        psi_mod = np.exp(1j * self.phases[channel_idx]) * psi_mod
        
        return psi_mod
    
    def bas_transform(self, psi):
        """
        Apply BAZ transformation using FFT as a simulation of interferometry
        
        Parameters:
        -----------
        psi : ndarray
            Quantum state to transform
            
        Returns:
        -----------
        psi_transformed : ndarray
            Transformed quantum state in Fourier space
        """
        # Use FFT to simulate the transform
        return fft(psi)
    
    def quantum_gate_operation(self, psi, channel_idx=0):
        """
        Apply quantum gate operation (phase shift)
        
        Parameters:
        -----------
        psi : ndarray
            Quantum state
        channel_idx : int
            Channel index for phase selection
            
        Returns:
        -----------
        psi_gate : ndarray
            State after gate operation
        """
        # Apply phase shift (simulate Josephson junction)
        phi = self.gate_phases[channel_idx]
        
        # Simplified 2x2 matrix operation across the state
        sigma_z = np.array([[1, 0], [0, -1]])
        
        # Reshape state for matrix operation
        if len(psi) % 2 != 0:
            # Pad if odd length
            psi = np.append(psi, 0)
            
        psi_reshaped = psi.reshape(-1, 2)
        psi_gate = np.zeros_like(psi_reshaped, dtype=complex)
        
        # Apply phase gate to each 2D subspace
        for i in range(psi_reshaped.shape[0]):
            psi_gate[i] = np.exp(-1j * phi * sigma_z) @ psi_reshaped[i]
            
        return psi_gate.flatten()[:len(psi)]
    
    def mach_zehnder_readout(self, psi):
        """
        Simulate Mach-Zehnder interferometer readout
        
        Parameters:
        -----------
        psi : ndarray
            Quantum state to read
            
        Returns:
        -----------
        output : float
            Readout value in [0,1]
        """
        # Calculate probability distribution (|Ψ|²)
        prob = np.abs(psi)**2
        
        # For a simple model, convert this to a single output
        # In a real Mach-Zehnder interferometer, the phase differences 
        # would create constructive/destructive interference
        
        # Normalize probabilities
        prob = prob / np.sum(prob)
        
        # Apply trigonometric transformation (emulating interference pattern)
        output = (1 - np.cos(np.pi * np.sum(np.arange(len(prob)) * prob))) / 2
        
        return output
    
    def process(self, input_data, n_channels=1):
        """
        Process input data through the atomtronic processor
        
        Parameters:
        -----------
        input_data : ndarray
            Input data vector
        n_channels : int
            Number of parallel OFDMA channels to use
            
        Returns:
        -----------
        outputs : ndarray
            Processed outputs
        """
        outputs = np.zeros(n_channels)
        
        # Process through multiple channels in parallel
        for i in range(n_channels):
            # Phase modulation
            psi_mod = self.phase_modulation(input_data, i % self.channels)
            
            # Apply BAZ transform
            psi_transformed = self.bas_transform(psi_mod)
            
            # Apply quantum gates
            psi_gate = self.quantum_gate_operation(psi_transformed, i % self.channels)
            
            # Readout
            outputs[i] = self.mach_zehnder_readout(psi_gate)
            
        return outputs
    
    def multi_channel_logic(self, input_data, function_type='AND'):
        """
        Implement logical operations using trigonometric computing
        
        Parameters:
        -----------
        input_data : ndarray
            Array containing multiple inputs (should be 0 or 1 values)
        function_type : str
            Type of logical function ('AND', 'OR', 'XOR', 'NOT')
            
        Returns:
        -----------
        result : float
            Logical operation result
        """
        # Process each input through a different channel
        channel_outputs = []
        for i in range(len(input_data)):
            # Create a one-hot encoding of position
            one_hot = np.zeros(self.dimensions)
            one_hot[i % self.dimensions] = input_data[i]
            
            # Process through a channel
            out = self.process(one_hot, 1)[0]
            channel_outputs.append(out)
            
        # Apply logical function using trigonometric forms
        if function_type == 'AND' and len(channel_outputs) >= 2:
            # Implement AND: (1 - cos(πx) - cos(πy) + cos(π(x+y)))/4
            result = (1 - np.cos(np.pi * channel_outputs[0]) - np.cos(np.pi * channel_outputs[1]) + 
                      np.cos(np.pi * (channel_outputs[0] + channel_outputs[1]))) / 4
            
        elif function_type == 'OR' and len(channel_outputs) >= 2:
            # Implement OR: (3 - cos(πx) - cos(πy) - cos(π(x+y)))/4
            result = (3 - np.cos(np.pi * channel_outputs[0]) - np.cos(np.pi * channel_outputs[1]) - 
                      np.cos(np.pi * (channel_outputs[0] + channel_outputs[1]))) / 4
            
        elif function_type == 'XOR' and len(channel_outputs) >= 2:
            # Implement XOR: (1 - cos(π(x+y)))/2
            result = (1 - np.cos(np.pi * (channel_outputs[0] + channel_outputs[1]))) / 2
            
        elif function_type == 'NOT' and len(channel_outputs) >= 1:
            # Implement NOT: (1 + cos(πx))/2
            result = (1 + np.cos(np.pi * channel_outputs[0])) / 2
            
        else:
            # Default: Pass through the first channel result
            result = channel_outputs[0] if channel_outputs else 0.0
            
        return result

    def learn(self, X, y, epochs=10, learning_rate=0.01):
        """
        Implement learning in the atomtronic processor by updating phases
        
        Parameters:
        -----------
        X : ndarray
            Training data (samples x features)
        y : ndarray
            Target values (0 or 1)
        epochs : int
            Number of training epochs
        learning_rate : float
            Learning rate for phase updates
            
        Returns:
        -----------
        losses : list
            Training loss history
        """
        losses = []
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            for i in range(n_samples):
                # Forward pass
                output = self.process(X[i], 1)[0]
                
                # Calculate error
                error = y[i] - output
                epoch_loss += error**2
                
                # Update phases (simplified backpropagation for simulation)
                # In real quantum hardware, this would be implemented as Hamiltonian evolution
                delta = error * np.pi * np.sin(np.pi * output) / 2
                
                # Update QPSK modulation phases
                self.phases[0] -= learning_rate * delta * X[i][:self.dimensions]
                
                # Update gate phases
                self.gate_phases[0] -= learning_rate * delta
            
            # Record average loss for this epoch
            losses.append(epoch_loss / n_samples)
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch}, Loss: {losses[-1]:.4f}")
                
        return losses
    
    def quantum_parallelism_simulation(self, n_operations=1000):
        """
        Simulate quantum parallelism by performing multiple operations in parallel
        
        Parameters:
        -----------
        n_operations : int
            Number of parallel operations to simulate
            
        Returns:
        -----------
        time_per_op : float
            Time per operation
        """
        # Prepare random inputs
        inputs = np.random.rand(n_operations, self.dimensions)
        
        # Start timing
        start_time = time.time()
        
        # In a real quantum system, all these would execute simultaneously
        # We'll count the time for a single operation since quantum parallelism
        # would perform all operations in the same time as one
        _ = self.process(inputs[0], 1)
        
        # End timing
        elapsed_time = time.time() - start_time
        
        # In a real quantum system with N_eff parallel channels, the time per operation would be:
        time_per_op = elapsed_time / n_operations
        
        return time_per_op
        
    def estimate_theoretical_speedup(self, problem_sizes=[10, 100, 1000, 10000]):
        """
        Estimate theoretical speedup for different problem sizes
        
        Parameters:
        -----------
        problem_sizes : list
            List of problem sizes to estimate
            
        Returns:
        -----------
        classical_times : list
            Estimated times for classical computation
        quantum_times : list
            Estimated times for quantum computation
        speedups : list
            Estimated speedup factors
        """
        # Constants for timing estimation
        # These are simplified for simulation purposes
        classical_op_time = 1e-9  # 1 nanosecond per classical operation
        quantum_op_time = 1e-12   # 1 picosecond per quantum operation
        quantum_parallelism = 1e12  # Simulating 10^12 parallel operations
        
        classical_times = []
        quantum_times = []
        speedups = []
        
        for size in problem_sizes:
            # Classical time: O(n^2) for matrix operations
            classical_time = size * size * classical_op_time
            classical_times.append(classical_time)
            
            # Quantum time: O(1) with parallelism
            quantum_time = quantum_op_time / quantum_parallelism
            quantum_times.append(quantum_time)
            
            # Speedup
            speedup = classical_time / quantum_time
            speedups.append(speedup)
            
        return classical_times, quantum_times, speedups


# Simulation of an Atomtronic Reinforcement Learning System
class AtomtronicRL:
    """Simulation of an Atomtronic Reinforcement Learning System using trigonometric Q-values"""
    
    def __init__(self, n_states, n_actions, channels=100):
        """
        Initialize the atomtronic RL system
        
        Parameters:
        -----------
        n_states : int
            Number of states in the environment
        n_actions : int
            Number of actions available
        channels : int
            Number of quantum channels to simulate
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.channels = channels
        
        # Initialize phase values for Q-value representation
        # In quantum hardware, these would be actual physical phases
        self.phases = np.random.uniform(0, 2*np.pi, (n_states, n_actions))
        
        # Create atomtronic processor for quantum operations
        self.processor = AtomtronicQPSK(dimensions=max(n_states, n_actions), channels=channels)
        
    def get_q_value(self, state, action):
        """
        Get Q-value for a state-action pair using trigonometric encoding
        
        Parameters:
        -----------
        state : int
            State index
        action : int
            Action index
            
        Returns:
        -----------
        q_value : float
            Q-value in [0,1] range
        """
        # Q-value is represented trigonometrically as (1-cos(θ))/2
        # where θ is the phase associated with the state-action pair
        phase = self.phases[state, action]
        q_value = (1 - np.cos(phase)) / 2
        return q_value
    
    def update_q_value(self, state, action, target, learning_rate=0.1):
        """
        Update Q-value through phase adjustment
        
        Parameters:
        -----------
        state : int
            State index
        action : int
            Action index
        target : float
            Target Q-value
        learning_rate : float
            Learning rate for phase update
        """
        # Current Q-value
        current_q = self.get_q_value(state, action)
        
        # Error
        error = target - current_q
        
        # Convert error to phase adjustment (using trigonometric derivative)
        # For q = (1-cos(θ))/2, dq/dθ = sin(θ)/2
        current_phase = self.phases[state, action]
        phase_gradient = np.sin(current_phase) / 2
        
        # Avoid division by zero
        if abs(phase_gradient) < 1e-10:
            phase_gradient = 1e-10
            
        # Update phase
        phase_adjustment = learning_rate * error / phase_gradient
        self.phases[state, action] += phase_adjustment
        
        # Keep phase in [0, 2π] range
        self.phases[state, action] = self.phases[state, action] % (2 * np.pi)
    
    def select_action(self, state, epsilon=0.1):
        """
        Select action using epsilon-greedy policy
        
        Parameters:
        -----------
        state : int
            Current state
        epsilon : float
            Exploration probability
            
        Returns:
        -----------
        action : int
            Selected action
        """
        # Explore: select random action
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.n_actions)
        
        # Exploit: select best action based on Q-values
        q_values = [self.get_q_value(state, a) for a in range(self.n_actions)]
        return np.argmax(q_values)
    
    def train(self, env, episodes=100, max_steps=100, gamma=0.95, learning_rate=0.1, epsilon=0.1):
        """
        Train the RL agent
        
        Parameters:
        -----------
        env : object
            Environment with step(action) and reset() methods
        episodes : int
            Number of training episodes
        max_steps : int
            Maximum steps per episode
        gamma : float
            Discount factor
        learning_rate : float
            Learning rate for updates
        epsilon : float
            Exploration probability
            
        Returns:
        -----------
        rewards : list
            Total reward per episode
        """
        rewards = []
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            
            for step in range(max_steps):
                # Select action
                action = self.select_action(state, epsilon)
                
                # Take action and observe next state and reward
                next_state, reward, done = env.step(action)
                total_reward += reward
                
                # Calculate target Q-value
                if done:
                    target = reward
                else:
                    # Max Q-value for next state
                    next_q_values = [self.get_q_value(next_state, a) for a in range(self.n_actions)]
                    target = reward + gamma * max(next_q_values)
                
                # Update Q-value
                self.update_q_value(state, action, target, learning_rate)
                
                # Update state
                state = next_state
                
                if done:
                    break
            
            rewards.append(total_reward)
            
            if episode % 10 == 0:
                print(f"Episode {episode}, Total Reward: {total_reward}")
                
        return rewards


# Simple grid world environment for testing RL
class GridWorldEnv:
    """Simple grid world environment for RL testing"""
    
    def __init__(self, size=5):
        """
        Initialize grid world
        
        Parameters:
        -----------
        size : int
            Grid size (size x size)
        """
        self.size = size
        self.state = 0  # Start at top-left corner
        self.goal = size * size - 1  # Bottom-right corner
        
        # Actions: 0=up, 1=right, 2=down, 3=left
        self.actions = [0, 1, 2, 3]
        
    def reset(self):
        """Reset environment"""
        self.state = 0
        return self.state
    
    def step(self, action):
        """
        Take a step in the environment
        
        Parameters:
        -----------
        action : int
            Action to take (0=up, 1=right, 2=down, 3=left)
            
        Returns:
        -----------
        next_state : int
            Next state
        reward : float
            Reward
        done : bool
            Whether episode is done
        """
        # Convert state to grid position
        row = self.state // self.size
        col = self.state % self.size
        
        # Apply action
        if action == 0:  # Up
            row = max(0, row - 1)
        elif action == 1:  # Right
            col = min(self.size - 1, col + 1)
        elif action == 2:  # Down
            row = min(self.size - 1, row + 1)
        elif action == 3:  # Left
            col = max(0, col - 1)
            
        # Convert back to state
        self.state = row * self.size + col
        
        # Check if done
        done = (self.state == self.goal)
        
        # Reward
        if done:
            reward = 1.0  # Goal reached
        else:
            reward = -0.01  # Step penalty
            
        return self.state, reward, done


# Demonstration and analysis functions
def run_quantum_simulation():
    """Run simulation of quantum atomtronic processing"""
    print("Running Atomtronic QPSK Processing Simulation...")
    
    # Initialize processor
    processor = AtomtronicQPSK(dimensions=10, channels=100)
    
    # Test logical operations
    print("\nTesting logical operations:")
    test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    for x, y in test_inputs:
        input_data = np.array([x, y])
        
        # Test AND
        and_result = processor.multi_channel_logic(input_data, 'AND')
        
        # Test OR
        or_result = processor.multi_channel_logic(input_data, 'OR')
        
        # Test XOR
        xor_result = processor.multi_channel_logic(input_data, 'XOR')
        
        print(f"{x} AND {y} = {and_result:.4f} (expected: {x and y})")
        print(f"{x} OR {y} = {or_result:.4f} (expected: {x or y})")
        print(f"{x} XOR {y} = {xor_result:.4f} (expected: {x ^ y})")
        print()
    
    # Generate simple binary classification data
    n_samples = 100
    n_features = 10
    
    X = np.random.rand(n_samples, n_features)
    y = np.zeros(n_samples)
    
    # Simple rule: if sum of first 3 features > 1.5, then class=1
    for i in range(n_samples):
        if np.sum(X[i, :3]) > 1.5:
            y[i] = 1
    
    # Split data
    train_idx = np.random.choice(n_samples, int(0.8 * n_samples), replace=False)
    test_idx = np.array([i for i in range(n_samples) if i not in train_idx])
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    # Train atomtronic processor
    print("\nTraining atomtronic processor...")
    losses = processor.learn(X_train, y_train, epochs=50, learning_rate=0.02)
    
    # Evaluate
    print("\nEvaluating...")
    correct = 0
    for i in range(len(X_test)):
        output = processor.process(X_test[i], 1)[0]
        predicted = 1 if output > 0.5 else 0
        if predicted == y_test[i]:
            correct += 1
    
    accuracy = correct / len(X_test)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Atomtronic Processor Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('atomtronic_learning_curve.png')
    plt.show()
    
    # Estimate quantum parallelism speedup
    print("\nEstimating quantum parallelism speedup...")
    time_per_op = processor.quantum_parallelism_simulation(n_operations=10000)
    print(f"Estimated time per operation with quantum parallelism: {time_per_op:.10f} seconds")
    print(f"Equivalent to {1/time_per_op:.2e} operations per second")
    
    # Theoretical speedup comparison
    problem_sizes = [10, 100, 1000, 10000, 100000]
    classical_times, quantum_times, speedups = processor.estimate_theoretical_speedup(problem_sizes)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(1, 2, 1)
    plt.loglog(problem_sizes, classical_times, '-o', label='Classical')
    plt.loglog(problem_sizes, quantum_times, '-o', label='Quantum')
    plt.xlabel('Problem Size')
    plt.ylabel('Time (seconds)')
    plt.title('Computation Time Scaling')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.loglog(problem_sizes, speedups, '-o')
    plt.xlabel('Problem Size')
    plt.ylabel('Speedup Factor')
    plt.title('Quantum Speedup')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('quantum_speedup.png')
    plt.show()
    
    return processor


def run_rl_simulation():
    """Run simulation of atomtronic reinforcement learning"""
    print("\nRunning Atomtronic Reinforcement Learning Simulation...")
    
    # Create environment
    env = GridWorldEnv(size=4)
    
    # Create atomtronic RL agent
    agent = AtomtronicRL(n_states=16, n_actions=4)
    
    # Train
    print("\nTraining RL agent...")
    rewards = agent.train(env, episodes=1000, max_steps=100)
    
    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.title('Atomtronic RL Learning Curve')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.savefig('atomtronic_rl_learning.png')
    plt.show()
    
    # Test policy
    print("\nTesting learned policy...")
    state = env.reset()
    steps = 0
    done = False
    
    states_visited = [state]
    
    while not done and steps < 20:
        action = agent.select_action(state, epsilon=0)
        next_state, reward, done = env.step(action)
        state = next_state
        states_visited.append(state)
        steps += 1
    
    # Convert states to grid positions
    grid_positions = []
    for s in states_visited:
        row = s // env.size
        col = s % env.size
        grid_positions.append((row, col))
    
    # Plot trajectory
    plt.figure(figsize=(8, 8))
    
    # Plot grid
    for i in range(env.size + 1):
        plt.axhline(y=i, color='k')
        plt.axvline(x=i, color='k')
    
    # Plot trajectory
    xs = [pos[1] + 0.5 for pos in grid_positions]
    ys = [pos[0] + 0.5 for pos in grid_positions]
    plt.plot(xs, ys, 'b-o', markersize=10)
    
    # Mark start and goal
    plt.plot(xs[0], ys[0], 'go', markersize=15, label='Start')
    plt.plot(env.goal % env.size + 0.5, env.goal // env.size + 0.5, 'ro', markersize=15, label='Goal')
    
    plt.xlim(0, env.size)
    plt.ylim(env.size, 0)  # Reverse y-axis to match grid coordinates
    plt.xticks(np.arange(0.5, env.size + 0.5), np.arange(env.size))
    plt.yticks(np.arange(0.5, env.size + 0.5), np.arange(env.size))
    plt.title('Learned Policy Trajectory')
    plt.legend()
    plt.grid(False)
    plt.savefig('atomtronic_rl_trajectory.png')
    plt.show()
    
    print(f"Path found in {steps} steps: {' -> '.join(map(str, states_visited))}")
    
    return agent


# Main demonstration
if __name__ == "__main__":
    # Run atomtronic QPSK processor simulation
    processor = run_quantum_simulation()
    
    # Run atomtronic RL simulation
    agent = run_rl_simulation()
    
    print("\nSimulation complete. Results saved to PNG files.")
