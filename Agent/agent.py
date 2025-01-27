import numpy as np
import random
from collections import deque
from Model.model import _build_model 

class TradingAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 1.0  # Exploration-exploitation tradeoff
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.batch_size = 64

        # Replay memory
        self.memory = deque(maxlen=2000)
        self.model_name = 'model_1'

        # Neural network for Q-value approximation
        self.q_network = _build_model(self)
        self.target_network = _build_model(self)
        self.target_network.set_weights(self.q_network.get_weights())
    
    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.choice(range(self.action_size))  # Explore

        state_tensor = np.expand_dims(state, axis=0)  # Prepare for prediction
        q_values = self.q_network.predict(state_tensor, verbose=1)
        return np.argmax(q_values[0])   # Exploit

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones, dtype=float)
        
        # Current Q-values
        q_values = self.q_network.predict(states, verbose=1)

        # Target Q-values
        next_q_values = self.target_network.predict(next_states, verbose=1)
        targets = q_values.copy()
        self.update_target_network()

        for i in range(self.batch_size):
            target = rewards[i]
            if not dones[i]:
                target += self.discount_factor * np.max(next_q_values[i])
            targets[i, actions[i]] = target

        # Train the Q-network
        self.q_network.fit(states, targets, epochs=1, verbose=1)


        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return self.q_network


    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())
