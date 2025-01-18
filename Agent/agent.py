import numpy as np
import pandas as pd
import random

class TradingAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 1.0  # Exploration-exploitation tradeoff
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.discount_factor = 0.99
        self.q_table = {}

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.choice(range(self.action_size))  # Explore
        state_key = tuple(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        return np.argmax(self.q_table[state_key])  # Exploit

    def learn(self, state, action, reward, next_state, done):
        state_key = tuple(state)
        next_state_key = tuple(next_state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_size)

        q_value = self.q_table[state_key][action]
        next_max = np.max(self.q_table[next_state_key])
        new_q_value = reward + self.discount_factor * next_max * (1 - done)
        self.q_table[state_key][action] = (1 - self.learning_rate) * q_value + self.learning_rate * new_q_value

        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

