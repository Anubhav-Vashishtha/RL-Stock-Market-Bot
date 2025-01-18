import numpy as np
import pandas as pd
import random

class StockMarketEnv:
    def __init__(self, data):
        self.data = data  # Stock price data
        self.current_step = 0  # Current index in the dataset
        self.max_steps = len(data) - 1

    def reset(self):
        self.current_step = 0
        return self.get_state()

    def get_state(self):
        # Return state (e.g., price and moving average)
        return np.array([self.data['Close'][self.current_step]])

    def step(self, action):
        # Actions: 0 = hold, 1 = buy, 2 = sell
        self.current_step += 1
        done = self.current_step >= self.max_steps
        reward = self.calculate_reward(action)
        next_state = self.get_state()
        return next_state, reward, done

    def calculate_reward(self, action):
        # Simplified reward based on price difference
        if action == 1:  # Buy
            return self.data['Close'][self.current_step] - self.data['Close'][self.current_step - 1]
        elif action == 2:  # Sell
            return self.data['Close'][self.current_step - 1] - self.data['Close'][self.current_step]
        else:  # Hold
            return 0