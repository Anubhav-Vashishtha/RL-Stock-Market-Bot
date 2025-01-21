import numpy as np
import pandas as pd

class StockMarketEnv:
    def __init__(self, data, window_size=5):
        self.data = data  # Stock price data
        self.window_size = window_size  # Number of past timesteps for the state
        self.current_step = 0
        self.max_steps = len(data) - 1
        self.action_space = [0, 1, 2]  # Actions: 0 = hold, 1 = buy, 2 = sell
        self.state_size = window_size  # Number of features in the state

    def reset(self):
        """Resets the environment to the initial state."""
        self.current_step = np.random.randint(0, len(self.data) - self.window_size - 1)
        return self.get_state()

    def get_state(self):
        """Returns the current state for the agent."""

        start = max(0, self.current_step - self.window_size + 1)
        end = self.current_step + 1
        
        prices = self.data['Close'][start:end].values
        
        return prices  

    def step(self, action):
        """Takes an action and updates the environment."""

        current_price = self.data['Close'][self.current_step]
        next_price = self.data['Close'][self.current_step + 1] if self.current_step + 1 < len(self.data) else current_price
        reward = 0

        # Calculate reward based on action
        if action == 1:  # Buy
            reward = next_price - current_price
        elif action == 2:  # Sell
            reward = current_price - next_price
        else:  # Hold
            reward = 0

        # Update step
        self.current_step += 1
        done = self.current_step >= self.max_steps

        # Get next state
        next_state = self.get_state()
        return next_state, reward, done
