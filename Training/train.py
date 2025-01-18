# Step 3: Training Loop
from Env.env import StockMarketEnv  # Import StockMarketEnv class
from Agent.agent import TradingAgent  # Import TradingAgent class

def train_agent(data, episodes=100):
    env = StockMarketEnv(data)
    agent = TradingAgent(state_size=1, action_size=3)

    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        print(f"Episode {e+1}/{episodes} - Total Reward: {total_reward:.2f}")

# Example Usage
if __name__ == "__main__":
    # Load your data (ensure it has a 'Close' column)
    df = pd.read_csv("your_stock_data.csv")
    train_agent(df)
