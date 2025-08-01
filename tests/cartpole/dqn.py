import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="rgb_array")
pobs, _ = env.reset()
