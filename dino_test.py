from stable_baselines3 import DQN
from dino_train import WebGame
import time

env = WebGame()

model = DQN.load('train/best_model_5000.zip')
for run in range(5):
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(int(action))
        total_reward += reward
    print('total reward for run {} is {}'.format(run, total_reward))
