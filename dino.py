# pydirectinput is used for sending commands
# import pydirectinput
import pyautogui
import numpy as np
# OCR for game over extraction
import pytesseract
import time
import cv2

from matplotlib import pyplot as plt
# environment components
from gym.spaces import Box, Discrete
from gym import Env
# MSS is used for screen capture
from mss import mss


# create game environment
class WebGame(Env):
    # set up the environment action and observation shapes
    def __init__(self):
        super().__init__()
        # set up spaces, returns a multidimensional array, shaped in 1 batch with 83x100 pixels and np.unit8 data type
        self.observation_space = Box(low=0, high=255, shape=(1, 83, 100), dtype=np.uint8)
        # Discrete(3) for 3 different actions
        self.action_space = Discrete(3)
        # define extraction parameters
        self.cap = mss()
        self.game_location = {'top':300, 'left':0, 'width':600, 'height': 500}
        self.done_location = {'top':405, 'left':630, 'width':660, 'height': 70}

    # called to do something in the game
    def step(self, action):
        # action key: 0 = jump, 1 = duck, 2 = no action (no op)
        action_map = {
            0: 'space',
            1: 'down',
            2: 'no_op'
        }
        if action != 2:
            pyautogui.press(action_map[action])

        # checking if game is done
        done, done_cap = self.get_done()
        # getting next observation
        new_observ = self.get_observation()
        # reward (getting a point for every frame we stay alive)
        reward = 1
        # info dict
        info = {}
        return new_observ, reward, done, info

    # visualize the game
    def render(self):
        img = np.array(self.cap.grab(self.game_location))
        cv2.imshow('Game', img)
        key = cv2.waitKey(1)
        while key not in [ord('q'), 27]:  # wait for 'q' or 'ESC' key
            key = cv2.waitKey(1)
        self.close()
        """
        cv2.imshow('Game', np.array(self.cap.grab(self.game_location)))
        if cv2.waitKey(0) & 0xFF == ord('q'):
            self.close()
        """

    # restart the game
    def reset(self):
        # wait 1 second
        time.sleep(1)
        # click on corner
        pyautogui.click(x=150, y=150)
        # hit space to rerun the game
        pyautogui.press('space')
        return self.get_observation()

    # close the observation
    def close(self):
        cv2.destroyAllWindows()

    # get a part of the game
    def get_observation(self):
        # get screen capture of the game
        raw = np.array(self.cap.grab(self.game_location))[:,:,:3].astype(np.uint8)
        # grayscale
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        # resize
        resized = cv2.resize(gray, (100,83))
        # add channels first
        channel = np.reshape(resized, (1,83,100))
        return channel

    # get the game over text using OCR
    def get_done(self):
        # get done screen
        done_cap = np.array(self.cap.grab(self.done_location))[:,:,:3].astype(np.uint8)
        # valid done texts
        done_str = ['GAME', 'GoAH']
        # apply OCR
        done = False
        res = pytesseract.image_to_string(done_cap)[:4]
        if res in done_str:
            done = True
        return done, done_cap


# Testing
"""
env = WebGame()

obs = env.get_observation()
done, done_cap = env.get_done()

print(done)

plt.imshow(obs[0])
plt.show()
plt.imshow(done_cap)
plt.show()
"""

# Play 10 games
for episode in range(10):
    obs = env.reset()
    done = False
    total_reward   = 0
    while not done:
        obs, reward,  done, info =  env.step(env.action_space.sample())
        total_reward  += reward
    print('Total Reward for episode {} is {}'.format(episode, total_reward))

