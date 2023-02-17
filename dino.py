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
        pass
    # visualize the game
    def render(self):
        pass
    # restart the game
    def reset(self):
        pass
    # close the observation
    def close(self):
        pass
    # get a part of the game
    def get_observation(self):
        pass
    # get the game over text
    def get_done(self):
        pass

