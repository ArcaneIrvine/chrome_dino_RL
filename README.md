# DINO_rl
A reinforced learning AI model for chrome Dino game using Python and Deep learning

## Model
for the model i used a DQN algorithm with a CnnPolicy on my game environment from stable_baselines3 library for a total of 10000 steps. Optimal models were saved every 1000 steps so in the end i had a total of 10 trained models.

## Steps
- Created a WebGame class environment with a step, render, reset, close, get_observaion and a get_done functions for each of the Dino game needs
- Tested the environment functionality and run it on 10 games
- Created a CallBack function for saving the model on specific steps i chose
- Created the DQN model and trained it with 10000 steps (10 total models)

## Results
**This project was done for educational purposes and getting intoduced to reinforced machine learning methods. That means that for optimal results alot more training steps are required but with this sample it is still visible that there is a significant improvement.
### Before training

### After training

## Requirements
- stable_baselines3
- pyautogui
- pytesseract
- cv2
- mss
- gym
