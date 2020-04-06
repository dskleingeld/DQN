# https://towardsdatascience.com/getting-started-with-reinforcement-learning-and-open-ai-gym-c289aca874f
# https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c
# Q learning in the book, seep age 78

from typing import Tuple

import gym
import numpy as np
import keras
from keras.layers import Dense
#import random
# #
# Type: Box(2)
# Num 	Observation 	Min 	Max
# 0 	position 	-1.2 	0.6
# 1 	velocity 	-0.07 	0.07
# Actions

# Type: Discrete(3)
# Num 	Action
# 0 	push left
# 1 	no push
# 2 	push right

#
# In the game’s default arrangement, for each time step where the car’s 
# position is <0.5, it receives a reward of -1, up to a maximum of 200 time steps
#

# DQN, instaed of discretising state space and then table use neural netw 

#env 

print("state_space: {}".format(env.observation_space)) 
print("action space: {}".format(env.action_space))

#env.observation_space.low[0] = car position
#env.observation_space.low[1] = car velocity
print(env.observation_space.low)
print(env.observation_space.high)

class Memory:
    def __init__(self):
        self.events = np.empty()
    def remember(self, event):
        self.events.append(event)
    def sample(self): #return 50 samples
        return np.random.choice(self.events, 50) 

class Predictor:
    def __init__(self, in_dim: Tuple[int], out_dim: Tuple[int]):
        self.model = keras.Sequential([
            Dense(input_shape=in_dim),
            Dense(24, activation='relu'),
            Dense(48, activation='relu'),
            Dense(24, activation='relu')
            Dense(out_dim)
        ])

#TODO implement DQN with infrequent weight updates (see page 192 and second blog post above)

# "ingredients":
#  - "memory" => class Memory
#  - Q table, in the form of a neural network => class Predictor
#  - exploration/exploitation switch,
#  - training the model
#  - replaying from memory
#
# exploitation being using the network, exploration doing something at random
# neural network approximates the Q 

# training session consists of multiple "trials". We continue each trail for N steps before we cut it off and start again the network and replay buffer is kept in between trials. Each step we:
# - act on the envirement (exploration/exploitation switch)
# - note the new state
# - remember the old state, new state, current reward and the action that brought us there
# - train the neural net on everything in the network, effectively updating the "Q table" that is the neural net.


def best_action(state: np.ndarray, model: Predictor) -> int:
    epsilon = 0.95 #TODO look into epsilon decay
    if np.random.random() < epsilon:

def main():
    env = gym.make("MountainCar-v0")

    env_state_dim = env.observation_space.shape
    env_action_dim = (env.action_space.__dict__["n"])
    model = Predictor(env_state_dim, env_action_dim)
    memory = Memory()

    for training_session in range(200):
        state = env.reset()

        for step in range(500):

            action = best_action(state, model)

    env.step() # returns [state: numpy.ndarray, reward: float, done: bool, debug: dict]


if __name__ == "__main__":
    main()