# https://github.com/pylSER/Deep-Reinforcement-learning-Mountain-Car
# Q learning in the book, seep age 78

from typing import Tuple
from copy import deepcopy
import random

import gym
import numpy as np
import keras
from keras.layers import Dense
from keras.optimizers import Adam
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

LEARNING_RATE = 0.005

class Memory:
    def __init__(self):
        self.events = []
    def remember(self, event):
        self.events.append(event)
    def sample(self, size=50): #return 50 samples
        if len(self.events) >= size:
            return random.sample(self.events, (size))
        else: 
            return []

class Epsilon:
    def __init__(self, start=1.0, min=0.1, decay=0.1):
        self.value = start
        self.min = min
        self.decay = decay
    def __lt__(self, other: float) -> bool:
        return self.value < other
    def decay(self):
        self.value = min(self.value-self.decay, self.min)

class Predictor:
    # Note on data shape: keras expects data in the shape (N, m)
    # with N the number of training examplesa and the number of 
    # possible clauses
    def __init__(self, in_dim: int, out_dim: int):
        print(f"in_dim: {in_dim}, out_dim: {out_dim}")
        self.model = keras.Sequential([
            Dense(24, activation='relu', input_shape=(in_dim,)),
            Dense(48, activation='relu'),
            Dense(24, activation='relu'),
            Dense(out_dim)
        ])
        self.model.compile(
            loss="mean_squared_error",
            optimizer=Adam(lr=LEARNING_RATE)
        )
        self.in_dim = in_dim
        self.out_dim = out_dim

    def clone(self):
        clone = Predictor(self.in_dim, self.out_dim)
        clone.set_weights(self.model.get_weights())
        return clone

    #predict effect of each action
    def predict(self, state):
        state = state.reshape((1,self.in_dim)) #1 case with 2 inputs
        res = self.model.predict(state)[0] #unpack list of lists as we have only one case
        return res

    def fit(self, state, predicted_actions):
        state = state.reshape((1,self.in_dim)) #1 state with 2 inputs
        predicted_actions = predicted_actions.reshape((1,self.out_dim)) #1 prediction with 3 actions
        self.model.fit(state, predicted_actions, verbose=0)

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

def replay_and_train(memory, model, sample_size):
    gamma = 0.85 #look into gamma

    data_before = np.empty((sample_size, model.in_dim))
    data_predicted_effects = np.empty((sample_size, model.out_dim))
    # TODO rewrite to do fit all in one pass using data shape 
    # instead of multiple passes in the for loop
    # TODO rewrite further to also bundle the predicts 
    for i, (before, action, after, score, game_ended) in enumerate(memory.sample(size=sample_size)):
        predicted_effects = model.predict(before) #returns the tree actions
        if game_ended: #set the correct value for the taken action
            #print("actually won once")
            predicted_effects[action] = score
        else:
            predicted_future_effects = model.predict(after)
            predicted_effects[action] = score+ gamma*max(predicted_future_effects)
        #print(f"before: {before}, predicted_effects: {predicted_effects}")
        data_before[i] = before
        data_predicted_effects[i] = predicted_effects
    model.model.fit(data_before, data_predicted_effects, verbose=0)

def best_action(state: np.ndarray, model: Predictor, env, epsilon: Epsilon) -> int:
    if np.random.random() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(model.predict(state))


def main():
    MAX_STEPS = 200
    SAMPLE_SIZE = 30
    
    env = gym.make("MountainCar-v0")
    env._max_episode_steps = MAX_STEPS+1

    env_state_dim = env.observation_space.shape[0]
    env_action_dim = (env.action_space.__dict__["n"])
    model = Predictor(env_state_dim, env_action_dim)
    model_train = model.clone()
    memory = Memory()
    epsilon = Epsilon()

    for training_session in range(1000):
        before = env.reset()

        if training_session % 50 ==0:
            env.render()

        for step in range(MAX_STEPS):
            #if training_session % 10 == 0:
            #    env.render()
            action = best_action(before, model, env, epsilon)
            after, score, success, debug = env.step(action) # returns [state: numpy.ndarray, reward: float, done: bool, debug: dict]

            if success:
                epsilon.decay()
                score = 10

            memory.remember((before, action, after, score, game_over))
            if step > SAMPLE_SIZE: #ensure some exploration has been done
                replay_and_train(memory, model, model_train, SAMPLE_SIZE)
            before = after
    
            if success:
                env.render()
                print(after, score, game_over, debug)
                #input("Press Enter to continue...")
                break
            

        print(f"training session {training_session} done in {step+1} steps")
        model.set_weights(model_train.get_weights())

if __name__ == "__main__":
    main()