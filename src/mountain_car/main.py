# https://github.com/pylSER/Deep-Reinforcement-learning-Mountain-Car
# Q learning in the book, seep age 78

from typing import Tuple
from copy import deepcopy
from collections import deque
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

#set keras to use gpu
LEARNING_RATE = 0.001

class Memory:
    def __init__(self, maxlen=20000):
        self.events = deque(maxlen=maxlen)

    def remember(self, event):
        self.events.append(event)

    def sample(self, size=50): #return 50 samples
        if len(self.events) >= size:
            return random.sample(self.events, (size))
        else: 
            return []
    def __len__(self):
        return len(self.events)

class Epsilon:
    def __init__(self, start=1.0, min=0.01, decay=0.00005): #was 1.0, 0.01, 0.05
        self.value = start
        self.min = min
        self.decay_by = decay
    def decay(self):
        self.value = max(self.value-self.decay_by, self.min)

class Predictor:
    # Note on data shape: keras expects data in the shape (N, m)
    # with N the number of training examplesa and the number of 
    # possible clauses
    def __init__(self, in_dim: int, out_dim: int):
        print(in_dim)
        print(out_dim)
        self.model = keras.Sequential([
            Dense(24, activation='relu', input_shape=(in_dim,)),
            Dense(48, activation='relu'),
            #Dense(24, activation='relu'),
            Dense(out_dim, activation="linear")
        ])
        self.model.compile(
            loss="mean_squared_error",
            optimizer=Adam(lr=LEARNING_RATE)
        )
        self.in_dim = in_dim
        self.out_dim = out_dim

    def clone(self):
        clone = Predictor(self.in_dim, self.out_dim)
        clone.model.set_weights(self.model.get_weights())
        return clone

    #predict effect of each action
    def predict(self, state):
        state = state.reshape((1,self.in_dim)) #1 case with 2 inputs
        res = self.model.predict(state)[0] #unpack list of lists as we have only one case
        return res

    def predict_batch(self, states):
        return self.model.predict(states)

    def fit(self, state, predicted_actions):
        state = state.reshape((1,self.in_dim)) #1 state with 2 inputs
        predicted_actions = predicted_actions.reshape((1,self.out_dim)) #1 prediction with 3 actions
        self.model.fit(state, predicted_actions, verbose=0)

    def copy_weights_from(self, other):
        self.model.set_weights(other.model.get_weights())

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


def samples_to_states(samples):
    state_dim = len(samples[0][0])
    before_states = np.empty((len(samples), state_dim ))
    after_states = np.empty((len(samples), state_dim ))
    for i, (before, _, after, _, _) in enumerate(samples):
        before_states[i] = before
        after_states[i] = after
    return before_states, after_states

def replay_and_train(memory: Memory, model: Predictor, model_train: Predictor, sample_size: int):
    gamma = 0.95 #look into gamma

    samples = memory.sample(sample_size)
    before_states, after_states = samples_to_states(samples)
    predict_effects = model.predict_batch(before_states) #for entire sample
    predict_future_effects = model.predict_batch(after_states) #for entire sample

    for i, (_, action, _, score, succes) in enumerate(samples):
        if succes:
            predict_effects[i][action] = score
        else:
            predict_effects[i][action] = score + gamma*max(predict_future_effects[i])

    model.model.fit(before_states, predict_effects, verbose=0) #TODO FIXME remove
    #model_train.model.fit(before_states, predict_effects, verbose=0) #TODO FIXME re-enable

def best_action(state: np.ndarray, model: Predictor, env, epsilon: Epsilon) -> int:
    epsilon.decay()
    if np.random.rand(1) <= epsilon.value:
        #print("random")
        #action = np.random.randint(0, 3)
        action = env.action_space.sample()
        #print(action)
        return action
    else:
        return np.argmax(model.predict(state))


def main():
    MAX_STEPS = 200
    SAMPLE_SIZE = 32
    MINIMAL_MEMORY_SIZE = 400
    
    env = gym.make("MountainCar-v0")
    env._max_episode_steps = MAX_STEPS+1

    env_state_dim = env.observation_space.shape[0]
    env_action_dim = (env.action_space.__dict__["n"])
    model = Predictor(env_state_dim, env_action_dim)
    model_train = model.clone()
    memory = Memory(maxlen=40_000)
    epsilon = Epsilon()
    print(epsilon.value)

    for training_session in range(1000):
        before = env.reset()

        for step in range(MAX_STEPS):
            #if training_session % 100 == 0:
            #    env.render()
            action = best_action(before, model, env, epsilon)
            after, score, success, debug = env.step(action) # returns [state: numpy.ndarray, reward: float, done: bool, debug: dict]

            if success:
                score = 10

            memory.remember((before, action, after, score, success))
            if len(memory) > MINIMAL_MEMORY_SIZE: #ensure some exploration has been done
                replay_and_train(memory, model, model_train, SAMPLE_SIZE)
            before = after
    
            if success:
                print(after, score, success, debug)
                break

        print("training session {} done in {} steps, espsilon: {}".format(training_session, step+1, epsilon.value))
        #model.copy_weights_from(model_train) #TODO FIXME re-enable

if __name__ == "__main__":
    main()