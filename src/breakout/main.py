import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #supress tf warnings

from typing import Tuple
from copy import deepcopy
from collections import deque
import random

import gym
import numpy as np
import keras

from models import models as model_specs
from models import cropped_grayscale
from keras.optimizers import Adam

#set keras to use gpu
LEARNING_RATE = 0.00001 #large MAX frames is wal larger #TODO make dependent on max frames?

class Memory:
    def __init__(self, maxlen=20000):
        self.events = deque(maxlen=maxlen)

    def remember(self, event):
        self.events.append(event)

    def sample(self, size): #return 50 samples
            return random.sample(self.events, (size))

    def __len__(self):
        return len(self.events)

class Epsilon: #simple epsilon decay wont work, need to start with epsilon 1 for a long time
    def __init__(self, decay_len_in_steps: int, no_decay_in_steps: int, start=1.0, stop=0.1):
        self.value = start
        self.min = stop
        self.decay_by = (start-self.min)/decay_len_in_steps
        self.no_decay_in_steps = no_decay_in_steps
        self.times_called = 0
    def decay(self):
        if self.times_called > self.no_decay_in_steps:
            self.value = max(self.value-self.decay_by, self.min)
        self.times_called += 1
    def __format__(self, formatstr):
        return self.value.__format__(formatstr)

class Predictor:
    # Note on data shape: keras expects data in the shape (N, m)
    # with N the number of training examplesa and the number of 
    # possible clauses
    def __init__(self, model):
        self.out_dim = 4
        self.model = model
        self.model.compile(
            loss="mean_squared_error",
            optimizer=Adam(lr=LEARNING_RATE)
        )

    def clone(self, model_spec):
        clone = Predictor(model_spec)
        clone.model.set_weights(self.model.get_weights())
        return clone

    #predict effect of each action
    def predict(self, state):
        state = state.reshape((1,)+state.shape+(1,)) #1 case with 2 inputs
        res = self.model.predict(state)[0] #unpack list of lists as we have only one case
        return res

    def predict_batch(self, states):
        return self.model.predict(states, steps=1)# FIXME can we remove steps?

    def fit(self, state, predicted_actions):
        predicted_actions = predicted_actions.reshape((1,self.out_dim))
        self.model.fit(state, predicted_actions, verbose=0)

    def copy_weights_from(self, other):
        self.model.set_weights(other.model.get_weights())

    def save(self, path: str):
        self.model.save_weights(path)

def samples_to_states(samples):
    state_dim = samples[0][0].shape
    before_states = np.empty((len(samples),)+state_dim+(1,))
    after_states = np.empty((len(samples),)+state_dim+(1,))
    for i, (before, _, after, _, _) in enumerate(samples):
        before_states[i] = np.reshape(before, state_dim+(1,))
        after_states[i] = np.reshape(after, state_dim+(1,))
    return before_states, after_states

def replay_and_train(memory: Memory, model: Predictor, model_train: Predictor, sample_size: int):
    gamma = 0.99 #look into gamma
    if len(memory) < sample_size:
        return

    samples = memory.sample(sample_size)
    before_states, after_states = samples_to_states(samples)
    predict_effects = model.predict_batch(before_states) #for entire sample
    predict_future_effects = model.predict_batch(after_states) #for entire sample

    for i, (_, action, _, score, succes) in enumerate(samples):
        if succes:
            predict_effects[i][action] = score
        else:
            predict_effects[i][action] = score + gamma*max(predict_future_effects[i])

    model_train.model.fit(before_states, predict_effects, verbose=0)

def best_action(state: np.ndarray, model: Predictor, env, epsilon: Epsilon) -> int:
    if epsilon.value > np.random.rand() :
        action = env.action_space.sample()
        epsilon.decay()
        return action
    else:
        epsilon.decay()
        return np.argmax(model.predict(state))

def main():
    MAX_STEPS = 1_000_000
    SAMPLE_SIZE = 32
    STRIDE=5 #determined by looking with 1 sec in between
    
    env = gym.make("Breakout-v4")
    env._max_episode_steps = MAX_STEPS+1

    env_state_dim = env.observation_space.shape
    env_action_dim = (env.action_space.__dict__["n"])

    import time
    for spec_name, spec in model_specs.items():
        reduce_state = cropped_grayscale
        model = Predictor(spec)
        model_train = model.clone(spec)
        memory = Memory()

        decay_free = 0.2*MAX_STEPS
        decay_len = 0.3*MAX_STEPS
        epsilon = Epsilon(decay_len, decay_free)
        highscore = 0#6 #only start storing networks if the score exceeds 6

        before = env.reset()
        before = reduce_state(before)
        session_max_score = 0
        lives = 5
        
        prev_steps = 0
        for step in range(MAX_STEPS):
            action = best_action(before, model, env, epsilon)
            #env.render()
            #time.sleep(1)

            cum_score = 0
            #do not think analyse every frame/step
            for _ in range(STRIDE-1):
                _, score, _, _ = env.step(action)
                cum_score += score

            #record state of last step
            after, score, game_over, info = env.step(action)
            after = reduce_state(after)
            cum_score += score
            session_max_score += cum_score

            if info["ale.lives"] < lives: #credits: https://github.com/fg91/Deep-Q-Learning/blob/master/DQN.ipynb
                #act as if game over during replay if we lose a life
                memory.remember((before, action, after, cum_score, True))
            else:
                memory.remember((before, action, after, cum_score, game_over))

            cum_score = 0
            replay_and_train(memory, model, model_train, SAMPLE_SIZE)
            before = after

            if step % 100 == 0:
                model.copy_weights_from(model_train)

            if game_over:
                print(f"game over, score: {session_max_score}")
                before = env.reset()
                before = reduce_state(before)
                session_max_score = 0
                lives = 5
                
                if session_max_score > highscore:
                    path = "data/{}_weights.h5".format(spec_name)
                    model_train.save(path) 
                print("score: {:5}, epsilon: {:5.3f}, session took: {:5} steps".format(session_max_score, epsilon, step-prev_steps))
                prev_steps = step
        

if __name__ == "__main__":
    main()