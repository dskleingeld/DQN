#https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #supress tf warnings

from typing import Tuple, List
from copy import deepcopy
import random

import gym
import numpy as np

from models import models as specs
from models import cropped_scaled_grayscale as reduce_state

from parts import Memory, Predictor, Epsilon, Stats, State
from params import *

#profiling note, costs only 2.8% of time which is pretty alright....
def samples_to_states(samples: List[State]):
    state_dim = samples[0].before.shape
    before_states = np.empty((len(samples),)+state_dim)
    after_states = np.empty((len(samples),)+state_dim)
    for i, state in enumerate(samples):
        before_states[i] = np.reshape(state.before, state_dim)
        after_states[i] = np.reshape(state.after, state_dim)
    return before_states, after_states

def replay_and_train(memory: Memory, model: Predictor, model_train: Predictor, sample_size: int):
    gamma = DISCOUNT_FACTOR
    if len(memory) < REPLAY_START_SIZE:
        return

    samples = memory.sample(sample_size)
    before_states, after_states = samples_to_states(samples)
    predict_effects = model.predict_batch(before_states) #for entire sample
    predict_future_effects = model.predict_batch(after_states) #for entire sample

    for i, state in enumerate(samples):
        if state.done:
            predict_effects[i][state.action] = state.score
        else:
            predict_effects[i][state.action] = state.score + gamma*max(predict_future_effects[i])

    model_train.model.fit(before_states, predict_effects, verbose=0)

def best_action(state: np.ndarray, model: Predictor, env, epsilon: Epsilon) -> int:
    if epsilon.value > np.random.rand() :
        epsilon.decay()
        return env.action_space.sample()
    else:
        epsilon.decay()
        return np.argmax(model.predict(state))

def no_action(epsilon: Epsilon):
    epsilon.decay() #even though this is a no action we do need to decay epsilon
    return 0 #action zero, do nothing

def reset(env, model: Predictor, epsilon: Epsilon, state: State):
    before = env.reset()
    before = reduce_state(before)
    lives = 5

    for _ in range(3):
        action = no_action(epsilon)
        after, score, _, info = env.step(action)
        after = reduce_state(after)
        if info["ale.lives"] < lives:
            lives = info["ale.lives"]
            state.push(before, after, action, score, True)
        else: #can not actually reach game over during reset (more lives then timesteps)
            state.push(before, after, action, score, False)
    return before, lives

def train(model=None):
    env = gym.make("BreakoutDeterministic-v4")
    env_state_dim = env.observation_space.shape
    env_action_dim = (env.action_space.__dict__["n"])
    state = State()
    
    if model is None:
        model = Predictor(specs["deepmind_paper"])
    model_train = model.clone(specs["deepmind_paper"])
    memory = Memory(maxlen=REPLAY_MEMORY_SIZE)

    decay_free = 0
    decay_len = EXPLORATION_DECAY_LEN
    epsilon = Epsilon(decay_len, decay_free)
    stats = Stats(MAX_STEPS)
    before, lives = reset(env, model, epsilon, state)

    #env._max_episode_steps this ensures the game ends if its going well
    for step in range(MAX_STEPS):
        action = best_action(state.before, model, env, epsilon)
        after, score, game_over, info = env.step(action)
        after = reduce_state(after)


        if info["ale.lives"] < lives: #credits: https://github.com/fg91/Deep-Q-Learning/blob/master/DQN.ipynb
            #act as if game over during replay if we lose a life
            lives = info["ale.lives"]
            state.push(before, after, action, score, True)
        else:
            state.push(before, after, action, score, game_over)
        memory.remember(state)
        replay_and_train(memory, model, model_train, BATCH_SIZE)
        

        before = after
        if step % 100 == 0:
            model.copy_weights_from(model_train)
        if step % MAX_STEPS/100 == 0:
            model.save("data/step_{}_weights".format(step))


        stats.update(score)
        if game_over:
            stats.handle(model_train, step, epsilon)
            before, lives = reset(env, model, epsilon, state)

if __name__ == "__main__":
    model = Predictor(specs["deepmind_paper"])
    #model.load("data/score_11.0_weights.h5")
    train(model=model)