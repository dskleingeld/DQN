from ringbuf import RingBuf
import numpy as np
import keras
from keras.optimizers import RMSprop
import time

from params import *

class State:
    def __init__(self):
        self.before = np.empty((80,72,4), dtype=np.uint8)
        self.after = np.empty((80,72,4), dtype=np.uint8)
        self.action = None
        self.score = None
        self.done = None
    def roll(self):
        self.before = np.roll(self.before,-1, axis=2)
        self.after = np.roll(self.after,-1, axis=2)
    def push(self, before:np.ndarray, after:np.ndarray, action: int, score: int, done: bool):
        self.roll()
        self.before[:,:,3] = before
        self.after[:,:,3] = after
        
        self.action = action
        self.score = score
        self.done = done


class Memory:
    def __init__(self, maxlen=REPLAY_MEMORY_SIZE):
        self.events = RingBuf(maxlen)

    def remember(self, event: State):
        self.events.append(event)

    def sample(self, size): #return 50 samples
        indices = np.random.randint(0, len(self.events), size)
        samples = [self.events[i] for i in indices]
        return samples

    def __len__(self):
        return len(self.events)

class Epsilon: #simple epsilon decay wont work, need to start with epsilon 1 for a long time
    def __init__(self, decay_len_in_steps: int, no_decay_in_steps: int, start=INIT_EXPLORATION, stop=FINAL_EXPLORATION):
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
            loss="mse", #from Human-level control through deep reinforcement learning paper
            optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
        )

    def clone(self, model_spec):
        clone = Predictor(model_spec)
        clone.model.set_weights(self.model.get_weights())
        return clone

    #predict effect of each action
    def predict(self, state):
        state = state.reshape((1,)+state.shape) #1 case with 2 inputs
        state = state/255
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

    def load(self, path: str):
        self.model.load_weights(path)


def seconds_to_duration_str(secs: float):
    total_minutes = int(secs) // 60
    total_hours = total_minutes // 60
    total_days = total_hours // 24
    return "{:>1} days, {:>2} hours, {:>2} minutes".format(total_days, total_hours%24, total_minutes%60)


ETA_PERIOD = 10_000
class Stats:
    def __init__(self, max_steps):
        self.session_score = 0
        self.highscore = 0
        self.prev_steps = 0
        self.max_steps = max_steps

        self.next_eta_print = REPLAY_START_SIZE
        self.prev_eta_time = time.time()
    def update(self, score):
        self.session_score += score
    
    def eta(self, step):
        now = time.time()
        rate = ETA_PERIOD/(now-self.prev_eta_time)
        eta = (self.max_steps-step)/rate
        self.prev_eta_time = now
        self.next_eta_print = step+ETA_PERIOD
        return seconds_to_duration_str(eta)

    def handle(self, model_train, step, epsilon):
        if self.session_score > self.highscore:
            self.highscore = self.session_score
            path = "data/score_{}_weights.h5".format(self.highscore)
            model_train.save(path) 
        
        if step > self.next_eta_print:
            self.next_eta_print += ETA_PERIOD 
            print("score: {:<3}, epsilon: {:<5.3f}, session took: {:<5} steps, done: {:<5.3f}%, eta: {}".format(
                self.session_score, 
                epsilon, 
                step-self.prev_steps, 
                100*step/self.max_steps,
                self.eta(step)
            ))
        else:
            print("score: {:<3}, epsilon: {:<5.3f}, session took: {:<5} steps, done: {:<5.3f}%".format(
                self.session_score, 
                epsilon, 
                step-self.prev_steps, 
                100*step/self.max_steps
            ))
            
        self.prev_steps = step
        self.session_score = 0

        