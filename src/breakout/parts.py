import numpy as np
import keras
from keras.optimizers import RMSprop
import time

from params import *

FLAT_IMG_SIZE = 5760

class State:
    SINGLE_STATE_SIZE = 11523 #flat size of state = 2*80*72+3 = 5763 (action, score, done = 3)
    def __init__(self):
        self.before = np.empty((80,72,4), dtype=np.uint8) #last state at -1
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

    def flatten_into(self, data):
        data[0:FLAT_IMG_SIZE] = self.before[:,:,-1].flat
        data[FLAT_IMG_SIZE:2*FLAT_IMG_SIZE] = self.after[:,:,-1].flat
        data[2*FLAT_IMG_SIZE+0] = self.action
        data[2*FLAT_IMG_SIZE+1] = self.score
        data[2*FLAT_IMG_SIZE+2] = int(self.done)


class Memory:

    def __init__(self, maxlen=REPLAY_MEMORY_SIZE):
        self.data = np.empty((maxlen, State.SINGLE_STATE_SIZE), dtype=np.uint8)
        self.start = 0
        self.end = 0

    def remember(self, state: State):
        state.flatten_into(self.data[self.end])

        self.end = (self.end + 1) % len(self.data)
        # end == start and yet we just added one element. This means the buffer has one
        # too many element. Remove the first element by incrementing start.
        if self.end == self.start:
            self.start = (self.start + 1) % len(self.data)

    def create_state(self, i):
        state = State()
        for j in range(4): #add the states 
            state.before[:,:,j] = self.data[i+j,0:FLAT_IMG_SIZE].reshape((80,72))
            state.after[:,:,j] = self.data[i+j,FLAT_IMG_SIZE:2*FLAT_IMG_SIZE].reshape((80,72))

        state.action = self.data[i+3, 2*FLAT_IMG_SIZE+0]
        state.score = self.data[i+3, 2*FLAT_IMG_SIZE+1]
        state.done = self.data[i+3, 2*FLAT_IMG_SIZE+1]
        state.push
        return state

    def random_valid_state(self):
        while(True): #skip states that do not have 3 consecutive non-failing states before them
            i = np.random.randint(3, len(self))#take a state
            if (self.data[i+2, -1] == int(False) and 
                self.data[i+1, -1] == int(False) and
                self.data[i+0, -1] == int(False)):
                
                return self.create_state(i)

    def sample(self, size): #return 50 samples
        samples = []
        for _ in range(size):
            state = self.random_valid_state()
            samples.append(state)
        return samples

    def __len__(self):
        return self.end-self.start


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

        
