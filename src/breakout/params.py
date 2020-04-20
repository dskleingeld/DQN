MAX_STEPS = 10_000_000 #should aim to do 10m
BATCH_SIZE = 32
#REPLAY_MEMORY_SIZE = 1_000_000 #might be to large for sysmem to fit
REPLAY_MEMORY_SIZE = 1_000_000//4 #ran out of sysmem, this should take up around 12G
#agent history len: 4
DISCOUNT_FACTOR = 0.99 #aka gamma
#action repeat: 4
UPDATE_FREQ = 4 #TODO FIXME look into

LEARNING_RATE = 0.00025
GRADIENT_MOMENTUM = 0.95
SQUARED_GRADIENT_MOMENTUM = 0.95
MIN_SQUARED_GRADIENT = 0.01

INIT_EXPLORATION = 1.0
FINAL_EXPLORATION = 0.1
NO_EXPLORATION_DECAY = 0
EXPLORATION_DECAY_LEN = 1_000_000
REPLAY_START_SIZE = 50_000

#profiling prams override:
#MAX_STEPS = REPLAY_START_SIZE+50_000
