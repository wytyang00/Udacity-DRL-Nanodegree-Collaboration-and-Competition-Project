# Memory Buffer & Agent Hyperparameters
SEED = 2                # seed for random number generation
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 1024       # minibatch size
START_SINCE = 1024      # number of steps to collect before start learning
GAMMA = 0.95            # discount factor
TAU = 0.2               # for soft update of target parameters
INIT_EPS = 0.           # initial value for the standard deviation of the OU process
ACTOR_LR = 1e-3         # learning rate for the actor
CRITIC_LR = 1e-3        # learning rate for the critic
WEIGHT_DECAY = 0        # Weight decay value
UPDATE_EVERY = 1        # how often to do update steps
N_UPDATES = 1           # how many updates to do every updates
A = 0.                  # randomness vs priority parameter
INIT_BETA = 0.          # initial importance-sampling weight
P_EPS = 1e-3            # priority epsilon
N_STEPS = 1             # multi-step number of steps
V_MIN = -0.1            # distributional learning maximum support bound
V_MAX = 0.1             # distributional learning minimum support bound
CLIP = None             # gradient clipping (`None` to disable)

# Model Hyperparameters
N_ATOMS = 51            # number of atoms used in distributional network
INIT_SIGMA = 0.500      # initial noise parameter values
LINEAR = 'noisy'        # type of linear layer ('linear', 'noisy')
FACTORIZED = True       # whether to use factorized gaussian noise
DISTRIBUTIONAL = True   # whether to use distributional learning


# Default Checks
assert isinstance(SEED, int), "invalid default SEED"
assert isinstance(BUFFER_SIZE, int) and BUFFER_SIZE > 0, "invalid default BUFFER_SIZE"
assert isinstance(BATCH_SIZE, int) and BATCH_SIZE > 0, "invalid default BATCH_SIZE"
assert isinstance(START_SINCE, int) and START_SINCE >= BATCH_SIZE, "invalid default START_SINCE"
assert isinstance(GAMMA, (int, float)) and 0 <= GAMMA <= 1, "invalid default GAMMA"
assert isinstance(TAU, (int, float)) and 0 <= TAU <= 1, "invalid default TAU"
assert isinstance(INIT_EPS, (int, float)) and 0 <= INIT_EPS <= 1, "invalid default INIT_EPS"
assert isinstance(ACTOR_LR, (int, float)) and ACTOR_LR >= 0, "invalid default ACTOR_LR"
assert isinstance(CRITIC_LR, (int, float)) and CRITIC_LR >= 0, "invalid default CRITIC_LR"
assert isinstance(WEIGHT_DECAY, (int, float)) and WEIGHT_DECAY >= 0, "invalid default WEIGHT_DECAY"
assert isinstance(UPDATE_EVERY, int) and UPDATE_EVERY > 0, "invalid default UPDATE_EVERY"
assert isinstance(N_UPDATES, int) and N_UPDATES > 0, "invalid default N_UPDATES"
assert isinstance(A, (int, float)) and 0 <= A <= 1, "invalid default A"
assert isinstance(INIT_BETA, (int, float)) and 0 <= INIT_BETA <= 1, "invalid default INIT_BETA"
assert isinstance(P_EPS, (int, float)) and P_EPS >= 0, "invalid default P_EPS"
assert isinstance(N_STEPS, int) and N_STEPS > 0, "invalid default N_STEPS"
assert isinstance(V_MIN, (int, float)) and isinstance(V_MAX, (int, float)) and V_MIN < V_MAX, "invalid default V_MIN"
if CLIP: assert isinstance(CLIP, (int, float)) and CLIP >= 0, "invalid default CLIP"
assert isinstance(N_ATOMS, int) and N_ATOMS > 0, "invalid default N_ATOMS"
assert isinstance(INIT_SIGMA, (int, float)), "invalid default INIT_SIGMA"
assert isinstance(LINEAR, str) and LINEAR.lower() in ('linear', 'noisy'), "invalid default LINEAR"
assert isinstance(FACTORIZED, bool), "invalid default FACTORIZED"
assert isinstance(DISTRIBUTIONAL, bool), "invalid default DISTRIBUTIONAL"
