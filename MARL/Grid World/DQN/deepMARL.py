'''
DQN
'''

import numpy as np
import tensorflow as tf
import keras.backend.tensorflow_backend as backend
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from collections import deque
import random
import time
from tqdm import tqdm
import os
from PIL import Image
import cv2
import matplotlib.pyplot as plt


SHOW_EVERY = 50

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 32  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = '2x256'
MIN_REWARD = -200  # For model save
MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 2000

# Exploration settings
epsilon = 0.9  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = False # True if want to see how it runs


LOAD_MODEL = None


'''
Agent
'''
class Agent:
    def __init__(self, size, pos, pos_num):
        self.size = size
        self.pos_num = pos_num
        if pos < 0:
            self.pos = pos_num[np.random.randint(0,np.shape(pos_num)[0])]
            self.x = self.pos % size
            self.y = self.pos // size
        else:
            self.pos = pos
            self.x = self.pos % size
            self.y = self.pos // size

    def __str__(self):
        return f"agent ({self.x}, {self.y})"

    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)
    # check if position is equal
    def hit(self, other):
        return self.pos == other.pos
    def hitAny(self,others):
        hitAnyBool = 0
        for i in range(3):
            if self.hit(others[i]):
                hitAnyBool = i+1
        return hitAnyBool

    def action(self, choice):
        '''
        Gives us 5 total movement options. (0,1,2,3,4)
        '''
        if choice == 0:
            self.move(x=0,y=0)
        elif choice == 1:
            self.move(x=0,y=-1)
        elif choice == 2:
            self.move(x=-1,y=0)
        elif choice == 3:
            self.move(x=1,y=0)
        elif choice == 4:
            self.move(x=0,y=1)

    def move(self, x=False, y=False):
        old_pos = self.pos
        old_x = self.x
        old_y = self.y
        # update x
        if x:
            self.x += x
        # update y
        if y:
            self.y += y
        # if hit the outer wall
        if self.x < 0: self.x = 0
        elif self.x > self.size-1: self.x = self.size-1
        if self.y < 0: self.y = 0
        elif self.y > self.size-1: self.y = self.size-1
        self.pos = self.size * self.y + self.x
        # if hit the black area
        if self.pos not in self.pos_num:
            self.pos = old_pos
            self.x = old_x
            self.y = old_y



'''
Environment
'''
class AgentEnv:
    SIZE = 5
    POS_NUM = [1,2,3,\
        5,6,7,8,9,
        10,11,12,13,14,
        15,16,17,18,19,
        21,22,23] # possible space
    AGENT_N = 3
    RETURN_IMAGES = True
    MOVE_PENALTY = 1
    TIME_PENALTY = 1
    TARGET_REWARD = 50
    HIT_PENALTY = 500
    OBSERVATION_SPACE_VALUES = (AGENT_N,)  # 4
    ACTION_SPACE_SIZE = 5

    CAR1_N = 1
    CAR2_N = 2
    CAR3_N = 3
    TARGET_N = 4
    NOTWALL_N = 5
    # the dict! (colors)
    d = {1: (255,0,0),# blue
        2: (0,0,255), # red
        3: (0,255,0), # green
        4: (0,160,255), # orange 
        5: (255,255,255)} # white

    def reset(self):
        self.car1 = Agent(self.SIZE,9,self.POS_NUM)
        self.car2 = Agent(self.SIZE,14,self.POS_NUM)
        self.car3 = Agent(self.SIZE,19,self.POS_NUM)
        self.target1 = Agent(self.SIZE,3,self.POS_NUM)
        self.target2 = Agent(self.SIZE,5,self.POS_NUM)
        self.target3 = Agent(self.SIZE,23,self.POS_NUM)
        self.done1 = False
        self.done2 = False
        self.done3 = False

        self.episode_step = 0
        '''
        if self.RETURN_IMAGES:
            observation = np.array(self.get_image())
        else:
            observation = (self.player-self.food) + (self.player-self.enemy)
        '''
        return [self.car1.pos,self.car2.pos,self.car3.pos] # observation

    def step(self, actions):
        self.episode_step += 1
        self.car1.action(actions[0])
        self.car2.action(actions[1])
        self.car3.action(actions[2])
        #### MAYBE ###
        #self.enemy.move()
        #self.food.move()
        ##############
        new_observation = [self.car1.pos,self.car2.pos,self.car3.pos]
        rewards, dones = self.reward([self.car1,self.car2,self.car3],[self.target1,self.target2,self.target3],actions,[self.done1,self.done2,self.done3])
        return new_observation, rewards, np.all(dones)

    def reward(self, agents,targets,actions,pre_dones):
        # update dones
        dones = [False,False,False]
        hit_target = [0,0,0]
        for i in range(3):
            hit_target[i] = agents[i].hitAny(targets)
            if hit_target[i] != 0:
                dones[i] = True
        # this step rewrads:
        rewards = [0,0,0]
        for j in range(3):
            if pre_dones[j] == False and dones[j] == False: # no hit
                if actions[j] == 0: # do not move
                    rewards[j] = -self.TIME_PENALTY
                else: # move
                    rewards[j] = -self.MOVE_PENALTY-self.TIME_PENALTY
            elif pre_dones[j] == False and dones[j] == True: # hit
                if hit_target.count(hit_target[j]) == 1: # no same hit
                    rewards[j] = self.TARGET_REWARD-self.MOVE_PENALTY-self.TIME_PENALTY
                else: # same hit
                    rewards[j] = -self.HIT_PENALTY-self.MOVE_PENALTY-self.TIME_PENALTY
            elif pre_dones[j] == True:
                rewards[j] = 0
        self.done1 = dones[0]
        self.done2 = dones[1]
        self.done3 = dones[2]
        return (rewards, dones)


    def render(self):
        img = self.get_image()
        img = img.resize((300, 300))  # resizing so we can see our agent in all its glory.
        cv2.imshow("image", np.array(img))  # show it!
        cv2.waitKey(1)

    # FOR CNN #
    def get_image(self):
        env = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
        env[self.car1.x][self.car1.y] = self.d[self.CAR1_N]  # sets the food location tile to green color
        env[self.car2.x][self.car2.y] = self.d[self.CAR2_N]
        env[self.car3.x][self.car3.y] = self.d[self.CAR3_N]
        env[self.target1.x][self.target1.y] = self.d[self.TARGET_N]  # sets the enemy location to red
        env[self.target2.x][self.target2.y] = self.d[self.TARGET_N]
        env[self.target3.x][self.target3.y] = self.d[self.TARGET_N]
        for i in self.POS_NUM:
            pos_x = i % self.SIZE
            pos_y = i // self.SIZE
            env[pos_y][pos_x] = self.d[self.NOTWALL_N]

        img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
        return img


env = AgentEnv()

# For stats
ep_rewards = [-200]

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.set_random_seed(1)


# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')


'''
Own Tensorboard class
'''
class ModifiedTensorBoard(TensorBoard):
    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)
    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass
    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)
    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass
    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass
    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)
    def _write_logs(self, logs, index):
        with self.writer.__enter__():
            for name, value in logs.items():
                tf.summary.scalar(name, value, collections=[index])
                self.step += 1
                self.writer.flush()



'''
DQN model
'''
class DQNAgent:
    def __init__(self):
        # main model: trained every step
        self.model = self.create_model()
        # target model: what we .predict against every step
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir = f"logs/{MODEL_NAME}-{int(time.time())}")
        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def create_model(self):
        if LOAD_MODEL is not None:
            print(f"Loading {LOAD_MODEL}")
            model = load_model(LOAD_MODEL)
            print(f"Model {LOAD_MODEL} loaded!")
        else:
            model = Sequential() # linear stack of layers
            model.add(Dense(20, input_shape=(env.OBSERVATION_SPACE_VALUES), activation="relu"))
            model.add(Dense(32, activation="relu"))
            model.add(Dense(64, activation="relu"))
            model.add(Dense(128, activation="relu"))
            model.add(Dense(env.ACTION_SPACE_SIZE*env.ACTION_SPACE_SIZE*env.ACTION_SPACE_SIZE, activation="linear"))
            model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self,transition):
        self.replay_memory.append(transition)

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.array([state]))


    def train(self, terminal_state, step):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return 
        # get minibatch from replay_memory randomly ; size = MINIBATCH_SIZE
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # reward, discount, current and futrue q value
        # current states of samples
        current_states = np.array([transition[0] for transition in minibatch]) # normalize it to 0 to 1
        # current q values
        current_qs_list = self.model.predict(current_states)
        # next stats
        new_current_states = np.array([transition[3] for transition in minibatch])
        # next q values
        futrue_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            # get new q after action
            if not done:
                max_future_q = np.max(futrue_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward
            # get current qs
            current_qs = current_qs_list[index]
            # update current qs at action
            current_qs[action] = new_q

            X.append(current_state) # features
            y.append(current_qs) # labels
        
        # fit the model
        self.model.fit(np.array(X), np.array(y), batch_size = MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)
        
        # update to determine if need to update target_model yet
        if terminal_state:
            self.target_update_counter += 1
        
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0


agent = DQNAgent()


'''
train
'''
episode_rewards = []

for episode in tqdm(range(1, EPISODES+1), ascii=True, unit="episode"):
    agent.tensorboard.step = episode

    episode_reward = 0
    step = 1
    current_state = env.reset()

    done = False

    while not done:
        # take actions based on current state
        if np.random.random()>epsilon:
            act = np.argmax(agent.get_qs(current_state))
            a1 = act//(env.ACTION_SPACE_SIZE*env.ACTION_SPACE_SIZE)
            a2 = (act-a1*env.ACTION_SPACE_SIZE*env.ACTION_SPACE_SIZE)//env.ACTION_SPACE_SIZE
            a3 = act-a1*env.ACTION_SPACE_SIZE*env.ACTION_SPACE_SIZE-a2*env.ACTION_SPACE_SIZE
            action = [a1,a2,a3]
        else:
            a1 = np.random.randint(0, env.ACTION_SPACE_SIZE)
            a2 = np.random.randint(0, env.ACTION_SPACE_SIZE)
            a3 = np.random.randint(0, env.ACTION_SPACE_SIZE)
            action = [a1,a2,a3]
        
        # do the action on env and got return
        new_state, rewards, done = env.step(action)
        
        # add accumulated reward
        episode_reward += np.sum(rewards)

        # show result
        if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
            env.render()
        
        # update info to replay_memory
        agent.update_replay_memory((current_state, action, rewards, new_state, done))
        # train the NN
        agent.train(done, step)

        # update the state and step
        current_state = new_state
        step += 1


     # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

        # Save model, but only when min reward is greater or equal a set value
        if min_reward >= MIN_REWARD:
            agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)
    
    episode_rewards.append(episode_reward)

# convolution of two one dimentional array
moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode="valid")
# plot the reward
plt.plot([i for i in range(len(moving_avg))],moving_avg)
plt.ylabel("reward: "+str(SHOW_EVERY))
plt.xlabel("episode num")
plt.show()