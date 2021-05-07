'''
Fully centralized structure
* Note: part of the code is from
 https://pythonprogramming.net/q-learning-reinforcement-learning-python-tutorial/
'''


import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time
from agent_env import agent

style.use("ggplot")

SIZE = 5 # 5*5 grid
ACTION = 5 # 5 actions: up, down, left, right, stay
EPISODES = 50000 # total episode 50000
SHOW_EVERY = 3000 # or 1 if with trained 3000
STEP = 200 # how many steps take in one episode/train
POS_NUM = [1,2,3,\
    5,6,7,8,9,
    10,11,12,13,14,
    15,16,17,18,19,
    21,22,23] # possible space

CAR1_N = 1
CAR2_N = 2
CAR3_N = 3
TARGET_N = 4
NOTWALL_N = 5
d = {1: (255,0,0),# blue
     2: (0,0,255), # red
     3: (0,255,0), # green
     4: (0,160,255), # orange 
     5: (255,255,255)} # white


# action random; will decrease with episode increase
epsilon = 0.9 # or 0.0 if with trained 0.9
EP_DECAY = 0.9998

ALPHA = 0.5 # learning rate
DISCOUNT = 0.95 # future reward


'''
initial q_table
'''
start_q_table = None
#start_q_table = "/Users/mac-pro/Desktop/21Spring/EE556/project/fully-centralised-1619993254.pickle" # or filename(one specific q_table)

if start_q_table is None:
    # observation space: ((x1,y1),(x2,y2))
    q_table = {}
    for p1 in POS_NUM:
        for p2 in POS_NUM:
            for p3 in POS_NUM:
                # initial random start space with 5 actions
                q_table[(p1,p2,p3)] = np.random.uniform(-10,0,(5,5,5))
else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)




'''
train
'''
episode_rewards = []

for episode in range(EPISODES):

    # initial environment
    car1 = agent(9)
    car2 = agent(14)
    car3 = agent(19)
    target1 = agent(3)
    target2 = agent(5)
    target3 = agent(23)

    # decide whether to show the result of this episode
    if episode % SHOW_EVERY == 0:
        print("episode: "+str(episode)+", epsilon: "+str(epsilon))
        print(str(SHOW_EVERY)+" ep mean: "+str(np.mean(episode_rewards[-SHOW_EVERY:])))
        show = True
    else:
        show = False

    # start train this episode
    episode_reward = 0
    dones = [False,False,False]
    for i in range(STEP): # run upto 200 steps
        # current state: observation state from car, target1, target2
        obs = (car1.pos, car2.pos, car3.pos)
        # choose actions
        if np.random.random()>epsilon:
            act = np.argmax(q_table[obs]) # index: 0 - 24
            a1 = act//(ACTION*ACTION)
            a2 = (act-a1*ACTION*ACTION)//ACTION
            a3 = act-a1*ACTION*ACTION-a2*ACTION
        else: 
            a1 = np.random.randint(0,ACTION)
            a2 = np.random.randint(0,ACTION)
            a3 = np.random.randint(0,ACTION)
        # do action if not done
        if not dones[0]: car1.action(a1)
        if not dones[1]: car2.action(a2)
        if not dones[2]: car3.action(a3)
        '''
        target1.move()
        target2.move()
        '''
        # next state: new observation state
        new_obs = (car1.pos, car2.pos, car3.pos)
        # reward of this action, update the q_table, whether hit the target(done)
        rewards,next_dones = agent.reward([car1,car2,car3],[target1,target2,target3],[a1,a2,a3],dones)
        # update q_table
        if np.all(next_dones): # all hit, end
            q_table[obs][a1,a2,a3] = np.sum(rewards)
        else: # update
            TD = np.sum(rewards) + DISCOUNT*np.max(q_table[new_obs]) - q_table[obs][a1,a2,a3]
            q_table[obs][a1,a2,a3] += ALPHA*TD
        # update total rewards
        reward = np.sum(rewards)

        '''
        show this episode/train every SHOW_EVERY trainings
        '''
        if show:
            env = np.zeros((SIZE,SIZE,3),dtype=np.uint8)
            # change color if agent at this position
            for i in POS_NUM:
                pos_x = i % SIZE
                pos_y = i // SIZE
                env[pos_y][pos_x] = d[NOTWALL_N]
            env[target1.y][target1.x] = d[TARGET_N]
            env[target2.y][target2.x] = d[TARGET_N]
            env[target3.y][target3.x] = d[TARGET_N]
            env[car1.y][car1.x] = d[CAR1_N]
            env[car2.y][car2.x] = d[CAR2_N] 
            env[car3.y][car3.x] = d[CAR3_N] # cover target if hit

            img = Image.fromarray(env,"RGB")
            img = img.resize((300,300),resample=Image.BOX)
            cv2.imshow("test1",np.array(img))
            # if hit, then stay longer; otherwise, just skip quick
            if np.all(next_dones):
                if cv2.waitKey(500) & 0xFF == ord("q"): #500
                        break
            else:
                if cv2.waitKey(250) & 0xFF == ord("q"): #1 or 250
                        break
        
        # update episode reward
        episode_reward += reward

        # if all hit the target, stop step; else update dones
        if np.all(next_dones):
            break
        else:
            dones = next_dones
    
    # append reward in the episode
    episode_rewards.append(episode_reward)
    # update epsilon for each episode
    epsilon *= EP_DECAY

# convolution of two one dimentional array
moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode="valid")
# plot the reward
plt.plot([i for i in range(len(moving_avg))],moving_avg)
plt.ylabel("reward: "+str(SHOW_EVERY))
plt.xlabel("episode num")
plt.show()
# save q_table
with open(f"fully-centralised-{int(time.time())}.pickle", "wb") as f:
     pickle.dump(q_table,f)