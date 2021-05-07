'''
Run fully decentralized structure 10 times, and compute the variance and mean.
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
ACTION = 4 # 5 actions: up, down, left, right, stay
EPISODES = 50000 # total episode
SHOW_EVERY = 3000 # or 1 if with trained 3000
STEP = 200 # how many steps take in one episode/train
POS_NUM = [1,2,3,\
    5,6,7,8,9,
    10,11,12,13,14,
    15,16,17,18,19,
    21,22,23] # possible space


ALPHA = 0.2 # learning rate
DISCOUNT = 0.95 # future reward

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

start_q_table1 = None
#start_q_table1 = "/Users/mac-pro/Desktop/21Spring/EE556/project/fully-decentralised-fixed-start-q1-1.pickle" # or filename(one specific q_table)
start_q_table2 = None
#start_q_table2 = "/Users/mac-pro/Desktop/21Spring/EE556/project/fully-decentralised-fixed-start-q2-1.pickle" # or filename(one specific q_table)
start_q_table3 = None
#start_q_table3 = "/Users/mac-pro/Desktop/21Spring/EE556/project/fully-decentralised-fixed-start-q3-1.pickle" # or filename(one specific q_table)


'''
run 10 times
'''
T = 10
final_reward = []
for t in range(T):

    # action random; will decrease with episode increase
    epsilon = 0.9 # or 0.0 if with trained 0.9
    EP_DECAY = 0.9999

    if start_q_table1 is None:
        # observation space: ((x1,y1),(x2,y2))
        q_table1 = {}
        for p1 in POS_NUM:
            # initial random start space with 5 actions
            q_table1[p1] = np.random.uniform(-10,0,(ACTION,))
    else:
        with open(start_q_table1, "rb") as f:
            q_table1 = pickle.load(f)

    if start_q_table2 is None:
        # observation space: ((x1,y1),(x2,y2))
        q_table2 = {}
        for p2 in POS_NUM:
            # initial random start space with 5 actions
            q_table2[p2] = np.random.uniform(-10,0,(ACTION,))
    else:
        with open(start_q_table2, "rb") as f:
            q_table2 = pickle.load(f)

    if start_q_table3 is None:
        # observation space: ((x1,y1),(x2,y2))
        q_table3 = {}
        for p3 in POS_NUM:
            # initial random start space with 5 actions
            q_table3[p3] = np.random.uniform(-10,0,(ACTION,))
    else:
        with open(start_q_table3, "rb") as f:
            q_table3 = pickle.load(f)



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
                a1 = np.argmax(q_table1[obs[0]]) 
                a2 = np.argmax(q_table2[obs[1]])
                a3 = np.argmax(q_table3[obs[2]])
            else: 
                a1 = np.random.randint(0,ACTION)
                a2 = np.random.randint(0,ACTION)
                a3 = np.random.randint(0,ACTION)
            # do action if not done
            if not dones[0]: car1.action(a1)
            if not dones[1]: car2.action(a2)
            if not dones[2]: car3.action(a3)

            # next state: new observation state
            new_obs = (car1.pos, car2.pos, car3.pos)
            # reward of this action, update the q_table, whether hit the target(done)
            rewards,next_dones = agent.reward([car1,car2,car3],[target1,target2,target3],[a1,a2,a3],dones)
            # update q_table1
            if next_dones[0]: # True, hit target, end
                q_table1[obs[0]][a1] = rewards[0]
            else:
                TD = rewards[0] + DISCOUNT*np.max(q_table1[new_obs[0]]) - q_table1[obs[0]][a1]
                q_table1[obs[0]][a1] += ALPHA*TD
            # update q_table2
            if next_dones[1]: # True, hit target
                q_table2[obs[1]][a2] = rewards[1]
            else:
                TD = rewards[1] + DISCOUNT*np.max(q_table2[new_obs[1]]) - q_table2[obs[1]][a2]
                q_table2[obs[1]][a2] += ALPHA*TD
            # update q_table3
            if next_dones[2]: # True, hit target
                q_table3[obs[2]][a3] = rewards[2]
            else:
                TD = rewards[2] + DISCOUNT*np.max(q_table3[new_obs[2]]) - q_table3[obs[2]][a3]
                q_table3[obs[2]][a3] += ALPHA*TD
            # update total rewards
            reward = np.sum(rewards)


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
                    if cv2.waitKey(1) & 0xFF == ord("q"): #1 or 250
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

    final_reward.append(episode_rewards[-1])
    # convolution of two one dimentional array
    moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode="valid")
    # plot the reward
    plt.plot([i for i in range(len(moving_avg))],moving_avg,color='gray')
    
    
plt.ylabel("reward: "+str(SHOW_EVERY))
plt.xlabel("episode num")
plt.show()

print(np.var(final_reward))
print(np.mean(final_reward))


