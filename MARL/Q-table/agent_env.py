'''
environment
'''
import numpy as np
import pickle
from matplotlib import style


SIZE = 5 # 5*5 grid
ACTION = 5 # 5 actions: up, down, left, right, stay
MOVE_PENALTY = 1 # penalty during each move
TIME_PENALTY = 1 # penalty of each step train
TARGET_REWARD = 50
HIT_PENALTY = 500
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
    


class agent:
    def __init__(self,pos):
        if pos < 0:
            # random initial the start state
            self.pos = POS_NUM[np.random.randint(0,np.shape(POS_NUM)[0])]
            self.x = self.pos % SIZE
            self.y = self.pos // SIZE
        else:
            self.pos = pos
            self.x = self.pos % SIZE
            self.y = self.pos // SIZE
    
    # return the position of agent
    def __str__(self):
        location = "("+str(self.x)+","+str(self.y)+")"
        return location

    # subtraction of two agents - used as states of q value
    def __sub__(self,other):
        return (self.x-other.x, self.y-other.y)
    
    def __dist__(self,other):
        return (self.x-other.x)^2+(self.y-other.y)^2

    # if in the same position
    def hit(self,other):
        if self.pos == other.pos:
            return True
        else:
            return False
    # if hit any position, return True; if hit none return false
    def hitAny(self,others):
        hitAnyBool = 0
        for i in range(3):
            if self.hit(others[i]):
                hitAnyBool = i+1
        return hitAnyBool

    
    # define actions could be done on the agent
    def action(self,choice):
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

    # define how the env should do on specific move
    def move(self,x=False,y=False):
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
        elif self.x > SIZE-1: self.x = SIZE-1
        if self.y < 0: self.y = 0
        elif self.y > SIZE-1: self.y = SIZE-1
        self.pos = SIZE * self.y + self.x
        # if hit the black area
        if self.pos not in POS_NUM:
            self.pos = old_pos
            self.x = old_x
            self.y = old_y
    
    # reward of the environment:
    # agents:[car1,car2,car3]
    # targets:[target1,target2,target3]
    def reward(agents,targets,actions,pre_dones):
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
                    rewards[j] = -TIME_PENALTY
                else: # move
                    rewards[j] = -MOVE_PENALTY-TIME_PENALTY
            elif pre_dones[j] == False and dones[j] == True: # hit
                if hit_target.count(hit_target[j]) == 1: # no same hit
                    rewards[j] = TARGET_REWARD-MOVE_PENALTY-TIME_PENALTY
                else: # same hit
                    rewards[j] = -HIT_PENALTY-MOVE_PENALTY-TIME_PENALTY
            elif pre_dones[j] == True:
                rewards[j] = 0
        return (rewards, dones)
