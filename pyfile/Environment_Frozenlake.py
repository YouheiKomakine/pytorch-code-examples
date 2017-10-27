# Environment_FrozenLake
#   init()
#       input : 
#           map_size=(width,height)     #
#           slip_rate                   # Player will slip when move from "F" (Icey ground) according to slip_rate.
#                                       #   (90 degrees curved from the direction which action you choosed)
#           hole_num=(a,b)              # Number of Hole "n" will choose randomly, s.t. a<=n<=b
#           start=(x,y), goal=(x,y)     # Start / Goal position, (right/down are positive direction)
#           max_retry=(a,b)             # Maximum number of retry to generate map.
#                                       #   a : max num to retry in the same hole_num.
#                                       #      if retry_count > a,  hole_num is decremented.
#                                       #   b : max num to decrement hole_num
#           action                      # dictionary to definit action.
#                                       #   For example, action[i]=(a,b) will change player's position
#                                       #   from (x,y) to (x+a,y+b) (if it can be done)
#           ground_name                 # dictionary to show map with Uppercase letter
#
#   return_next()
#       input  : 
#           action
#       output : 
#           reward                      # +1 (when reach to Goal), -1 (when reach to Hole)
#           next_state                  # int n.  s.t. 0<=n<=(width*height)-1 
#                                       #   When player is on (x,y), n= (y-1)*height + (x-1)
#           episode_end_flag            # True when player move to Goal or Hole, False in other position.


import numpy as np
import copy

class Environment_FrozenLake(object):
    def __init__(self, map_size=(4,4), slip_rate=0.33, hole_num=(1,3), 
                 start=(1,1), goal=(-1,-1),max_retry=(10,10),
                 reach_to_outside_map = False,
                 action={0:(0,-1), 1:(1,0), 2:(0,1), 3:(-1,0)},
                 ground_name={1:"F", 2:"S", 3:"G", 4:"H"}):
        self.map_size = np.array(map_size)
        self.slip_rate = slip_rate
        self.hole_num = np.random.randint(hole_num[0], hole_num[1]+1)
        self.start = np.array((start[0]-1, start[1]-1)) if min(start)>=1 else np.array((0,0))
        self.goal = np.array((goal[0]-1, goal[1]-1)) if min(goal)>=1 else np.array(map_size)+np.array(goal)
        self.max_retry = max_retry
        self.reach_to_outside_map=False
        self.action = action
        self.ground_name = ground_name
        self.ground_name[0] = "N"

        self.player_position = np.array(self.start, dtype=int)
        self.reward = 0
        self.done=False
        
    def check_route(self, map_dup):
        checkmap = copy.deepcopy(map_dup)
        checkmap[(self.start[0],self.start[1])] = 1
        checkmap[(self.goal[0],self.goal[1])] = 0
        for i in range(self.map_size[0] * self.map_size[1]):
            if (0 in checkmap)==False:
                break
            for h in range(self.map_size[1]):
                for w in range(self.map_size[0]):
                    if checkmap[h][w]==0: 
                        if checkmap[max(0,h-1)][w]==1 or checkmap[h][max(0,w-1)]==1 or \
                            checkmap[min(self.map_size[1]-1,h+1)][w]==1 or checkmap[h][min(self.map_size[0]-1,w+1)]==1:
                            checkmap[h][w] = 1
                            
        # for debug
        #print("checkmap, ", checkmap)
        #can_reach=True
                            
        can_reach = True if checkmap[(self.goal[0],self.goal[1])]==1 else False
        return can_reach
                            
    def generate_map(self, max_retry=None):
        # 番号との対応は、１：氷、２：スタート地点、３：ゴール、４：穴
        if (max_retry is None) or (min(max_retry) <= 0) or (max(max_retry) >= 256):
            max_retry = self.max_retry
        self.map = np.zeros(self.map_size, dtype=int)
        self.map[(self.start[0],self.start[1])] = 2
        self.map[(self.goal[0],self.goal[1])] = 3
        retry_count = [0,0]
        success_check = False
        
        # for debug
        #print("start, goal = ", self.start, self.goal)
        #print("generate_map, before, \n", self.map)
                
        while (retry_count[1] < max_retry[1]) and (success_check==False):
            while (retry_count[0] < max_retry[0]) and (success_check==False):
                map_dup = copy.deepcopy(self.map)
                chip_index_list = np.where(self.map==0)
                chip_index = np.random.choice(chip_index_list[0].size, self.hole_num, replace=False)
                
                for i in range(len(chip_index)):
                    map_dup[chip_index_list[0][chip_index[i]]][chip_index_list[1][chip_index[i]]] = 4 
                
                retry_count[0] += 1
                if self.check_route(map_dup):
                    self.map = np.where(map_dup==4, map_dup, self.map)
                    self.map = np.where(self.map==0, 1, self.map)
                    success_check = True
                    break
            else:
                self.hole_num = max(1, self.hole_num-1)
                retry_count[1] += 1
        return success_check

    def show_map(self, raw_data=False):
        map_text = ""
        for h in range(self.map_size[0]):
            for w in range(self.map_size[1]):
                map_text += self.ground_name[self.map[h][w]]
                if w == self.map_size[1]-1:
                    map_text += "\n"
        if raw_data:
            print(self.map)
        else:
            print(map_text)
                
    def return_next(self, action=None, render_flag=False):
        if action is None:
            return self.player_position[1] * self.map_size[0] + self.player_position[0]

        # move direction
        move_p = np.array(self.action[action]) # action={0:(0,-1), 1:(1,0), 2:(0,1), 3:(-1,0)} in default settings
        slip_A, slip_B = move_p[::-1], move_p[::-1]*-1
        move = np.random.choice([move_p, slip_A, slip_B],
                                p=[1-self.slip_rate, self.slip_rate/2, self.slip_rate/2])

        if self.reach_to_outside_map==False:
            self.player_position[0] = min(max(self.player_position[0]+move[0], 0), self.max_size[0]-1)
            self.player_position[1] = min(max(self.player_position[1]+move[1], 0), self.max_size[1]-1)
        else:
            self.player_position = self.player_position + move

        # goal / hole check
        if (self.player_position[0]<0) or (self.map_size[0]-1 < self.player_position[0]) \
            or (self.player_position[1]<0) or (self.map_size[1]-1 < self.player_position[1]) \
            or self.map[self.player_position] == 4: # Hole or Outside of map
            self.reward = -1
            self.done = True
        elif self.map[self.player_position] == 3: # Goal
            self.reward = 1
            self.done = True
        else:
            self.reward = 0
            self.done = False

        next_state = self.player_position[1] * self.map_size[0] + self.player_position[0]

        return self.reward, next_state, self.done

    def reset_state(self):
        self.player_position = np.array(self.start, dtype=int)
        self.reward, self.done = 0, False