# WIP

# エージェントと環境との相互作用は離散的であり、エピソード的タスク群に分解されること、
# および行動の集合Aと状態の集合Sは有限の要素しか持たず、
# その数は学習開始時に既知であることを仮定する。
# policyとしてはe-greedyを用いる。

import numpy as np

# Q_function is same as one in SARSA.py
class Q_function(object):
    def __init__(self, state_space_size, action_space_size,
                 learning_rate=0.01, discount_rate=0.95, initial_value=1,random_initial_value=True,
                 decay_learning_rate=1):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.initial_value = initial_value
        self.random_initial_value = random_initial_value
        self.decay_learning_rate = decay_learning_rate

        if self.random_initial_value:
            self.q_table = np.random.rand(self.state_space_size, self.action_space_size) * self.initial_value
        else:
            self.q_table = np.ones((self.state_space_size, self.action_space_size), dtype=float64) * self.initial_value

        self.last_state = None
        self.last_action = None
    
    def estimate_q_value(self, state, action=None):
        self.last_state, self.last_action = state, action
        if action is None:
            return self.q_table[state]
        else:
            return self.q_table[state][action]
    
    def update_q_table(self, reward, next_state, next_action,
                       last_state=None, last_action=None):
        last_state = self.last_state if last_state is None else last_state
        last_action = self.last_action if last_action is None else last_action

        delta = reward + self.discount_rate * self.estimate_q_value(next_state, next_action) \
                - self.estimate_q_value(last_state, last_action)
        self.q_table[state][action] = self.q_table[state][action] + self.learning_rate * delta

    def decay_learning_rate_value(self, decay_rate=None):
        decay_rate = self.decay_learning_rate if decay_rate is None else decay_rate
        if 0<=decay_rate<=1:
            self.learning_rate = self.learning_rate * decay_rate

    def save_q_table(self):
        return self.q_table

    def load_q_table(self, q_table):
        self.q_table = q_table

    def reset(self, reset_q_table=True, learning_rate=None, discount_rate=None, decay_learning_rate=None):
        if reset_q_table:
            if self.random_initial_value:
                self.q_table = np.random.rand(self.state_space_size, self.action_space_size) * initial_value
            else:
                self.q_table = np.ones((self.state_space_size, self.action_space_size), dtype=float64) * initial_value
        self.learning_rate = learning_rate if learning_rate is not None else self.learning_rate
        self.discount_rate = discount_rate if discount_rate is not None else self.discount_rate
        self.decay_learning_rate = decay_learning_rate if decay_learning_rate is not None else self.decay_learning_rate

    
class Agent_e_greed(object):
    def __init__(self, value_function, state_space_size, action_space_size,
                 action_count_list=None, 
                 initial_play_count=None, epsilon=0.1, min_choose=1):
        self.value_function = value_function
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.total_play_count = 0
        self.min_choose = min_choose
        self.epsilon = epsilon

        self.last_state = None
        self.last_action = None

        # self.action_count_list[action] = number of [action] is choosed
        if action_count_list is None:
            self.action_count_list = np.zeros(self.action_space_size, dtype=int)
        else:
            self.action_count_list = action_count_list
        if initial_play_count is not None:
            self.total_play_count = initial_play_count
        
    def act_greedy(self, state):
        index_of_less_selected = np.where(self.action_count_list)
        if index_of_less_selected[0].size == 0:
            value_table = self.value_function.estimate_q_value(state)
            max_index = np.where(value_table == value_table.max())
            action = np.random.choice(max_index[0], 1)
        else:
            action = int(np.random.choice(index_of_less_selected, 1))

    def act(self, state, update_last=True):
        if np.random.choice([1, 0], p=[self.epsilon, 1-self.epsilon]):
            action = int(np.random.choice(range(action_space_size)))
        else:
            action = self.act_greedy(state)
        if update_last:
            self.last_state, self.last_action = state, action
        return action
 
    def update(self,reward, next_state, last_state=None, last_action=None):
        last_state = self.last_state if last_state is None else last_state
        last_action = self.last_action if last_action is None else last_action

        self.value_function.update_q_table(reward, next_state, next_action,
                                           last_state, last_action)

    def save_record(self):
        return self.action_count_list, self.total_play_count, self.value_function.save_q_table()
    
    def load_record(self, action_count_list=None, total_play_count=None, q_table=None):
        self.action_count_list = action_count_list if action_count_list is not None else self.action_count_list
        self.total_play_count = total_play_count if total_play_count is not None else self.total_play_count
        if q_table is not None:
            self.value_function.load_q_table(q_table)
    
    def reset_record(self, reset_count=True, reset_last=True, reset_q_function=True,
                     reset_q_table=True, learning_rate=None, discount_rate=None, decay_learning_rate=None):
        if reset_count:
            self.total_play_count = 0
            self.action_count_list = np.zeros(self.action_space_size, dtype=int)
        if reset_last:
            self.last_state = None
            self.last_action = None
        if reset_q_function:
            self.value_function.reset(reset_q_table, learning_rate, discount_rate, decay_learning_rate)
    
