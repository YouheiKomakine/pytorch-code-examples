# WIP

# エージェントと環境との相互作用は離散的であり、エピソード的タスク群に分解されること、
# および行動の集合Aと状態の集合Sは有限の要素しか持たず、
# その数は学習開始時に既知であることを仮定する。
# また、テーブル型TD(0)アルゴリズムとして実装しており、
# state(i) | i=0~nが一次元的に並べられることを前提としている。
# policyとしてはe-greedyを用いる。

import numpy as np

class Q_table_function(object):
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

    def load_q_table(self, q_table=None):
        if q_table is not None:
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


class Policy_e_greedy(object):
    def __init__(self, state_space_size, action_space_size,
                 action_count_list=None,
                 initial_play_count=None, epsilon=0.1, min_choose=1):
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

    def choose_act_greedy(self, state, value_table):
        index_of_less_selected = np.where(self.action_count_list)
        if index_of_less_selected[0].size == 0:
            max_index = np.where(value_table == value_table.max())
            action = np.random.choice(max_index[0], 1)
        else:
            action = int(np.random.choice(index_of_less_selected, 1))

    def choose_act(self, state, update_flag=True):
        if np.random.choice([1, 0], p=[self.epsilon, 1-self.epsilon]):
            action = int(np.random.choice(range(action_space_size)))
        else:
            action = self.choose_act_greedy(state)

        self.last_state, self.last_action = state, action
        if update_flag:
            self.total_play_count += 1
            self.action_count_list[action] += 1
        
        return action

    def save_record(self):
        return self.action_count_list, self.total_play_count
    
    def load_record(self, action_count_list=None, total_play_count=None):
        self.action_count_list = action_count_list if action_count_list is not None else self.action_count_list
        self.total_play_count = total_play_count if total_play_count is not None else self.total_play_count
    
    def reset_record(self, reset_count=True, reset_last=True):
        if reset_count:
            self.total_play_count = 0
            self.action_count_list = np.zeros(self.action_space_size, dtype=int)
        if reset_last:
            self.last_state = None
            self.last_action = None


class Agent_SARSA(object):
    def __init__(self, state_space_size, action_space_size,
                 state_function, q_function=None, policy_function=None,
                 learning_rate=0.01, discount_rate=0.95, 
                 initial_value=1, random_initial_value=True, decay_learning_rate=1,
                 action_count_list=None, initial_play_count=None, epsilon=0.1, min_choose=1):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.initial_value = initial_value
        self.random_initial_value = random_initial_value
        self.decay_learning_rate = decay_learning_rate
        self.action_count_list = action_count_list
        self.initial_play_count = initial_play_count
        self.epsilon = epsilon
        self.min_choose = min_choose

        self.total_play_count = 0
        self.last_state = None
        self.last_action = None

        self.state_function = self.state_function

        if q_function is None:
            self.q_function = Q_table_function(self.state_space_size, self.action_space_size,
                                               self.learning_rate, self.discount_rate, self.initial_value,
                                               self.random_initial_value, self.decay_learning_rate)
        else:
            self.q_function = q_function

        if policy_function is None:
            self.policy_function = Policy_e_greedy(self.state_space_size, self.action_space_size,
                                                   self.action_count_list, self.initial_play_count,
                                                   self.epsilon, self.min_choose)
        else:
            self.policy_function = policy_function


    def act(self, state, update_flag=True):
        action = self.policy_function.choose_act(state, update_flag=update_flag)
        self.last_state, self.last_action = state, action
        return action

    def observe_state(action=None):
        # state_function must return (reward, next_state, episode_end_flag)
        return self.state_function.return_next(action)


    def learning_step(self, action, update_flag=True):
        reward, next_state, episode_end_flag = self.observe_state(action)
        if episode_end_flag:
            next_action = -1
        else:
            next_action = self.act(next_state, update_flag)
        if update_flag:
            self.q_function.update_q_table(reward, next_state, next_action,
                                           self.last_state, self.last_action)
        return reward, next_state, next_action, episode_end_flag

    def learn(self, episode_num, maximum_trial_per_episode=1000, 
              save_flag=True, update_flag=True, reset_when_finished=False):
        reward = 0
        episode_end_flag = False
        trial_count = 0

        # episode loop
        for episode in range(episode_num):
            # choose initial action
            self.state_function.reset_state()
            state = self.observe_state()
            action = self.act(state, update_flag)
            
            while episode_end_flag==False and (trial_count < maximum_trial_per_episode):
                reward, state, action, episode_end_flag = self.learning_step(action)
                trial_count += 1
            else:
                reward, episode_end_flag, trial_count = 0, False, 0

        save_data = self.save() if save_flag else None
        if reset_when_finished:
            self.reset()

        return save_data

    def demo_play(self, episode_num=1, maximum_trial_per_episode=1000):
        self.learn(episode_num, maximum_trial_per_episode, safe_flag=False, update_flag=False)


    def save(self):
        # returns (q_table, (action_count_list, total_play_count))
        q_save_data = self.q_function.save_q_table()
        policy_save_data = self.policy_function.save_record()
        return q_save_data, policy_save_data

    def load(self, q_save_data=None, policy_save_data=None):
        if q_save_data is not None:
            self.q_function.load_q_table(q_table)
        if policy_save_data is not None:
            self.policy_function.load_record(self, action_count_list=policy_save_data[0], 
                                             total_play_count=policy_save_data[1])

    def reset(self, reset_q_table=True, reset_count=True, reset_last=True,
              learning_rate=None, discount_rate=None,decay_learning_rate=None):
        self.q_function.reset(reset_q_table, learning_rate, discount_rate, decay_learning_rate)
        self.policy_function.reset_record(reset_count, reset_last)




                


