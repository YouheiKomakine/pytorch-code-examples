# WIP

# エージェントと環境との相互作用は離散的であり、エピソード的タスク群に分解されること、
# および行動の集合Aと状態の集合Sは有限の要素しか持たず、
# その数は学習開始時に既知であることを仮定する。
# policyとしてはe-greedyを用いる。

import numpy as np
from SARSA import Agent_SARSA, Q_table_function, Policy_e_greedy

class Agent_Q_learning(Agent_SARSA):
    def learning_step(self, action, update_flag=True):
        reward, next_state, episode_end_flag = self.observe_state(action)
        if episode_end_flag:
            next_action = -1
        else:
            # choose a' which gives max Q(s',a')
            next_action = self.act(next_state, update_flag, epsilon=0)
        if update_flag:
            self.q_function.update_q_table(reward, next_state, next_action,
                                           self.last_state, self.last_action)
        return reward, next_state, next_action, episode_end_flag
