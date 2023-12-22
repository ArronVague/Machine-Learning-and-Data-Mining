import tkinter as tk
import numpy as np
import time
import pandas as pd
from Grid_Paint import Grid_Paint

grid = Grid_Paint()
grid.mainloop()


class QLearning:
    # 初始化，包括action列表，学习率，衰弱率，贪婪程度，以及Q-table
    def __init__(self, actions, alpha=0.9, gamma=0.9, epsilon=0.9):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    # 选择action，输入参数为当前的state，输出的为在当前state的下一步action
    def choose_action(self, s):
        # 首先判断该state在Q-table中是否存在，如果不存在则加入到Q-table
        # action 选择
        if str(s) not in self.q_table.index:
            # 将state加入Q表中
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions), index=self.q_table.columns, name=str(s)
                )
            )

        # 贪婪模式
        # 生成随机数a
        num = np.random.uniform()

        # 挑选最佳的action
        if num < self.epsilon:
            # 从Q表中选择value值最大的action
            actions = self.q_table.loc[str(s), :]

            # 如果有多个action的value值都是最大，那就从这些中随机挑选
            if len(actions[actions == np.max(actions)]) > 1:
                a = np.random.choice(actions[actions == np.max(actions)].index)
            else:
                a = actions.idxmax()
        else:
            # 非贪婪，探索模式
            # 随机挑选action
            a = np.random.choice(self.q_table.columns)

        return a

    # 学习以此不断更新Q-table，输入参数为一个state，做出的动作a，收获的奖励r，下一个state s_
    def learn(self, s, a, r, s_):
        # 首先判断下一个state s_在Q-table中是否存在，如果不存在则加入到Q-table
        if str(s_) not in self.q_table.index:
            # 将state加入Q表中
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions), index=self.q_table.columns, name=str(s_)
                )
            )

        # 先从Q-table中查询到Q(s,a)
        q_predict = self.q_table.loc[str(s), a]

        # 如果下一个state代表游戏结束，则不需要找下一个state s_能获得得最大value值

        # 如果下一个state游戏继续，则首先找到下一个state s_能获得的最大value值

    # 检查state是否存在，输入为要检查的state
    # def check_state_exist(self, s):
