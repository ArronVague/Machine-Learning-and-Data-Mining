import numpy as np
import pandas as pd
import time
import sys
import tkinter as tk

pixel_unit = 80  # 像素
maze_height = 6  # 迷宫高度
maze_weight = 6  # 迷宫宽度


class Grid_Paint(tk.Tk, object):
    def __init__(self):
        super(Grid_Paint, self).__init__()
        # 初始化动作指令
        self.action_space = ["u", "d", "l", "r"]
        self.n_actions = len(self.action_space)

        self.title("Grid_Paint")
        # 设置窗口的宽和高
        self.geometry(
            "{0}x{1}".format(maze_height * pixel_unit, maze_weight * pixel_unit)
        )
        # 调用构建迷宫函数来搭建迷宫
        self._build_maze()

    # 创造黑色的正方形方块（障碍）
    # 输入分别为【一个网格的相对中心位置】，【障碍的横坐标】，【障碍的纵坐标】
    def creat_barrier(self, origin, x, y):
        # 计算出方块的中心位置
        center = origin + np.array([pixel_unit * x, pixel_unit * y])
        # 以该中心位置向四周进行黑色填充生成黑色方块
        self.barrier = self.canvas.create_rectangle(
            center[0] - 25, center[1] - 25, center[0] + 25, center[1] + 25, fill="black"
        )
        return self.barrier

    # 构建迷宫
    def _build_maze(self):
        # 画出白色背景
        self.canvas = tk.Canvas(
            self,
            bg="white",
            height=maze_height * pixel_unit,
            width=maze_weight * pixel_unit,
        )

        # 通过画线来构建网格
        for c in range(0, maze_weight * pixel_unit, pixel_unit):
            x0, y0, x1, y1 = c, 0, c, maze_height * pixel_unit
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, maze_height * pixel_unit, pixel_unit):
            x0, y0, x1, y1 = 0, r, maze_weight * pixel_unit, r
            self.canvas.create_line(x0, y0, x1, y1)

        # 每个网格的相对中心位置
        origin = np.array([int(0.5 * pixel_unit), int(0.5 * pixel_unit)])
        # print(origin)

        # 障碍
        self.barrier1 = self.creat_barrier(origin, 0, 1)

        # 陷阱
        red_center = origin + np.array([pixel_unit * 3, pixel_unit * 3])
        # print(red_center)
        self.oval = self.canvas.create_oval(
            red_center[0] - 25,
            red_center[1] - 25,
            red_center[0] + 25,
            red_center[1] + 25,
            fill="red",
        )

        # 起点
        self.rect = self.canvas.create_rectangle(
            origin[0] - 25, origin[1] - 25, origin[0] + 25, origin[1] + 25, fill="green"
        )

        # 迷宫中的实时位置

        # 终点
        pink_center = origin + np.array([pixel_unit * 5, pixel_unit * 5])
        # print(pink_center)
        self.rect = self.canvas.create_rectangle(
            pink_center[0] - 25,
            pink_center[1] - 25,
            pink_center[0] + 25,
            pink_center[1] + 25,
            fill="pink",
        )

        # 打包所有元素
        self.canvas.pack()

    def reset(self):
        self.canvas.delete(self.rect)
        origin = np.array([int(0.5 * pixel_unit), int(0.5 * pixel_unit)])
        self.rect = self.canvas.create_rectangle(
            origin[0] - 25, origin[1] - 25, origin[0] + 25, origin[1] + 25, fill="green"
        )
        return self.canvas.coords(self.rect)

        # 玩家移动，输入为移动指令

    def step(self, action):
        # 记录当前的state，也就是玩家现在的位置，s是一个1x4的数组，分别代表其左上角像素位置，右上角像素位置，左下角像素位置，右下角像素位置
        s = self.canvas.coords(self.rect)

    #     #向上
        if action == 0:
            if s[]

    #     #向下

    #     #向右

    #     #向左

    #     #第一个参数是移动目标，第二个参数是到左上角的水平距离，第三个参数是距左上角的垂直距离。
    #     self.canvas.move(

    #     #移动后的位置，也就是下一个state
    #     s_ =

    #     #reward判断
    #     #如果碰到了陷阱，游戏结束

    #     #如果碰到了障碍，游戏结束

    #     #如果到达了终点，则奖励为50，且游戏结束

    #     #如果都没有碰到，则游戏继续，但是奖励为-1，代表移动的步数，否则无法去寻找最低步数

    #     #返回state s在经过action之后的下一个state s_，获得的奖励 reward，以及此时游戏状态 status
    #     return s_, reward, status
    def render(self):
        time.sleep(0.1)
        self.update()


grid = Grid_Paint()
grid.mainloop()
