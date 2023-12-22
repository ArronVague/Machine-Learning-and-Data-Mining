import tkinter as tk
import numpy as np
import time

pixel_unit = 80  # 像素
maze_height = 6  # 迷宫高度
maze_weight = 6  # 迷宫宽度


class Grid_Paint(tk.Tk, object):
    def __init__(self):
        super(Grid_Paint, self).__init__()
        # 初始化动作指令
        self.action_space = ["u", "d", "l", "r"]
        # 方便之后用0，1，2，3代表上下左右
        self.n_actions = len(self.action_space)
        self.title("Grid_Paint")
        # 设置窗口的宽和高
        self.geometry(
            "{0}x{1}".format(maze_height * pixel_unit, maze_weight * pixel_unit)
        )
        # 调用构建迷宫函数来搭建迷宫
        self._build_maze()

    # 创造黑色的正方形方块
    # 输入分别为【一个网格的相对中心位置】，【障碍的横坐标】，【障碍的纵坐标】
    def create_barrier(self, origin, x, y):
        # 计算出方块的中心位置
        center = origin + np.array([pixel_unit * x, pixel_unit * y])
        # 以该中心位置向四周进行黑色填充生成黑色方块
        barrier = self.canvas.create_rectangle(
            center[0] - 25, center[1] - 25, center[0] + 25, center[1] + 25, fill="black"
        )
        return barrier

    def create_trap(self, origin, x, y):
        center = origin + np.array([pixel_unit * x, pixel_unit * y])
        # 以该中心位置向四周进行黑色填充生成红色方块
        trap = self.canvas.create_rectangle(
            center[0] - 25, center[1] - 25, center[0] + 25, center[1] + 25, fill="red"
        )
        return trap

    def create_goal(self, origin, x, y):
        center = origin + np.array([pixel_unit * x, pixel_unit * y])
        # 以该中心位置向四周进行黑色填充生成粉色方块
        goal = self.canvas.create_rectangle(
            center[0] - 25, center[1] - 25, center[0] + 25, center[1] + 25, fill="pink"
        )
        return goal

    def create_rect(self, origin, x, y):
        center = origin + np.array([pixel_unit * x, pixel_unit * y])
        # 以该中心位置向四周进行黑色填充生成蓝绿色方块
        rect = self.canvas.create_rectangle(
            center[0] - 25, center[1] - 25, center[0] + 25, center[1] + 25, fill="green"
        )
        return rect

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

        # 创造障碍
        self.barrier1 = self.create_barrier(origin, 1, 1)
        self.barrier2 = self.create_barrier(origin, 2, 2)

        # 创造陷阱
        self.trap = self.create_trap(origin, 3, 3)

        # 迷宫中的实时位置
        self.rect = self.create_rect(origin, 0, 0)

        # 创造终点
        self.goal = self.create_goal(origin, 5, 5)

        # 打包所有元素
        self.canvas.pack()

    # 重置玩家位置
    def reset(self):
        self.canvas.delete(self.rect)
        origin = np.array([int(0.5 * pixel_unit), int(0.5 * pixel_unit)])
        self.rect = self.create_rect(origin, 0, 0)
        return self.canvas.coords(self.rect)

    # 玩家移动，输入为移动指令
    def step(self, action):
        s = self.canvas.coords(self.rect)

        # 向上
        if action == 0:
            self.canvas.move(self.rect, 0, -pixel_unit)
        # 向下
        elif action == 1:
            self.canvas.move(self.rect, 0, pixel_unit)
        # 向右
        elif action == 2:
            self.canvas.move(self.rect, pixel_unit, 0)
        # 向左
        elif action == 3:
            self.canvas.move(self.rect, -pixel_unit, 0)

        s_ = self.canvas.coords(self.rect)

        # reward 判断
        reward = 0
        # 如果碰到了陷阱，游戏结束
        if s_ == self.canvas.coords(self.trap):
            reward = -50
            status = "game over"
        # 如果碰到了障碍，游戏结束
        elif s_ in [
            self.canvas.coords(self.barrier1),
            self.canvas.coords(self.barrier2),
        ]:
            reward = -50
            status = "game over"
        # 如果到达了终点，则奖励为50，且游戏结束
        elif s_ == self.canvas.coords(self.goal):
            reward = 50
            status = "game over"
        # 如果都没有碰到，则游戏继续，但是奖励为-1，代表移动的步数
        else:
            reward = -1
            status = "continue"

        return s_, reward, status

    def render(self):
        time.sleep(0.1)
        self.update()


# env = Grid_Paint()
# env.mainloop()
