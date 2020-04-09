import numpy as np
import random

def REWARD(x):
    if x < 9:
        return 0
    return 2.0 ** (x - 10)
DIE = -4

class Environment():

    def __init__(self, row=4, col=4):
        self.row = row
        self.col = col
        self.reset()
        self.moves = {
            0: self.move_up,
            1: self.move_right,
            2: self.move_down,
            3: self.move_left
        }
    
    def reset(self):
        self.board = np.zeros((self.row, self.col), dtype=np.int8)
        self.prev = self.board.copy()
        self.done = False
        self.prev = self.board.copy()
        self.throw()
        return self.process_state()

    def step(self, action):
        self.reward = 0
        self.done = False
        self.moves[action]()
        self.throw()
        mx = np.max(self.board)
        return self.process_state(), self.reward, self.done, 2**mx
        
    def process_state(self):
        max_depth = 14 # 2^13 max
        state = np.zeros((max_depth*2, 4, 4))
        for i in range(4):
            for j in range(4):
                state[self.board[i,j], i, j] = 1
                state[self.prev[i,j]+max_depth, i, j] = 1
        self.prev = self.board.copy()
        return state

    def throw(self):
        empty_tiles = []
        for i in range(self.row):
            for j in range(self.col):
                if self.board[i,j] == 0:
                    empty_tiles.append((i,j))
        if len(empty_tiles) == 0:
            self.done = True
            self.reward = DIE
            return
        add_to = random.choice(empty_tiles)
        if np.random.random() > 0.9:
            self.board[add_to] = 2
        else:
            self.board[add_to] = 1

    def move_right(self):
        for i in range(self.row):
            for j in range(self.col-1,-1,-1):
                if self.board[i, j] != 0:
                    curr = self.board[i, j]
                    for k in reversed(range(j)):
                        if self.board[i, k] == curr:
                            self.board[i, j] += 1
                            self.reward = REWARD(self.board[i, j])
                            self.board[i, k] = 0
                            break
                        elif self.board[i, k] != 0:
                            break
            for j in range(self.col):
                if self.board[i, j] == 0:
                    self.board[i, :j + 1] = [0, *self.board[i, :j]]

    def move_up(self):
        self.board = self.board.T
        self.move_left()
        self.board = self.board.T

    def move_left(self):
        for i in range(self.row):
            for j in range(self.col):
                if self.board[i, j] != 0:
                    curr = self.board[i, j]
                    for k in range(j + 1, self.col):
                        if self.board[i, k] == curr:
                            self.board[i, j] += 1
                            self.reward = REWARD(self.board[i, j])
                            self.board[i, k] = 0
                            break
                        elif self.board[i, k] != 0:
                            break
            for j in range(self.col-1,-1,-1):
                if self.board[i, j] == 0:
                    self.board[i, j:] = [*self.board[i, j + 1:], 0]

    def move_down(self):
        self.board = self.board.T
        self.move_right()
        self.board = self.board.T
