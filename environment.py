import numpy as np
import random
import sys

REWARD = lambda x: x**2/64
DIE = -10

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
        self.throw()
        self.done = False
        return self.board

    def step(self, action):
        self.reward = 0
        self.done = False
        self.info = ""
        self.moves[action]()
        self.throw()
        return self.board, self.reward, self.done, self.info

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
                            self.reward = REWARD(curr)
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
                            self.reward = REWARD(curr)
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