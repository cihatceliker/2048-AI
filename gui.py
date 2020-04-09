from tkinter import Frame, Label, CENTER
import numpy as np
import threading
import time
import sys
from environment import Environment
from agent import Agent, load_agent
import pickle

SIZE = 500
GRID_PADDING = 10

BACKGROUND_COLOR_GAME = "#fff"
BACKGROUND_COLOR_CELL_EMPTY = "#9e948a"
BACKGROUND_COLOR_DICT = {   2:"#eee4da", 4:"#ede0c8", 8:"#f2b179", 16:"#f59563", \
                            32:"#f67c5f", 64:"#f65e3b", 128:"#edcf72", 256:"#edcc61", \
                            512:"#edc850", 1024:"#edc53f", 2048:"#edc22e", 4096:"#1395bd" }
CELL_COLOR_DICT = { 2:"#776e65", 4:"#776e65", 8:"#f9f6f2", 16:"#f9f6f2", \
                    32:"#f9f6f2", 64:"#f9f6f2", 128:"#f9f6f2", 256:"#f9f6f2", \
                    512:"#f9f6f2", 1024:"#f9f6f2", 2048:"#f9f6f2", 4096:"#f9f6f2" }
FONT = ("Verdana", 40, "bold")


class GameGrid(Frame):
    def __init__(self):
        Frame.__init__(self)
        self.env = Environment()
        #self.agent = Agent(4); self.agent = load_agent(sys.argv[1])
        self.history = pickle.load(open(sys.argv[1], mode="rb"))
        self.row = self.env.row
        self.col = self.env.col
        self.grid()
        self.master.title('2048')
        self.grid_cells = []
        self.init_grid()
        #threading.Thread(target=self.watch_play).start()
        threading.Thread(target=self.watch_history).start()
        self.mainloop()

    def watch_play(self):
        while True:
            done = False
            score = 0
            ep_duration = 0
            state = self.env.reset()
            while not done:
                action = self.agent.select_action(state)
                state, reward, done, max_tile = self.env.step(action)
                self.board = self.env.board
                self.update_grid_cells()
                time.sleep(0.1)

    def watch_history(self):
        for i in range(1, len(self.history)):
            self.board = self.history[i]
            self.update_grid_cells()
            time.sleep(0.02)

    def update_grid_cells(self):
        for i in range(self.row):
            for j in range(self.col):
                new_number = 2**int(self.board[i][j])
                if new_number == 1:
                    self.grid_cells[i][j].configure(text="", bg=BACKGROUND_COLOR_CELL_EMPTY)
                else:
                    self.grid_cells[i][j].configure(text=str(new_number), bg=BACKGROUND_COLOR_DICT[new_number], fg=CELL_COLOR_DICT[new_number])
        self.update_idletasks()
        
    def init_grid(self):
        background = Frame(self, bg=BACKGROUND_COLOR_GAME, width=SIZE, height=SIZE)
        background.grid()
        for i in range(self.row):
            grid_row = []
            for j in range(self.col):
                cell = Frame(background, bg=BACKGROUND_COLOR_CELL_EMPTY, width=SIZE/self.col, height=SIZE/self.row)
                cell.grid(row=i, column=j, padx=GRID_PADDING, pady=GRID_PADDING)
                t = Label(master=cell, text="", bg=BACKGROUND_COLOR_CELL_EMPTY, justify=CENTER, font=FONT, width=4, height=2)
                t.grid()
                grid_row.append(t)
            self.grid_cells.append(grid_row)
    

if __name__ == "__main__":
    GameGrid()
