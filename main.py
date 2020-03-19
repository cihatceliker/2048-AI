from environment import Environment
from dqn import Agent, Brain
import numpy as np
import math
import pickle
import torch

max_pow = 16
num_actions = 4
row = 4
col = 4
num_iter = 5000
env = Environment(row, col)
#agent = Agent(Brain(max_pow, num_actions), Brain(max_pow, num_actions), num_actions)
#start = 1

def process_state(obs):
    state = np.zeros((max_pow, row, col))
    for i in range(row):
        for j in range(col):
            if obs[i,j] != 0:
                state[obs[i,j]-1, i, j] = 1
    return state
    
pickle_in = open("1200.two","rb"); agent = pickle.load(pickle_in)
start = agent.episodes[-1] + 1


agent.eps_end = 0.01

for episode in range(start, num_iter):
    state = process_state(env.reset())
    done = False
    score = 0
    while not done:
        action = agent.select_action(state)
        new_obs, reward, done, _ = env.step(action)
        next_state = process_state(new_obs)
        agent.store_experience(state, action, reward, next_state, 1-done)
        state = next_state
        agent.learn()
    score = 2**np.max(env.board)
    
    agent.eps_start = max(agent.eps_end, agent.eps_decay * agent.eps_start)

    agent.episodes.append(episode)
    agent.scores.append(score)
    

    if episode % 10 == 0:
        avg_score = np.mean(agent.scores[max(0, episode-10):(episode+1)])
        print('episode: ', episode,'score: %.6f' % score, ' average score %.3f' % avg_score, "epsilon",agent.eps_start)
        if avg_score >= 300 or episode % 50 == 0:
            pickle_out = open(str(episode)+".two","wb")
            pickle.dump(agent, pickle_out)
            pickle_out.close()
            print("weights are safe for ", episode)
    else: print('episode: ', episode,'score: %.6f' % score)