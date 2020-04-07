from environment import Environment
from agent import Agent, load_agent
import numpy as np
import math
import pickle
import torch
import sys

num_actions = 4
num_iter = 5000000000
print_interval = 10
save_interval = 200
env = Environment()

agent = Agent(num_actions) if len(sys.argv) == 1 else load_agent(sys.argv[1])
agent.optimizer = torch.optim.Adam(agent.local_Q.parameters(), 5e-4)


mx = 0
for episode in range(agent.start, num_iter):
    done = False
    score = 0
    ep_duration = 0
    state = env.reset()
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, max_tile = env.step(action)
        agent.store_experience(state, action, reward, next_state, 1-done)
        state = next_state
        score += reward
        ep_duration += 1
    mx = max(max_tile, mx)
    agent.learn()
    agent.episodes.append(episode)
    agent.scores.append(score)
    agent.durations.append(ep_duration)
    agent.start = episode
    
    if episode % print_interval == 0:
        avg_score = np.mean(agent.scores[max(0, episode-print_interval):(episode+1)])
        avg_duration = np.mean(agent.durations[max(0, episode-print_interval):(episode+1)])
        if episode % save_interval == 0:
            agent.start = episode + 1
            agent.save(str(episode))
        print("Episode: %d - Max Score: %d - Avg. Duration: %d - Avg. Score: %.3f - Epsilon: %.3f" % 
                    (episode, mx, avg_duration, avg_score, agent.eps_start))
        mx = 0
