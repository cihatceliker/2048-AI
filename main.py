from environment import Environment
from dqn import Agent, Brain
import numpy as np
import math
import pickle
import torch

num_actions = 4
num_iter = 500000
print_interval = 10
save_interval = 100
env = Environment()
agent = Agent(num_actions)
#agent.load("36800") before halving
agent.load("44000")
print(agent.optimizer)
#agent.optimizer = torch.optim.Adam(agent.local_Q.parameters(), 2e-3)
start = agent.start + 1

for s in agent.replay_memory.memory:
    if s[2] > 7:
        print(s[2])
mx = 1
for episode in range(start, num_iter):
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

    agent.eps_start = max(agent.eps_end, agent.eps_decay * agent.eps_start)
    agent.episodes.append(episode)
    agent.scores.append(score)
    agent.durations.append(ep_duration)
    agent.start = episode
    
    if episode % print_interval == 0:
        avg_score = np.mean(agent.scores[max(0, episode-print_interval):(episode+1)])
        avg_duration = np.mean(agent.durations[max(0, episode-print_interval):(episode+1)])
        if episode % save_interval == 0:
            agent.save(str(episode))
        print("Episode: %d - Max Score: %d - Avg. Duration: %d - Avg. Score: %.3f - Epsilon: %.3f" % 
                    (episode, 2**mx, avg_duration, avg_score, agent.eps_start))
        mx = 1
