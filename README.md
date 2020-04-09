# 2048-AI

### 2048 AI using DQN with CNN, without external information. Reward function is 'if x > 256: x / 512' and the death reward is -4.

# 
Input has a shape of (28,20,10). Each tile is encoded along with the depth of 14 depending on which power of 2 it is. Empty tile is on channel 0, 2 on channel 1, 4 on channel 2, etc. So the maximum number of suppported is 2^13. The last 2 timesteps are stacked together to have some kind of motion info.

# 
A 'cherry-picked' gameplay(Up to 4096) after a night of training. As long as it keeps the biggest tile on a corner, it does a good job. In every 2 or 3 games, it's able to achieve 2048. Some frames may look skipped in the gif due to a recording issue.

![alt text](/4096.gif)
# 