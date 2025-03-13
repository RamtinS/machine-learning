# Reinforcement learning

### What is reinforcement learning?

Reinforcement learning is a type of machine learning where an agent
(like a computer program) learns to make decisions by interacting
with its environment to maximize rewards over time.

When the agent acts, the environment provides feedback, a reward
(+1) or penalty (-1) depending on the action's quality. The agent's
goal is to figure out which actions lead to the highest total
reward by avoiding penalties.

It learns through trial and error. Initially, it may make random
moves and receive penalties, but over time, it remembers which
actions earned rewards and improves its strategy (policy), balancing
trying new things (exploring) with doing things it already knows
work well (exploiting).

For instance, if the agent is a robot learning to walk in a straight
line, it gets rewards for moving forward and penalties for veering
off. Over time, it learns to walk straight to maximize rewards.

### Mountain car problem

The mountain car problem is a classic reinforcement learning
challenge where an underpowered car must escape a valley. 
The car cannot reach the top of the hill in one go, so it must
learn to build momentum by moving back and forth.  

The goal is to reach the flag at the top using as few moves
as possible. The agent gets a small negative reward for each
step to encourage efficiency. Through trial and error, it learns
the best way to swing back and forth until it gains enough
speed to escape the valley.