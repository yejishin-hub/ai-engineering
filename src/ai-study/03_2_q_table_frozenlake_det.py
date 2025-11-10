import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register

register(
	id='FrozenLake-v3',
	entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name':'4x4','is_slippery':False}
)
env = gym.make('FrozenLake-v3')

# Initialize table with all zeros
Q = np.zeros([env.observation_space.n,env.action_space.n])
# Discount factor
dis = .9
num_episodes = 2000

# create lists to contain total rewards and step per episode
rList = []

for i in range(num_episodes):
    # Reset environment and get first new observation
    state = env.reset()
    rAll = 0
    done = False

    e = 1. / ((i // 100) + 1) # Python 2&3

    # The Q-Table learning algorithm
    while not done:
        # Choose an action by e greedy
        if np.random.rand(1) < e: # 앞에서 계산한 e값보다 랜덤값이 작으면
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :]) # 우리가 가지고 있는 가장 좋은 방법

        # Get new state and reward from environment
        new_state, reward, done,_ = env.step(action)

        # Update Q-Table with new knowledge using decay rate
        Q[state,action] = reward + dis * np.max(Q[new_state,:])

        rAll += reward
        state = new_state
    
    rList.append(rAll)

print("Success rate: " + str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print(Q)
plt.bar(range(len(rList)), rList, color="blue")
plt.show()