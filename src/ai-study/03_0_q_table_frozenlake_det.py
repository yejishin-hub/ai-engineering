import gym
#import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random as pr

# 랜덤하게 이동하라는 의미에서 random argument 구현
def rargmax(vector): # https://gist.github.com/stober/1943451
	""" Argmax that chooses randomly among eligible maximum indices. """
	m = np.amax(vector)
	indices = np.nonzero(vector == m)[0]
	return pr.choice(indices)

register(
	id='FrozenLake-v3',
	entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name':'4x4','is_slippery':False}
)
env = gym.make('FrozenLake-v3')

# Initialize table with all zeros
Q = np.zeros([env.observation_space.n,env.action_space.n]) # 16,4
# Set learning parameters
num_episodes = 2000 # loop

# create lists to contain total rewards and steps per episode
rList = []
for i in range(num_episodes):
	# Reset environment and get first new observation
	state = env.reset() #초기화 후 첫 번째 상태 수집
	rAll = 0
	done = False
	
	# The Q-Table learning algorithm
	while not done: #게임이 끝나는지 (done) 확인
		action = rargmax(Q[state, :]) #random argument
		
		# Get new state and reward from environment
		new_state, reward, done,_ = env.step(action)
		
		# Update Q-Table with new knowlegde using learning rate
		Q[state,action] = reward + np.max(Q[new_state,:])
		
		rAll += reward # reward를 다 합하기
		state = new_state
	
	rList.append(rAll)

print("Success rate: " + str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print("LEFT DOWN RIGHT UP")
print(Q)
plt.bar(range(len(rList)), rList, color="blue")
plt.show()