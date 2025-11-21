"""
FrozenLake solver using Q-table
https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0#.pjz9g59ap
"""

import time

import gym
import numpy as np

#ModuleNotFoundError: No module named 'utils'
#import utils.prints as print_utils
import matplotlib.pyplot as plt

N_ACTIONS = 4
N_STATES = 16

#LEARNING_RATE = .5
#DISCOUNT_RATE = .98

#N_EPISODES = 2000

def main():
    """Main"""
    ## v0 Deprecated (Not Working) 
    #frozone_lake_env = gym.make("FrozenLake-v0")
    frozone_lake_env = gym.make("FrozenLake-v1")

    # Initialize table with all zeros
    Q = np.zeros([N_STATES, N_ACTIONS])

    # Set learning parameters
    learning_rate = .85
    dis = .99
    num_episodes = 2000

    # create lists to contain total rewards and steps per episode
    #rewards = []
    rList = []
    for i in range(num_episodes):
        # Reset environment and get first new observation
        state = frozone_lake_env.reset()
        #episode_reward = 0
        rAll = 0
        done = False

        # The Q-Table learning algorithm
        while not done:
            # Choose an action by greedily (with noise) picking from Q table
            noise = np.random.randn(1, frozone_lake_env.action_space.n) / (i + 1)
            action = np.argmax(Q[state, :] + noise)

            # Get new state and reward from environment
            new_state, reward, done, _ = frozone_lake_env.step(action)

            #reward = -1 if done and reward < 1 else reward

            # Update Q-Table with new knowledge using learning rate
            Q[state, action] = (1 - learning_rate) * Q[state, action] \
                + learning_rate * (reward + dis * np.max(Q[new_state, :]))

            #episode_reward += reward
            rAll += reward
            state = new_state

        rList.append(rAll)

    print("Score over time: " + str(sum(rList) / num_episodes))
    print("Final Q-Table Values")
    print(Q)
    plt.bar(range(len(rList)), rList, color="blue")
    plt.show()




if __name__ == '__main__':
    main()