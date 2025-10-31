import gymnasium as gym
from gym.envs.registration import register
#import sys,tty,termios # windows not working
import msvcrt
import readchar
import colorama as cr

# Register FrozenLake with is_slippery False
#register(
#    id='FrozenLake-v3',
#    entry_point='gym.envs.toy_text:FrozenLakeEnv',
#    kwargs={'map_name' : '4x4', 'is_slippery': False}
#)

# Environment version `v3` for environment `FrozenLake` doesn't exist
#env = gym.make('FrozenLake-v3',render_mode="human").env
#env = gym.make('FrozenLake-v1',render_mode="human").env
#gym.Env.registration.register(id='FrozenLake-v1',entry_point='gym.envs.toy_text:FrozenLakeEnv',kwargs={'map_name':'4x4','is_slippery':False})
cr.init(autoreset=True)
env = gym.make('FrozenLake-v1',render_mode="human").env
env.reset()
env.render() # Show the initial board

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

arrow_keys = {
    '\x1b[A' : UP,
    '\x1b[B' : DOWN,
    '\x1b[C' : RIGHT,
    '\x1b[D' : LEFT
}

while True:
    # Choose an action from keyboard
    #key = inkey() # [NameError: name 'inkey' is not defined]
    key = readchar.readkey()
    if key not in arrow_keys.keys():
        print("Game aborted!")
        break

    action = arrow_keys[key]
    state, reward, done, info = env.step(action)
    env.render() # Show the board after action
    print("State: ", state, "Action: ", action, "Reward: ", reward, "Info: ", info)

    if done:
        print("Finished with reward", reward)
        break