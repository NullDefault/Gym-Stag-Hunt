import time

from gym_stag_hunt.envs import ClassicStagHunt, MarkovStagHunt

ENVS = 'CLASSIC', 'MARKOV'
ENV = 'MARKOV'

if __name__ == "__main__":
    if ENV == 'CLASSIC':
        env = ClassicStagHunt()
    elif ENV == 'MARKOV':
        env = MarkovStagHunt()

    env.reset()
    for i in range(1000):
        obs, rewards, done, info = env.step([env.action_space.sample(), env.action_space.sample()])
        time.sleep(.4)
        if ENV == 'CLASSIC':
            env.render(rewards=rewards)
        else:
            env.render()
    env.close()
    quit()
