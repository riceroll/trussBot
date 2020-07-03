import gym
import argparse


# ====== config ======
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--train", action="store_true",
                    help="Set to True to use test, False to train.")
parser.add_argument("-p", "--preTrained", action="store_true",
                    help="Set to True to train from the pre-trained agent, False from scratch.")
args = parser.parse_args()

train = args.train
preTrained = args.preTrained
# ============

name = 'approximateCurvedSheetLarge'
env = gym.make('gym_trussbot:trussbot-v0',      # name of the gym environment, details can be found under ./gym-trussbot
               modelName='8x8x1',
               agentName='actuate',
               criterionName='curvedSheet',
               timeStep=0.0005
               )                        # initialize the environment

# setting
env.model.g = 0                         # disable gravity
env.model.sMax = 0.5                    # increase shrinkage ratio
env.model.reset()
env.optimizer.nStages = 2
env.optimizer.reset()

# env.optimizer.nSteps = 1
env.optimizer.nPop = 40
env.optimizer.pbMut = 0.2
env.optimizer.pbMutDig = 0.4

if train:
    if preTrained:
        env.optimizer.load(name)
    env.optimizer.maximize(nSteps=50)   # evolve for 10 generation
    p = env.optimizer.getSurvivor(i=0)  # pick the best gene(shrinkage ratio)
    env.agent.setPolicy(p)              # set the shrinkage ratio
    env.agent.save(name)                # save the policy
else:
    p = env.agent.load(name)

try:
    env.run()       # main loop with visualization
except:
    print('The environment cannot be run.')
