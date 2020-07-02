import gym

# ====== config ======
train = False   # set train to True to use the pre-trained agent, False to train


# ============

name = 'approximateCurvedSheet'
env = gym.make('gym_trussbot:trussbot-v0',      # name of the gym environment, details can be found under ./gym-trussbot
               modelName='8x8x1',
               agentName='actuate',
               criterionName='shape',
               timeStep=0.001
               )                        # initialize the environment
env.criterion.setTarget('bending')        # set the target shape
env.model.g = 0                         # disable gravity

if train:
    env.optimizer.maximize(nSteps=10)   # evolve for 10 generation
    p = env.optimizer.getSurvivor(i=0)  # pick the best gene(shrinkage ratio)
    env.agent.setPolicy(p)              # set the shrinkage ratio
    env.agent.save(name)                # save the policy
else:
    p = env.agent.load(name)

try:
    env.run()       # main loop with visualization
except:
    print('The environment cannot be run.')
