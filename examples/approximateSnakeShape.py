import gym

# ====== config ======
train = False   # set train to True to use the pre-trained agent, False to train

# ============

name = 'approximateSnakeShape'
env = gym.make('gym_trussbot:trussbot-v0',      # name of the gym environment, details can be found under ./gym-trussbot
               modelName='column',
               agentName='actuate',
               criterionName='shape',
               timeStep=0.002
               )                        # initialize the environment
env.criterion.setTarget('snake')        # set the target shape

if train:
    env.optimizer.maximize(nSteps=10)   # evolve for 10 generation
    p = env.optimizer.getSurvivor(i=0)  # pick the best gene(shrinkage ratio)
    env.agent.setPolicy(p)              # set the shrinkage ratio
    env.agent.save(name)                # save the policy
else:
    p = env.agent.load(name)

env.run()       # main loop with visualization
