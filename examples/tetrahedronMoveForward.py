import gym

# ====== config ======
train = True   # set train to True to use the pre-trained agent, False to train

# ============

name = 'tetrahedronMoveForward'
env = gym.make('gym_trussbot:trussbot-v0',  # name of the gym environment, details can be found under ./gym-trussbot
               modelName='tet',
               agentName='binary',
               criterionName='moveForward'
               )        # initialize the environment

if train:
    env.optimizer.maximize(nSteps=5)        # evolve for 10 generation
    p = env.optimizer.getSurvivor(i=0)      # pick the best gene(shrinkage ratio)
    env.agent.setPolicy(p)                  # set the shrinkage ratio
    env.agent.save(name)                    # save the policy
else:
    p = env.agent.load(name)

env.run()
