import os
import numpy as np
rootPath = os.path.split(os.path.realpath(__file__))[0]


class Agent(object):
    # superclass of the control agent, which gives command(shrinkage ratio) to the truss robot

    def __init__(self, env=None):
        self.env = env
        self.policy = None

    def reset(self):
        raise NotImplementedError('Please implement nextAction function.')

    def load(self, policyName):
        name = os.path.join('./data/agent/', policyName+'.npy')
        self.policy = np.load(name, allow_pickle=True)
        print(self.policy)

    def save(self, policyName):
        name = os.path.join('./data/agent/', policyName)
        np.save(name, self.policy)

    def setPolicy(self, policy=None):
        raise NotImplementedError('Please implement nextAction function.')

    def nextAction(self):
        raise NotImplementedError('Please implement nextAction function.')


class AgentBinary(Agent):
    # The agent that switches ON / OFF of the trussbot

    def __init__(self, env=None):
        super().__init__(env)
        self.nStepsOn = 60
        self.nStepsOff = 60
        self.actionOff = self.env.model.actionSpace.low
        self.actionOn = None    # the shrinkage ratio when actuated

        self.nSteps = None

        self.reset()

    def reset(self):
        self.nSteps = 0
        self.actionOn = self.env.model.actionSpace.high  # the action to be optimized

    def load(self, policyName):
        super().load(policyName)
        assert(self.env.model.actionSpace.contains(self.policy))
        self.actionOn = self.policy

    def setPolicy(self, policy=None):
        assert(self.env.model.actionSpace.contains(policy))
        self.policy = policy
        self.actionOn = self.policy

    def nextAction(self):
        n = self.nSteps % (self.nStepsOn + self.nStepsOff)
        self.nSteps += 1

        if n < self.nStepsOn:
            return self.actionOn
        else:
            return self.actionOff


class AgentActuate(Agent):
    # The agent that actuates the trussbot once, for shape approximation

    def __init__(self, env=None):
        super().__init__(env)
        self.actionOn = None        # the shrinkage ratio when actuated

    def reset(self):
        self.actionOn = self.env.model.actionSpace.high

    def load(self, policyName):
        super().load(policyName)
        assert(self.env.model.actionSpace.contains(self.policy))
        self.actionOn = self.policy

    def setPolicy(self, policy=None):
        assert (self.env.model.actionSpace.contains(policy))
        self.policy = policy
        self.actionOn = self.policy

    def nextAction(self):
        return self.actionOn
