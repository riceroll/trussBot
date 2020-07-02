import gym

from model import Model
from viewer import Viewer
from agent import AgentBinary, AgentActuate
from criterion import CriterionMoveForward, CriterionShape, CriterionCurvedSheet
from optimizer import EvolutionAlgorithm


class Env(gym.Env):
    # environment class that wraps all the components
    # including simulation model, agent, criterion function and optimizer(evolution)

    def __init__(self, modelName='tet', agentName='binary', criterionName='moveForward', timeStep=None):
        # input:    modelName: file name of the geometry model, under ./data/model
        #           agentName: label of the control agent, check the dict in Env.setAgent()
        #           criterionName: label of the criterion function, check the dict in Env.setCriterion
        #           timeStep: simulation timeSteps, if the model blows up, decrease this value

        self.model = Model(modelName)                       # simulation model
        self.agent = self.setAgent(agentName)               # agent controlling the model(contraction ratio)
        self.criterion = self.setCriterion(criterionName)   # criterion function to evaluate the agent
        self.optimizer = EvolutionAlgorithm(self)           # evolution optimizer to optimize the agent

        # initial setting
        self.model.rough()              # set the ground to be non-directionally rough
        self.model.setShrinkage(0.3)    # set all the shrinkage to be 0.3
        self.model.setPressure(1)       # set the pressure to be max
        if timeStep is not None:
            self.model.h = timeStep     # simulation time step

    def setAgent(self, agentName="binary"):
        agentDict = {
            'binary': AgentBinary,
            'actuate': AgentActuate
        }
        assert(agentName in agentDict)
        self.agent = agentDict[agentName](self)
        return self.agent

    def setCriterion(self, criterionName="moveForward"):
        criterionDict = {
            'moveForward': CriterionMoveForward,
            'shape': CriterionShape,
            'curvedSheet': CriterionCurvedSheet
        }
        assert(criterionName in criterionDict)
        self.criterion = criterionDict[criterionName](self)
        return self.criterion

    def getAction(self):
        return self.agent.nextAction()

    def step(self, action=None):
        if not(action is None):
            self.model.setShrinkage(action)
        self.model.step()

    def render(self, mode='human', actor=None):
        # visualize the current step of the simulation
        viewer = Viewer(self)
        viewer.reset()
        viewer.drawAll()
        # self.viewer.registerAnimationCallback()
        viewer.registerKeyCallback()
        viewer.run()

    def run(self):
        # visualize the simulation dynamically
        viewer = Viewer(self)
        viewer.reset()
        viewer.drawAll()
        viewer.registerAnimationCallback()
        viewer.registerKeyCallback()
        viewer.run()

    def reset(self):
        self.model.reset()
        self.agent.reset()

    def close(self):
        del self.model
