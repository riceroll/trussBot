import os
import numpy as np
import trimesh
rootPath = os.path.split(os.path.realpath(__file__))[0]


class Criterion(object):
    # the superclass of the criterion function for evaluating the performance of the agent

    def __init__(self, env=None):
        self.env = env

    def __call__(self, x):
        raise NotImplementedError('Please define a reward function.')


class CriterionMoveForward(Criterion):
    # the criterion is to move forward along x as fast as possible and do not deviate along y axis

    def __init__(self, model):
        super().__init__(model)
        self.nSteps = 500

    def __call__(self, x):
        self.env.reset()
        self.env.agent.setPolicy(x)
        c0 = self.env.model.getCentroid()
        for i in range(self.nSteps):
            action = self.env.getAction()
            self.env.step(action)
        c1 = self.env.model.getCentroid()

        dx = c1[0] - c0[0]
        dy = np.abs(c1[1] - c0[1])

        score = 0
        score += dx * 5
        score -= dy

        return score


class CriterionShape(Criterion):
    # the criterion is to change the shape to approximate the target shape

    def __init__(self, model, targetName="snake"):
        super().__init__(model)
        self.nSteps = 1000

        self.targetMesh = None
        self.setTarget(targetName)

    def __call__(self, x):
        self.env.reset()
        self.env.agent.setPolicy(x)
        for i in range(self.nSteps):
            action = self.env.getAction()
            self.env.step(action)

        (closest_points, distances, triangle_id) = self.targetMesh.nearest.on_surface(self.env.model.v)
        distanceMean = np.mean(distances, axis=0)
        return -distanceMean

    def setTarget(self, targetName):
        name = os.path.join("./data/target/", targetName+".obj")
        try:
            self.targetMesh = trimesh.load(name)
        except:
            print('No such target.')
