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
    # the criterion is to minimize the different between a shape and a target shape

    def __init__(self, model, targetName="snake"):
        super().__init__(model)
        self.nSteps = 1000

        self.targetName = targetName
        self.targetMesh = None

    def __call__(self, x):
        if self.targetMesh is None:
            self.setTarget(self.targetName)

        self.env.reset()
        self.env.agent.setPolicy(x)
        for i in range(self.nSteps):
            action = self.env.getAction()
            self.env.step(action)

        (closest_points, distances, triangle_id) = self.targetMesh.nearest.on_surface(self.env.model.v)
        distanceMean = np.mean(np.power(distances, 2), axis=0)
        return -distanceMean

    def setTarget(self, targetName):
        name = os.path.join(rootPath, "data/target/", targetName+".obj")
        try:
            self.targetMesh = trimesh.load(name)
        except:
            print('No such target.')


class CriterionCurvedSheet(Criterion):
    # the criterion is to minimize the difference between the 8x8x1 model and a curved sheet math function
    R = 3

    def __init__(self, model):
        super().__init__(model)
        self.nSteps = 1000
        self.targetV = self.convertToTarget()

    def convertToTarget(self):
        print('The model is reset in CriterionBendingSheet.')
        self.env.model.reset()
        vs = self.env.model.v

        R = CriterionCurvedSheet.R
        d = -0.6
        ps = []
        for v in vs:
            x, y, z = v
            alpha = (x - 4) / R
            xx = 4 + R * np.sin(alpha)
            zz = R + d - R * np.cos(alpha)
            if z == 1:
                vec = np.array([4 - xx, 0, R + d - zz])
                vec = vec / np.sqrt(np.sum(vec ** 2))
                xx = xx + vec[0]
                zz = zz + vec[2]
            ps.append(np.array([xx, y, zz]))
        ps = np.array(ps)
        ps = ps.reshape(-1, 3)
        return ps

    def __call__(self, x):
        self.env.reset()
        self.env.agent.setPolicy(x)
        for i in range(self.nSteps):
            action = self.env.getAction()
            self.env.step(action)

        assert(self.targetV.shape[1] == 3)
        assert(self.env.model.v.shape[1] == 3)
        squaredDistance = np.sum((self.targetV - self.env.model.v) ** 2, axis=1)
        meanSquareDistance = np.mean(squaredDistance)
        return -meanSquareDistance
