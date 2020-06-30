import os
import json
import numpy as np
from gym import spaces
rootPath = os.path.split(os.path.realpath(__file__))[0]


class Model(object):
    def __init__(self, modelName='tet'):
        # input: modelName: the file name of the geometry model, under ./data/model

        # ======= variable =======
        self.v = None       # vertices locations    [nv x 3]
        self.e = None       # edge                  [ne x 2] int
        self.vel = None     # velocity              [nv x 3]
        self.FEdge = None  # F_{edge}               [ne x 1], positive: contracting

        self.s = None       # shrinkage ratio       [ne x 1]
        self.pressure = None   # normalized pressure 0 - 1 float, corresponding to actuation progress
        self.l = None       # current lengths       [ne x 1]
        self.lInit = None   # initial lengths       [ne x 1]
        self.l0 = None      # original length after shrinking: target length [ne x 1]

        self.Fs = None          # F_{spring} [nv x 3]
        self.Fg = None          # F_{gravity}   [nv x 3]  always downwards
        self.maskContact = None     # whether the vertex contacting ground or not, [nv x 1] bool
        self.Fn = None          # F_{normal}    [nv x 3]  always upwards
        self.F = None           # F_{sum}   [nv x 3]

        self.nSteps = 0
        self.time = 0

        # ======= constant =======
        self.m = 1
        self.mu = 1.0           # friction coefficient
        self.kGd = 100000.0     # stiffness of ground
        self.kEd = 2000.0       # stiffness of edge springs
        self.damping = 0.1      # global discount on velocity, vel_t+dt = vel_t * (1 - damping * dt)
        self.dampingSpring = 2  # damping of the spring
        self.dampingGround = 0  # damping of the ground
        self.g = 9.8 * 2
        self.gUnit = np.array([0, 0, -1])  # unit gravity direction
        self.sMax = 0.3         # max shrinkage ratio
        self.h = 0.004          # step size

        self.ground = 0  # 0: smooth, 1: all friction, 2: directional friction
        self.modelName = modelName

        # ======= spaces =======
        self.actionSpace = None
        self.observationSpace = None

        # ======= init =======
        self.reset()

    def reset(self):
        # reset
        self.read(self.modelName)
        self.nSteps = 0
        self.time = 0

        #  initialize parameters
        self.s = np.zeros([self.e.shape[0], 1])  # shrinkage
        self.pressure = 1
        self.vel = np.zeros_like(self.v)

        self.computeLength()
        self.lInit = self.l
        self.computeGeometry()
        self.computeF()

        self.getObservation()
        self.getSpaces()

    # =============== info ===============
    def json(self):
        # return : a json string including v and e information, mainly for web interface
        v = self.v.tolist()
        e = self.e.tolist()
        data = {'v': v, 'e': e}
        js = json.dumps(data)
        return js

    def getObservation(self):
        # return : the observation of the environment, not used yet.
        v0 = self.v[0]                        # position of v0
        v = (self.v - self.v[0])              # relative positions of vs to v0
        vel = self.vel
        observation = np.vstack([v0, v, vel]).reshape(-1)
        return observation

    def getCentroid(self):
        return np.sum(self.v, 0) / self.v.shape[0]

    def getSpaces(self):
        self.actionSpace = spaces.Box(
            low=np.zeros(self.s.shape[0]),
            high=np.full(self.s.shape[0], self.sMax),
            dtype=np.float64
        )
        observation = self.getObservation()
        self.observationSpace = spaces.Box(
            low=np.full(observation.shape[0], -np.Inf),
            high=np.full(observation.shape[0], np.Inf),
            dtype=np.float64
        )
        return self.actionSpace, self.observationSpace

    # =============== environment setting ===============
    def read(self, name='tet'):
        # read the geometry of the bot from a json file
        # input: name: file name of the geometry file in json format under 'data/model/' folder

        self.modelName = name
        self.v = []
        self.e = []
        name = os.path.join(rootPath, "data/model/", name+".json")
        with open(name) as f:
            content = f.read()
            data = json.loads(content)
            self.e = np.array(data['e'])
            self.v = np.array(data['v'])

    def smooth(self):
        self.ground = 0

    def rough(self):
        self.ground = 1

    def directional(self):
        self.ground = 2

    # =============== robot setting ===============
    def setShrinkage(self, s=None):
        if s is None:
            s = self.s * 0
        elif type(s) in [float, int]:
            s = np.ones_like(self.s) * s
        else:
            s = np.array(s).reshape(-1, 1)
        assert (self.actionSpace.contains(s.reshape(-1)))
        self.s = s

    def setPressure(self, pressure=1):
        assert(0 <= pressure <= 1)
        self.pressure = pressure

    # =============== numerical simulation ===============

    # compute geometry
    def computeLength(self):
        v0 = self.v[self.e[:, 0]].reshape(-1, 3)
        v1 = self.v[self.e[:, 1]].reshape(-1, 3)
        self.l = np.sqrt(np.sum((v0 - v1) ** 2, 1)).reshape(-1, 1)

    def computeLength0(self):
        self.l0 = self.lInit * (1 - self.s * self.pressure)

    def computeContact(self):
        self.maskContact = (self.v[:, 2] <= 0)

    def computeGeometry(self):
        self.computeLength()
        self.computeLength0()
        self.computeContact()

    # compute force
    def computeFEdge(self):
        Fk = self.kEd * (self.l - self.l0)

        v0 = self.v[self.e[:, 0]]  # [ne x 3]
        v1 = self.v[self.e[:, 1]]
        displacement = v1 - v0
        norm = np.sum(displacement ** 2, 1).reshape(-1, 1)
        dispUnit = displacement / norm  # [ne x 3]

        vel0 = self.vel[self.e[:, 0]]  # [ne x 3]
        vel1 = self.vel[self.e[:, 1]]
        mVel0 = np.sum(vel0 * dispUnit, 1).reshape(-1, 1)  # [ne x 1]
        mVel1 = np.sum(vel1 * dispUnit, 1).reshape(-1, 1)
        dVel = mVel1 - mVel0  # positive: elongating  # [ne x 1]
        FDamping = self.dampingSpring * dVel ** 3  # positive: contract

        self.FEdge = Fk + FDamping

    def computeFShrinkage(self):
        Fss = []
        for i, v in enumerate(self.v):
            idsEdge = np.where((self.e == i)[:, 0] + (self.e == i)[:, 1])[0]
            pairs = self.e[idsEdge]  # [? x 2] tuple of iv of incident edges
            vsAdj = self.v[pairs[pairs != i]]  # [? x 3] pos of adjacent vertices
            displacement = vsAdj - v[np.newaxis, :]  # [? x 3]
            norm = np.sqrt(np.sum(displacement ** 2, 1)).reshape(-1, 1)  # [? x 1]
            dispUnit = displacement / norm  # [? x 3]
            Fs = np.sum(self.FEdge[idsEdge] * dispUnit, 0).reshape(1, 3)  # [1, 3]

            velAdj = self.vel[pairs[pairs != i]]    # [? x 3] velocity of adjacent vertices
            velDiff = (velAdj - self.vel[i].reshape(1, 3))  # [? x 3] difference of adjacent velocity
            Fdamping = np.sum(velDiff * dispUnit * self.dampingSpring, 0).reshape(1, 3) # [1, 3]

            Fs = Fs + Fdamping
            Fss.append(Fs)
        self.Fs = np.vstack(Fss)

    def computeFg(self):
        self.Fg = np.copy(self.gUnit) * self.g

    def computeFn(self):
        Fk = np.zeros_like(self.v)
        Fk[:, 2] = (0 - self.v[:, 2]) * self.kGd
        Fk[np.invert(self.maskContact)] = 0

        Fdamping = np.zeros_like(self.v)
        iRows = np.arange(self.v.shape[0])[self.maskContact]
        iCols = np.ones_like(iRows) * 2
        iCells = np.vstack([iRows, iCols])
        Fdamping[iCells] = self.vel[iCells] ** 3 * self.dampingGround

        self.Fn = Fk + Fdamping

    def computeF(self):
        self.computeFEdge()
        self.computeFShrinkage()
        self.computeFg()
        self.F = self.Fs + self.Fg

    # compute velocity
    def computeVelGndFrc(self):
        # post process the velocity to simulate the ground friction

        if self.ground == 2:
            boolRows = (self.vel[:, 0] < 0) * self.maskContact
            iRows = np.arange(self.v.shape[0])[boolRows]
            iCols = np.zeros_like(iRows)
            iCells = tuple(np.vstack([iRows, iCols]))
            self.vel[iCells] = 0
        elif self.ground == 1:
            boolRows = self.maskContact
            iRows = np.arange(self.v.shape[0])[boolRows]
            iCols = np.zeros_like(iRows)
            iCells = tuple(np.vstack([iRows, iCols]))
            self.vel[iCells] = 0
            iCols = np.ones_like(iRows)
            iCells = tuple(np.vstack([iRows, iCols]))
            self.vel[iCells] = 0
        elif self.ground == 0:
            pass

    def computeVelGndCon(self):
        # post process velocity to simulate ground contact

        boolRows = self.maskContact * (self.v[:, 2] < 0)
        iRows = np.arange(self.v.shape[0])[boolRows]
        iCols = np.ones_like(iRows) * 2
        iCells = tuple(np.vstack([iRows, iCols]))
        self.v[iCells] = 0

    def computeVelDamping(self):
        # post process velocity for damping

        d = (1 - np.exp(-self.vel ** 6 * 0.0001)) * self.damping
        d = self.damping
        self.vel = self.vel * (1 - d)

    def computeVel(self):
        self.vel = self.vel + self.F / self.m * self.h
        self.computeVelDamping()
        self.computeVelGndFrc()
        self.computeVelGndCon()

    # compute position
    def computePos(self):
        self.v = self.v + self.h * self.vel + self.h ** 2 * self.F / 2

    # simulate with n steps
    def step(self, nSteps=1):
        self.nSteps += nSteps
        self.time += self.h
        for i in range(nSteps):
            self.computeGeometry()
            self.computeF()
            self.computeVel()
            self.computePos()

