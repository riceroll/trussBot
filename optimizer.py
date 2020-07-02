from pathos.multiprocessing import ProcessPool as Pool
import multiprocessing
import numpy as np


class Optimizer(object):
    def __init__(self, env=None):
        self.env = env

    def maximize(self):
        raise NotImplementedError('Please implement maximize function.')


class EvolutionAlgorithm(Optimizer):
    def __init__(self, env=None, nGenMax=50, nPop=40, nHero=3,
                 mortality=0.8, pbCross=0.5, pbMut=0.1, pbCrossDig=0.5, pbMutDig=0.2):
        super().__init__(env)

        self.lb = self.env.model.actionSpace.low
        self.ub = self.env.model.actionSpace.high
        self.nStages = 4
        self.interval = (self.ub - self.lb) / 4
        self.lbInt = np.zeros_like(self.lb)
        self.ubInt = np.ones_like(self.ub) * self.nStages

        self.nPop = nPop
        self.mortality = mortality
        self.nHero = nHero
        self.pbCross = pbCross
        self.pbMut = pbMut
        self.pbCrossDig = pbCrossDig
        self.pbMutDig = pbMutDig

        # variables
        self.oldPop = None
        self.pop = None
        self.fits = None
        self.nGen = None

        self.reset()

    def reset(self):
        self.nGen = 0
        self.oldPop = None
        self.initPop()

    def randPop(self, n):
        return np.random.randint(self.lbInt, self.ubInt, size=(n, len(self.lbInt)))

    def initPop(self):
        self.pop = self.randPop(self.nPop)  # [nPop x len(lb) ]

    def popIntToFloat(self, pop):
        return pop * self.interval

    def evaluate(self, disp=False):
        popFloat = self.popIntToFloat(self.pop)
        pops = [p for p in popFloat]
        print(len(pops))
        with Pool(multiprocessing.cpu_count()) as p:
            self.fits = np.array(p.map(self.env.criterion, pops))

        meanFit = np.mean(self.fits)
        maxFit = np.max(self.fits)
        minFit = np.min(self.fits)
        self.sort()
        if disp:
            print('nGen: ', self.nGen)
            print('mean: ', meanFit)
            print('max: ', maxFit)
            print('min: ', minFit)

    def getSurvivor(self, i=0):
        return self.popIntToFloat(self.pop[i])

    def sort(self):
        order = np.argsort(self.fits)[::-1]
        self.pop = self.pop[order]
        self.fits = self.fits[order]

    def shuffle(self):
        order = np.random.permutation(len(self.pop))
        self.pop = self.pop[order]
        self.fits = self.fits[order]

    def select(self):
        fitMin = np.min(self.fits)
        fitMax = np.max(self.fits)
        fitInterval = fitMax - fitMin + 1e-8
        distance = (fitMax - self.fits) / fitInterval  # normalized distance of the fitness to the max fitness
        pbDie = distance ** 3 * self.mortality
        pbDie[:self.nHero] = 0

        dice = np.random.rand(len(self.pop))
        maskSurvived = np.invert(dice < pbDie)
        self.oldPop = np.copy(self.pop)
        self.pop = self.pop[maskSurvived]
        self.fits = self.fits[maskSurvived]

    def crossOver(self):
        self.shuffle()

        diceDig = np.random.rand(self.pop.shape[0] // 2, self.pop.shape[1])
        maskDig = diceDig < self.pbCrossDig
        dice = np.random.rand(self.pop.shape[0] // 2)
        mask = dice < self.pbCross
        maskDig = maskDig * mask[:, np.newaxis]
        maskDig = np.repeat(maskDig, 2, axis=0)
        maskDig0 = np.copy(maskDig)
        maskDig1 = np.copy(maskDig)
        maskDig0[np.arange(len(maskDig0))[::2]] = False
        maskDig1[np.arange(len(maskDig1))[1::2]] = False

        if len(maskDig0) < len(self.pop):
            maskDig0 = np.pad(maskDig0, ((0, 1), (0, 0)), 'constant', constant_values=((False, False), (False, False)))
            maskDig1 = np.pad(maskDig1, ((0, 1), (0, 0)), 'constant', constant_values=((False, False), (False, False)))

        self.pop[maskDig0], self.pop[maskDig1] = self.pop[maskDig1], self.pop[maskDig0]

    def mutate(self):
        diceDig = np.random.rand(self.pop.shape[0], self.pop.shape[1])
        maskDig = diceDig < self.pbMutDig
        dice = np.random.rand(self.pop.shape[0])
        mask = dice < self.pbMut
        maskDig = maskDig * mask[:, np.newaxis]

        newPop = self.randPop(self.pop.shape[0])
        self.pop[maskDig] = newPop[maskDig]

    def regenerate(self):
        nDead = self.nPop - len(self.pop)
        self.pop = np.append(self.pop, self.oldPop[:nDead], axis=0)
        self.fits = np.pad(self.fits, (0, nDead), 'wrap')

    def maximize(self, nSteps=1):
        self.initPop()
        self.evaluate(True)
        self.sort()
        for i in range(nSteps):
            self.nGen += 1
            self.select()
            self.crossOver()
            self.mutate()
            self.regenerate()
            self.evaluate(True)
        return self.getSurvivor(0)
