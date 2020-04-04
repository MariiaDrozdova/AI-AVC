import Environments_StochasticBandits as env
import Algorithms_Baselines as alg
import Algorithms_MyStrategy as myalg
import Experiments_MakeBanditExperiments as xps
import time

nbArms=3
timeHorizon=500
# 1. Create a bandit environment
bandit = env.StochasticBandit(nbArms)
bandit.createBernoulliArmsFromMeans([0.2,0.4,0.6])
# bandit.createBernoulliArms(0.1) # Using a minimal gap of 0.1

#2. Create a bandit algorithm
learner = alg.FTL(nbArms)
# learner = alg.UCB(nbArms)
# learner = alg.myUCB(nbArms)
# learner = alg.KLUCB(nbArms)
# learner = alg.TS(nbArms)
# learner = alg.BESA(nbArms)

#3. Run an experiment and collects data
arms,rewards,regrets,cumulativeregrets = xps.OneBanditOneLearnerOneRun(bandit, learner, timeHorizon)

# 4. Generate a name and plot the data
name = str(int(time.time())) + bandit.name() + "-" + learner.name()
xps.plotOneBanditOneLearnerOneRun(name, arms, rewards, regrets, cumulativeregrets, show=False)
