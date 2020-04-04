import numpy as np
import Algorithms_kullback as kl
import scipy.stats as ss
import math
import random


class FTL:
    def __init__(self,nbArms, distribution='bern'):
        self.A = nbArms
        self.clear()

    def clear(self):
        self.NbPulls = np.zeros(self.A)
        self.Means = np.ones(self.A)

    def chooseArmToPlay(self):
        return np.argmax(self.Means)

    def receiveReward(self,arm,reward):
        self.Means[arm] = (self.Means[arm]*self.NbPulls[arm]+reward)/(self.NbPulls[arm]+1.)
        self.NbPulls[arm] = self.NbPulls[arm] +1

    def name(self):
        return "FTL"


class UCB:
    def __init__(self, nbArms, maxReward=1., distribution='bern'):
        self.A = nbArms
        self.mR = maxReward
        
        self.clear()

    def clear(self):
        self.NbPulls = np.zeros(self.A)
        self.Means = np.ones(self.A)*self.mR
        #self.NbPullsTime = [[0]*self.A]
        self.t = 0

    def chooseArmToPlay(self):
        self.t += 1
        #np_NbPullsTime = np.array(self.NbPullsTime)
        arg1 = 0.5*np.log(self.t*self.t*(self.t+1))/((self.NbPulls == 0)*1.0 + self.NbPulls)
        arg1 = (arg1 >= 0)*arg1
        array = self.Means + np.sqrt(arg1)

        #array = self.Means + np.sqrt(2*np.std(np_NbPullsTime, axis=0)*3*np.log(self.t)/(self.NbPullsTime[-1])) + 21*np.log(self.t)/(3*self.NbPullsTime)
        res = np.argmax(array)
        max_elements = (array==np.argmax(array))
        if np.sum(max_elements)>1:
            array = max_elements*self.NbPulls + np.inf*(1-max_elements)
            res = argmin(array)

        min_elements = (array==np.argmin(array))
        if np.sum(min_elements)>1:
            res = np.random.choice(np.where(np.array(min_elements)>=1)[0])
        #newline = self.NbPullsTime[-1].copy()
        #newline[res] += 1
        #self.NbPullsTime.append(newline)
        return res

    def receiveReward(self, arm, reward):
        self.Means[arm] = (self.Means[arm]*self.NbPulls[arm]+reward)/(self.NbPulls[arm]+1.)
        self.NbPulls[arm] = self.NbPulls[arm] +1

    def name(self):
        return "UCB"


class KLUCB:
    def __init__(self, nbArms, maxReward=1., distribution='bern'):
        self.A = nbArms
        self.mR = maxReward
        self.clear()
        self.return_Ua = kl.klucbBern
        

    def clear(self):
        self.NbPulls = np.zeros(self.A)
        self.Means = np.ones(self.A)*self.mR
        self.t = 0
        for a in range(self.A):
            if self.NbPulls[a] == 0:
                self.t += 1
                self.receiveReward(a, self.mR)   

    def chooseArmToPlay(self):
             
        self.t += 1
        f_t = np.log(self.t)+3*np.log(np.log(self.t))
        U = []
        for a in range(self.A):
            Ua = self.return_Ua(self.Means[a], f_t/self.NbPulls[a])
            U.append(Ua)
        a = np.argmax(U)
        return a
    
    
    def receiveReward(self, arm, reward):
        self.Means[arm] = (self.Means[arm]*self.NbPulls[arm]+reward)/(self.NbPulls[arm]+1.)
        self.NbPulls[arm] = self.NbPulls[arm] +1

    def name(self):
        return "KL-UCBBern"
    
    
class KLUCBgeneral:
    def __init__(self, nbArms, maxReward=1., distribution='bern'):
        self.A = nbArms
        self.mR = maxReward
        self.clear()
        self.return_Ua = kl.klucbBern
        if distribution == 'gauss':
            self.return_Ua = kl.klucbGauss         
        if distribution == 'poisson':
            self.return_Ua = kl.klucbPoisson             
        if distribution == 'exp':
            self.return_Ua = kl.klucbExp
            
    def clear(self):
        self.NbPulls = np.zeros(self.A)
        self.Means = np.ones(self.A)*self.mR
        self.t = 0
        for a in range(self.A):
            if self.NbPulls[a] == 0:
                self.t += 1
                self.receiveReward(a, self.mR)   

    def chooseArmToPlay(self):
             
        self.t += 1
        f_t = np.log(self.t)+3*np.log(np.log(self.t))
        U = []
        for a in range(self.A):
            Ua = self.return_Ua(self.Means[a], f_t/self.NbPulls[a])
            U.append(Ua)
        a = np.argmax(U)
        return a
    
    def receiveReward(self, arm, reward):
        self.Means[arm] = (self.Means[arm]*self.NbPulls[arm]+reward)/(self.NbPulls[arm]+1.)
        self.NbPulls[arm] = self.NbPulls[arm] +1

    def name(self):
        return "KL-UCB"



class TS:
    def __init__(self, nbArms, maxReward=1., distribution='bern'):
        self.A = nbArms
        self.mR = maxReward
        self.distribution = distribution
        self.clear()

    def clear(self):
        self.NbPulls = np.zeros(self.A)
        self.Means = np.ones(self.A)*self.mR
        self.rewards = np.zeros(self.A)
        self.t = 0

    def chooseArmToPlay(self):
        thetas = []
        for arm in range(self.A):
            if self.distribution == 'bern':
                r_arm = ss.beta.rvs(self.rewards[arm]+1, self.NbPulls[arm] - self.rewards[arm] + 1, size=1)[0]
            if self.distribution == 'gauss' or self.distribution != 'bern':
                r_arm = self.Means[arm] + random.random()*(1./np.sqrt(self.NbPulls[arm]+1.))
            thetas.append(r_arm)
        return np.argmax(np.array(thetas))

    def receiveReward(self, arm, reward):
        self.Means[arm] = (self.Means[arm]*self.NbPulls[arm]+reward)/(self.NbPulls[arm]+1.)
        self.rewards[arm] += 1.0*(reward == self.mR)
        self.NbPulls[arm] = self.NbPulls[arm] +1

    def name(self):
        return "TS"
    
class UCBpeeling:
    def __init__(self, nbArms, maxReward=1., distribution='bern'):
        self.A = nbArms
        self.mR = maxReward
        self.alpha = 8
        self.delta = 0.01   
        self.clear()

    def clear(self):
        self.NbPulls = np.zeros(self.A)
        self.Means = np.ones(self.A)*self.mR
        self.t = 0

    def chooseArmToPlay(self):
        self.t += 1
        #np_NbPullsTime = np.array(self.NbPullsTime)
        #print((self.NbPulls + 1.0*(self.NbPulls==0)))
        coef = self.alpha/(2*(self.NbPulls + 1.0*(self.NbPulls==0)))
        #print(math.ceil((np.log(self.t)/np.log(self.alpha))+1))
        #print(np.log(self.t)/np.log(self.alpha))
        arg1 = coef*np.log(1./self.delta*(math.ceil((np.log(self.t)/np.log(self.alpha))+1)))
        #print(arg1)
        arg1 = (arg1 >= 0)*arg1
        array = self.Means + np.sqrt(arg1)
        #print(array)

        res = np.argmax(array)
        max_elements = (array==np.argmax(array))
        if np.sum(max_elements)>1:
            array = max_elements*self.NbPulls + np.inf*(1-max_elements)
            res = argmin(array)

        min_elements = (array==np.argmin(array))
        if np.sum(min_elements)>1:
            res = np.random.choice(np.where(np.array(min_elements)>=1)[0])
        return res

    def receiveReward(self, arm, reward):
        self.Means[arm] = (self.Means[arm]*self.NbPulls[arm]+reward)/(self.NbPulls[arm]+1.)
        #print(self.Means)
        self.NbPulls[arm] = self.NbPulls[arm] +1

    def name(self):
        return "UCB-peeling"
    
class UCBlaplace:
    def __init__(self, nbArms, maxReward=1., distribution='bern'):
        self.A = nbArms
        self.mR = maxReward
        self.delta = 0.01   
        self.clear()

    def clear(self):
        self.NbPulls = np.zeros(self.A)
        self.Means = np.ones(self.A)*self.mR
        self.t = 0

    def chooseArmToPlay(self):
        self.t += 1
        #np_NbPullsTime = np.array(self.NbPullsTime)
        NbPulls = (self.NbPulls + 1.0*(self.NbPulls==0))
        arg1 = (1+1./NbPulls)/(2*NbPulls)*np.log(np.sqrt(self.NbPulls+1)/self.delta)
        arg1 = (arg1 >= 0)*arg1
        array = self.Means + np.sqrt(arg1)

        res = np.argmax(array)
        max_elements = (array==np.argmax(array))
        if np.sum(max_elements)>1:
            array = max_elements*self.NbPulls + np.inf*(1-max_elements)
            res = argmin(array)

        min_elements = (array==np.argmin(array))
        if np.sum(min_elements)>1:
            res = np.random.choice(np.where(np.array(min_elements)>=1)[0])
        return res

    def receiveReward(self, arm, reward):
        self.Means[arm] = (self.Means[arm]*self.NbPulls[arm]+reward)/(self.NbPulls[arm]+1.)
        self.NbPulls[arm] = self.NbPulls[arm] +1

    def name(self):
        return "UCB-Laplace"
    
    
class BESA:
    def __init__(self, nbArms=2, maxReward=1., distribution='bern'):
        self.A = 2 #FOR TWO ONLY
        self.mR = maxReward
        self.clear()
        
        self.lim = 20

    def clear(self):
        self.rewards = [self.mR, self.mR]
        self.arms = [0, 1]
        
        self.t = 0

    def chooseArmToPlay(self):
        self.t += 1
        
        #np_NbPullsTime = np.array(self.NbPullsTime)
        b = np.array(self.arms)
        a = np.array(self.rewards)
        n = min(max(len(np.where(b==0)[0]),1), max(len(np.where(b==1)[0]),1))
        array_0 = a[np.where(b==0)]
        array_1 = a[np.where(b==1)]
        if n==max(len(np.where(b==0)[0]),1):
            m = max(len(np.where(b==1)[0]),1)
            indexes = np.random.choice(m, n, replace=False)
            array_1 = array_1[indexes]
        else:
            m = max(len(np.where(b==0)[0]),1)
            indexes = np.random.choice(m, n, replace=False)
            array_0 = array_0[indexes]            
            
        means = [np.mean(array_0), np.mean(array_1)]
        means = np.array(means)
        res = np.argmax(means)
        #print(means)
        #print(res)
        return res

    def receiveReward(self, arm, reward):
        self.rewards.append(reward)
        self.arms.append(arm)

    def name(self):
        return "BESA"
    
class BESAcheat:
    def __init__(self, nbArms=2, maxReward=1., distribution='bern'):
        self.A = 2 #FOR TWO ONLY
        self.mR = maxReward
        self.clear()
        
        self.lim = 100

    def clear(self):
        self.rewards = [self.mR, self.mR]
        self.arms = [0, 1]
        
        self.t = 0

    def chooseArmToPlay(self):
        self.t += 1
        
        #np_NbPullsTime = np.array(self.NbPullsTime)
        b = np.array(self.arms)
        a = np.array(self.rewards)
        
        n = min(max(len(np.where(b==0)[0]),1), max(len(np.where(b==1)[0]),1))
        array_0 = a[np.where(b==0)]
        array_1 = a[np.where(b==1)]
        if n==max(len(np.where(b==0)[0]),1):
            m = max(len(np.where(b==1)[0]),1)
            indexes = np.random.choice(m, n, replace=False)
            array_1 = array_1[indexes]
        else:
            m = max(len(np.where(b==0)[0]),1)
            indexes = np.random.choice(m, n, replace=False)
            array_0 = array_0[indexes]            
            
        means = [np.mean(array_0), np.mean(array_1)]
        means = np.array(means)
        res = np.argmax(means)
        #print(means)
        #print(res)
        return res

    def receiveReward(self, arm, reward):
        self.rewards.append(reward)
        self.arms.append(arm)
        if len(self.rewards) > self.lim:
            self.rewards = self.rewards[-self.lim:]
            self.arms =  self.arms[-self.lim:]
        

    def name(self):
        return "BESA-cheat"

