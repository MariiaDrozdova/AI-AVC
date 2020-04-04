import pylab as pl
import numpy as np
import random

def OneBanditOneLearnerOneRun(bandit, learner, timeHorizon):
    arms= []
    rewards = []
    regrets = []
    cumulativeregrets = []
    cumulativeregret =0
    for t in range(0,timeHorizon):
        arm = learner.chooseArmToPlay()
        reward,expectedInstantaneousRegret=bandit.GenerateReward(arm)
        learner.receiveReward(arm,reward)
        # Update statistics
        arms.append(arm)
        rewards.append(reward)
        regrets.append(expectedInstantaneousRegret)
        cumulativeregret = cumulativeregret+expectedInstantaneousRegret
        cumulativeregrets.append(cumulativeregret)
    return arms,rewards,regrets,cumulativeregrets

def ManyBanditOneLearnerOneRun(bandit, learner, timeHorizon, N):
    arms= [[]]*N
    rewards = [[]]*N
    regrets = [[]]*N
    cumulativeregrets = [[]]*N
    cumulativeregret =0
    for arm in range(bandit.A):
            reward,expectedInstantaneousRegret=bandit.GenerateReward(arm)
            learner.receiveReward(arm,reward) 
    for i in range(N):
        
        cur_arms,cur_rewards,cur_regrets,cur_cumulativeregrets = OneBanditOneLearnerOneRun(bandit, learner, timeHorizon)
        arms[i] = cur_arms
        rewards[i] = cur_rewards
        regrets[i] = cur_regrets
        cumulativeregrets[i] = cur_cumulativeregrets
        learner.clear()
    return arms,rewards,regrets,cumulativeregrets

def plotOneBanditOneLearnerOneRun(name, arms, rewards, regrets, cumulativeregrets, show=True):
    pl.figure(1)
    pl.clf()
    pl.xlabel("Arms", fontsize=16)
    pl.ylabel("Arm histogram", fontsize=16)
    pl.hist(arms, max(arms) + 1)
    if (show):
        pl.show()
    else:
        pl.savefig("./Figure-"+name+"-Arm histogram"+ '.pdf')

    pl.figure(2)
    pl.clf()
    pl.xlabel("Time steps", fontsize=16)
    pl.ylabel("Instantaenous rewards", fontsize=16)
    pl.plot(range(0, len(rewards)), rewards, 'black', linewidth=0, marker='.', markeredgewidth=1,
            markerfacecolor='none', markersize=1)
    if (show):
        pl.show()
    else:
        pl.savefig("./Figure-"+name+"-Instantaenous rewards"+ '.pdf')

    pl.figure(3)
    pl.clf()
    pl.xlabel("Regret values", fontsize=16)
    pl.ylabel("Instantaenous Regret histogram", fontsize=16)
    pl.hist(regrets, 50)
    if (show):
        pl.show()
    else:
        pl.savefig("./Figure-"+name+"-Instantaenous Regret histogram"+ '.pdf')

    pl.figure(4)
    pl.clf()
    pl.xlabel("Time steps", fontsize=16)
    pl.ylabel("Cumulative regret", fontsize=16)
    pl.plot(range(0, len(cumulativeregrets)), cumulativeregrets, 'black', linewidth=1, marker='.', markeredgewidth=1,
            markerfacecolor='none', markersize=4)
    if (show):
        pl.show()
    else:
        pl.savefig("./Figure-"+name+"-Cumulative regret"+ '.pdf')


def plotManyBanditOneLearnerOneRun(bandit, name, arms, rewards, regrets, cumulativeregrets, show=True,
                                   show_low_bound=True, show_upper_bound=False):
    average_cumulativeregrets = np.mean(cumulativeregrets, axis=0)
    std_cumulativeregrets = np.std(cumulativeregrets, axis=0)


    pl.figure(1)
    pl.clf()
    pl.xlabel("CumRegrets", fontsize=16)
    pl.ylabel("CumRegrets Histogram", fontsize=16)
    cumulativeregrets = np.array(cumulativeregrets)
    pl.hist(cumulativeregrets[:, -1], len(cumulativeregrets[:, -1]))
    if (show):
        pl.show()
    else:
        pl.savefig("./Figure-"+name+"-CumHist histogram"+ '.pdf')


    pl.figure(2)
    pl.clf()

    if show_low_bound:
        plot_LowerBound(bandit, len(average_cumulativeregrets), show=True)
    if show_upper_bound:
        plot_UpperBound(bandit, len(average_cumulativeregrets), show=True)


    pl.xlabel("Time steps", fontsize=16)
    pl.ylabel("Average cumulative regret", fontsize=16)
    tsave = range(0, len(average_cumulativeregrets))

    upperQuantile = np.quantile(cumulativeregrets, 0.95, 0) # requires numpy 1.15
    lowerQuantile = np.quantile(cumulativeregrets, 0.05, 0)


    pl.plot(tsave, upperQuantile, linestyle="dashed", color="b")
    pl.plot(tsave, lowerQuantile, linestyle="dashed", color="b")
    pl.plot(tsave, average_cumulativeregrets, 'b', linewidth=1, marker='.', markeredgewidth=1,
            markerfacecolor='none', markersize=4, label=name)
    pl.legend()
    
    if (show):
        pl.show()
    else:
        pl.savefig("./Figure-"+name+"-Average cumulative regret"+ '.pdf')


def plotToCompare(bandit,
                  name1, arms1, rewards1, regrets1, cumulativeregrets1, 
                  name2, arms2, rewards2, regrets2, cumulativeregrets2, 
                  algo1='FCN', algo2='UCB', show=True,
                  show_low_bound=True, show_upper_bound=False):
    average_cumulativeregrets1 = np.mean(cumulativeregrets1, axis=0)
    std_cumulativeregrets1 = np.std(cumulativeregrets1, axis=0)

    average_cumulativeregrets2 = np.mean(cumulativeregrets2, axis=0)
    std_cumulativeregrets2 = np.std(cumulativeregrets2, axis=0)

    pl.figure(1)
    pl.clf()
    pl.xlabel("Time steps", fontsize=16)
    pl.ylabel("Average cumulative regret", fontsize=16)

    #pl.errorbar(range(0, len(average_cumulativeregrets)), average_cumulativeregrets, yerr=std_cumulativeregrets, label='both limits (default)')
    pl.plot(range(0, len(average_cumulativeregrets1)), average_cumulativeregrets1, linewidth=1, marker='.', markeredgewidth=1,
            markerfacecolor='none', markersize=4,label=algo1)
    pl.plot(range(0, len(average_cumulativeregrets2)), average_cumulativeregrets2, linewidth=1, marker='.', markeredgewidth=1,
            markerfacecolor='none', markersize=4, label=algo2)
    if show_low_bound:
        plot_LowerBound(bandit, len(average_cumulativeregrets1), show=True)
    if show_upper_bound:
        plot_UpperBound(bandit, len(average_cumulativeregrets1), show=True)
    pl.legend()
    if (show):
        pl.show()
    else:
        pl.savefig("./Figure-"+name+"-Average cumulative regret"+ '.pdf')
        
    pl.figure(2)
    pl.clf()
    pl.xlabel("CumRegrets Comparison", fontsize=16)
    pl.ylabel("CumRegrets Histogram Comparison", fontsize=16)

    cumulativeregrets2 = np.array(cumulativeregrets2)
    pl.hist(cumulativeregrets2[:, -1], 10, fc=(0, 1, 0, 0.5), label=algo2)
    cumulativeregrets1 = np.array(cumulativeregrets1)
    pl.hist(cumulativeregrets1[:, -1], 10, fc=(1, 0, 0, 0.5), label=algo1)
    pl.legend()
    

    if (show):
        pl.show()
    else:
        pl.savefig("./Figure-"+name+"-CumHist histogram"+ '.pdf')



def plotToCompare2(bandit,
                   outputs,
                   algos, 
                   show=True,
                   show_error=False,
                   show_low_bound=True):

    average_cumulativeregrets = [np.mean(output[3], axis=0) for output in outputs]
    std_cumulativeregrets = [np.std(output[3], axis=0) for output in outputs]

    pl.figure(1)
    pl.clf()
    pl.xlabel("Time steps", fontsize=16)
    pl.ylabel("Average cumulative regret", fontsize=16)

    #pl.errorbar(range(0, len(average_cumulativeregrets)), average_cumulativeregrets, yerr=std_cumulativeregrets, label='both limits (default)')
    for i in range(len(outputs)): 
        col = (random.random(), random.random(), random.random())
        cumulativeregrets = np.array(outputs[i][3])
        upperQuantile = np.quantile(cumulativeregrets, 0.95, 0) # requires numpy 1.15
        lowerQuantile = np.quantile(cumulativeregrets, 0.05, 0)

        tsave = range(0, len(upperQuantile))
        if show_error:
            pl.plot(tsave, upperQuantile, linestyle="dashed", color=col)
            pl.plot(tsave, lowerQuantile, linestyle="dashed", color=col)
        pl.plot(tsave, average_cumulativeregrets[i], color=col, linewidth=1, marker='.', markeredgewidth=1,
                markerfacecolor='none', markersize=4, label=algos[i])
        pl.legend()
    if show_low_bound:
        plot_LowerBound(bandit, len(tsave), show=True)
    pl.legend()
    if (show):
        pl.show()
    else:
        pl.savefig("./Figure-"+name+"-Average cumulative regret"+ '.pdf')
        
    pl.figure(2)
    pl.clf()
    
        
    for i in range(len(outputs)): 
        cumulativeregrets = np.array(outputs[i][3])
        pl.hist(cumulativeregrets[:, -1], 10, fc=(random.random(), random.random(), random.random(), 0.5), label=algos[i])

    pl.legend()
    pl.xlabel("CumRegrets Comparison", fontsize=16)
    pl.ylabel("CumRegrets Histogram Comparison", fontsize=16)
    
    



    if (show):
        pl.show()
    else:
        pl.savefig("./Figure-"+name+"-CumHist histogram"+ '.pdf')

def plot_LowerBound(bandit, max_T, show=True):
    def kl(x, y):
        res = x*np.log(x/y) + (1-x)*np.log((1-x)/(1-y))
        res = (res == 0) + res
        return res
    means = np.array(bandit.armMeans)
    m_star = bandit.bestarm
    
    T = np.array(range(max_T))+1
    print(kl(means, means[m_star]))
    
    F = np.sum((means[m_star] - means)/kl(means, means[m_star]))*np.log(T)

    pl.xlabel("Time steps", fontsize=16)
    pl.ylabel("LowerBound for R", fontsize=16)
    pl.plot(T, F, label='LBA')
    if (show):
        pass
        #pl.show()
    else:
        pl.savefig("./Figure-"+name+"-Lower Bound for R"+ '.pdf')
        
def plot_UpperBound(bandit, max_T, show=True):
    def kl(x, y):
        res = x*np.log(x/y) + (1-x)*np.log((1-x)/(1-y))
        res = (res == 0) + res
        return res
    means = np.array(bandit.armMeans)
    m_star = bandit.bestarm
    
    T = np.array(range(max_T))+1
    print(kl(means, means[m_star]))
    
    F = np.sum((means[m_star] - means))*(T)

    pl.xlabel("Time steps", fontsize=16)
    pl.ylabel("UpperBound for R", fontsize=16)
    pl.plot(T, F, label='UBA')
    if (show):
        pass
        #pl.show()
    else:
        pl.savefig("./Figure-"+name+"-Upper Bound for R"+ '.pdf')
