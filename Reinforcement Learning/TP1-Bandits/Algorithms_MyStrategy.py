import numpy as np

class Fancy:
    def __init__(self,nbArms,maxReward=1.):
        ...
        self.clear()

    def clear(self):
        ...

    def chooseArmToPlay(self):
        return ...

    def receiveReward(self,arm,reward):
        ...

    def name(self):
        return "myFancyStrategy"