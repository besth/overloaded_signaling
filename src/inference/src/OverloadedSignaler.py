import numpy as np 
import pandas as pd
from OverloadedReceiver import *
from mindConstruction import getMultiIndexMindSpace

class SignalerOne(object):
    def __init__(self, alpha, signalSpace, getActionUtility, getReceiverZero):
        self.alpha = alpha
        self.signalSpace = signalSpace
        self.getActionUtility = getActionUtility
        self.getReceiverZero = getReceiverZero

    def __call__(self, observation):
        signalSpaceDF = getMultiIndexMindSpace({'signals': self.signalSpace})

        #get the signal utility for each signal with respect to the observation from environment
        getConditionUtility = lambda x: np.exp(self.alpha*self.getUtilityofSignal(observation, x.index.get_level_values('signals')[0]))
        utilities = signalSpaceDF.groupby(signalSpaceDF.index.names).apply(getConditionUtility)
        
        #normalize probabilities and return as a pd DF column
        sumOfUtilities = sum(utilities)
        probabilities = utilities.groupby(utilities.index.names).apply(lambda x: x/sumOfUtilities)
        signalSpaceDF['probabilities'] = signalSpaceDF.index.get_level_values(0).map(probabilities.get)
        return(signalSpaceDF)  

    def getUtilityofSignal(self, observation, signal):
        #determine which mind components are observed
        if 'goals' in observation.keys():
            goal = observation['goals']
        else:
            goal = None

        if 'worlds' in observation.keys():
            world = observation['worlds']
        else:
            world = None

        #get the posterior of the mind, sum across all possible minds to get a distribution of actions
        mindPosterior = self.getReceiverZero(signal)
        actionPosterior = pd.DataFrame(mindPosterior.groupby(level=['actions']).sum())

        #find the action utilities and evaluate with respect to information from the speaker (observation) E_a[U(mind_speaker, a)|signal]
        getConditionActionUtility = lambda x: self.getActionUtility(x.index.get_level_values('actions')[0], world, goal)
        actionPosterior['utility'] = actionPosterior.groupby(actionPosterior.index.names).apply(getConditionActionUtility)
        return(sum(actionPosterior['p(mind|signal)']*actionPosterior['utility']))
