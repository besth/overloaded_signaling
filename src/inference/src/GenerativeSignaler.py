import numpy as np 
import pandas as pd

"""
    signalSpace: list of all possible signals, inpit type must be valid pandas index
    signalIsConsistent: function specifying whether the signal is logically consistent given a mind and signaler category
"""
class SignalerZero(object):
    def __init__(self, signalSpace, signalIsConsistent):
        self.signalSpace = signalSpace
        self.signalIsConsistent = signalIsConsistent

    def __call__(self, targetMind, signalerCategory): #p(signal|mind, category), for all signals in signal space
        #create a dataframe that adds signal as an index to the target mind and a column labeled p(signal|mind,c)
        likelihoodComponents = pd.DataFrame(data=np.inf,index=targetMind.index, columns=self.signalSpace).stack()
        likelihoodComponents.index.names = targetMind.index.names + ['signals']
        likelihoodComponents.name = 'p(signal|mind,c)'

        #for each condition apply the get likelihood function, returns a distribution
        signalLikelihoods = likelihoodComponents.groupby(likelihoodComponents.index.names).apply(self.getSignalLikelihoodGivenMind, signalerType = signalerCategory)
        return(pd.DataFrame(signalLikelihoods))

    def getSignalLikelihoodGivenMind(self, signalingCondition, signalerType):
        #extract the world and signal from the index condition
        signal = signalingCondition.index.get_level_values('signals')[0]

        world = signalingCondition.index.get_level_values('worlds')[0]
        desire = signalingCondition.index.get_level_values('desires')[0]
        goal = signalingCondition.index.get_level_values('goals')[0]
        action = signalingCondition.index.get_level_values('actions')[0]
        mind = {'worlds': world, 'desires':desire, 'goals':goal, 'actions':action}
        
        #check if signal is consistent with signaler type and mind, if so return 1/size of possible consisent signals
        if self.signalIsConsistent(signal, mind, signalerType) and (signal in self.signalSpace):
            numberPossibleSignals = sum([self.signalIsConsistent(s, mind, signalerType) for s in self.signalSpace])
            return(1.0/numberPossibleSignals)
        return(0.0)