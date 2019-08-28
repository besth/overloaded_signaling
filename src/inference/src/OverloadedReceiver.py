import numpy as np
import pandas as pd 
from mindConstruction import getMultiIndexMindSpace

"""
    commongroundDictionary: dictionary {'mind component label': list of items in mind component space}
    constructMind: function generating prior over possible target minds from a dictionary with mind component labels as keys and desired pandas indices as values,
    getSignalLikelihood: generative signaler function
    signaCategoryPrior: dictionary {signalType: probability of signalType}
"""
class ReceiverZero(object):
    def __init__(self, commonGroundDictionary, constructMind, getSignalerZero, signalCategoryPrior):
        self.mindPrior = constructMind(commonGroundDictionary)
        self.getSignalLikelihood = getSignalerZero
        self.signalCategoryPrior = signalCategoryPrior

        #index and column names for dataframe
        self.mindLabels = list(commonGroundDictionary.keys())
        self.signalerTypeLabel = 'signalerType' 
        self.signalLabel = 'signals'
        self.pMindLabel, self.pCatLabel, self.pJointLabel, self.pLikekihoodLabel, self.pPosteriorLabel = ['p(mind)','p(c)', 'p(mind,c)', 'p(signal|mind,c)', 'p(mind|signal)']

    def __call__(self, signal):
        mindAndCategoryPrior = self.constructJointMindSignalCategoryPrior(self.mindPrior, self.signalCategoryPrior)
        likelihoodDF = self.constructLikelihoodDataFrameFromMindConditions(self.mindPrior)
        mindPosterior = self.getMindPosterior(mindAndCategoryPrior, likelihoodDF, signal)
        return(mindPosterior)

    def constructLikelihoodDataFrameFromMindConditions(self, mindPrior):
        categoryNames = list(self.signalCategoryPrior.keys())

        # find the signal likelihood distribution for each signaler type and concatenate dataframes into a single pandas DF distribution
        likelihoodByCategory = [self.getSignalLikelihood(mindPrior, signalerType) for signalerType in categoryNames]
        likelihoodDistributionList =  [pd.concat([likelihoodDist], keys=[categoryName], names=[self.signalerTypeLabel]) for likelihoodDist, categoryName in zip(likelihoodByCategory, categoryNames)]
        likelihoodDistributionDF = pd.concat(likelihoodDistributionList)

        return(likelihoodDistributionDF)

    def constructJointMindSignalCategoryPrior(self, mindPrior, categoryPrior):
        #from signal category prior, create a pandas df with index as category type label and column of p(c) probability
        categoryPriorDF = pd.DataFrame(list(categoryPrior.items()), columns=[self.signalerTypeLabel, self.pCatLabel])
        categoryPriorDF.set_index(self.signalerTypeLabel, inplace=True)

        #duplicate the mind prior * the number of possible signal type categories, set the index to the joint p(mind, c) combinations
        categoryNames = list(categoryPrior.keys())
        numberOfCategories = len(categoryNames)
        mindCPrior = pd.concat([mindPrior]*numberOfCategories, keys=categoryNames, names=[self.signalerTypeLabel])

        #merge the categoryPriorDF into the mindCPrior, take the product of p(mind)*p(c) columns and return the resulting column p(mind, c)
        jointPrior = pd.merge(left=mindCPrior.reset_index(level=self.mindLabels),right=categoryPriorDF,on=[self.signalerTypeLabel])
        jointPrior[self.pJointLabel] = jointPrior[self.pMindLabel] * jointPrior[self.pCatLabel]
        jointPrior = jointPrior.set_index(self.mindLabels,append=True)[[self.pJointLabel]]

        return(jointPrior)

    def getMindPosterior(self, jointPrior, likelihood, signal):
        #merge the prior and likelihood dataframes, take the product of p(mind,c)*p(signal|mind,c) and get the posterior distribution 
        posterior = pd.merge(left=jointPrior,right=likelihood.reset_index(level=[self.signalLabel]),on=[self.signalerTypeLabel]+self.mindLabels)
        posterior[self.pPosteriorLabel] = posterior[self.pJointLabel] * posterior[self.pLikekihoodLabel]
        posterior = posterior.set_index(posterior[self.signalLabel],append=True)[[self.pPosteriorLabel]]
        posterior = posterior.reorder_levels([self.signalLabel,self.signalerTypeLabel]+self.mindLabels)

        #extract the signal location, normalize, and integrate out the category type to get p(mind|signal)
        mindAndCPosterior = posterior.loc[signal]
        mindAndCPosterior[self.pPosteriorLabel] = mindAndCPosterior[self.pPosteriorLabel]/sum(mindAndCPosterior[self.pPosteriorLabel])
        mindPosterior = mindAndCPosterior.groupby(level=self.mindLabels).sum()
        return(mindPosterior)
