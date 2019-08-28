import pandas as pd
import numpy as np

#helper functions for pandas
def getMultiIndexMindSpace(mindLevelsDictionary):
    conditions = list(mindLevelsDictionary.keys())
    levelValues = list(mindLevelsDictionary.values())
    speakerMindIndex = pd.MultiIndex.from_product(levelValues, names=conditions)
    jointMind = pd.DataFrame(index=speakerMindIndex)
    return(jointMind)

def normalizeValuesPdSeries(pandasSeries):
    totalSum = sum(pandasSeries)
    probabilities = pandasSeries.groupby(pandasSeries.index.names).apply(lambda x: x/totalSum)
    return(probabilities)
    
# P(w) component of the mind - uniform distribution
def getWorldProbabiltiy_Uniform(worldSpace):
    worldDict = {'worlds': worldSpace}
    worldSpaceDF = getMultiIndexMindSpace(worldDict)
    worldSpaceDF = worldSpaceDF.loc[~worldSpaceDF.index.duplicated(keep='first')]

    numberOfWorlds = len(worldSpace)
    getConditionWorldProbabiltiy = lambda x: 1.0/numberOfWorlds
    worldProbabilities = worldSpaceDF.groupby(worldSpaceDF.index.names).apply(getConditionWorldProbabiltiy)
    worldSpaceDF['p(w)'] = worldSpaceDF.index.get_level_values(0).map(normalizeValuesPdSeries(worldProbabilities).get)
    return(worldSpaceDF)

# p(d) component of the mind - uniform distribution
def getDesireProbability_Uniform(desireSpace):
    desireDict = {'desires': desireSpace}
    desireSpaceDF = getMultiIndexMindSpace(desireDict)

    getConditionDesireProbability = lambda x: 1.0/len(desireSpace)
    desireProbabilities = desireSpaceDF.groupby(desireSpaceDF.index.names).apply(getConditionDesireProbability)
    desireSpaceDF['p(d)'] = desireSpaceDF.index.get_level_values(0).map(normalizeValuesPdSeries(desireProbabilities).get)
    return(desireSpaceDF)

#p(g|w,d) component of the mind

#single goal case - bananas in boxes Misyak
def getGoalGivenWorldAndDesire_SingleGoal(goalSpace, world, desire):
    goalDict = {'goals': goalSpace}
    goalSpaceDF = getMultiIndexMindSpace(goalDict)
    
    goalProbabilities = goalSpaceDF.groupby(goalSpaceDF.index.names).apply(lambda x: 1.0)
    goalSpaceDF['p(g|w,d)'] = goalSpaceDF.index.get_level_values(0).map(normalizeValuesPdSeries(goalProbabilities).get)
    return(goalSpaceDF)

#multiple goal - Grosse battery example, uniform
def getGoalGivenWorldAndDesire_Grosse(goalSpace, world, desire):
    goalDict = {'goals': goalSpace}
    goalSpaceDF = getMultiIndexMindSpace(goalDict)
    
    goalProbabilities = goalSpaceDF.groupby(goalSpaceDF.index.names).apply(lambda x: getConditionGoalPDF_Grosse(goal = x.index.get_level_values('goals')[0], world = world))
    goalSpaceDF['p(g|w,d)'] = goalSpaceDF.index.get_level_values(0).map(normalizeValuesPdSeries(goalProbabilities).get)
    return(goalSpaceDF)

def getConditionGoalPDF_Grosse(goal, world):
    if 'n' in world:
        assert world == 'n', 'incongruous world: cannot have neither battery and a battery value in the world'
    if 'n' in goal:
        assert goal == 'n', "incongruous goal: cannot have both neither battery and a battery value as a goal"
        return(1)
    return(int(all(g in world for g in goal)))



# utility of action U(a,w,g) - miysak
"""
    costOfLocation: int/float or iterable. If int, a fixed cost associated with the number of actions taken. If iterable, the cost of each action in each location
    valueOfReward = int/float. The value of each reward an action taken receives
    costOfNonReward = int/float. The cost associated with each locaiton action that does not result in reward
"""
class ActionUtility(object):
    def __init__(self, costOfLocation, valueOfReward, costOfNonReward):
        self.costOfLocation = costOfLocation
        self.valueOfReward = valueOfReward
        self.costOfNonReward = costOfNonReward

    def __call__(self, action, world, goal=None):
        numberOfLocationsInWorld = len(world)
        locationActionCost = self.getLocationCostList(numberOfLocationsInWorld)
        locationRewardValue = [self.valueOfReward if location == 1 else 0.0 for location in world]
        locationNonRewardCost = [-abs(self.costOfNonReward) if location == 0 else 0.0 for location in world]

        totalLocationValue = [sum((costAct,costNoReward,valueReward)) for costAct, costNoReward, valueReward in zip(locationActionCost, locationNonRewardCost, locationRewardValue)]
        utilityOfAction = [actionValue if action == 1 else 0 for actionValue, action in zip(totalLocationValue,action)]
        return(sum(utilityOfAction))

    def getLocationCostList(self, numberLocations):
        if (type(self.costOfLocation) is int) or (type(self.costOfLocation) is float):
            locationCost = [-abs(self.costOfLocation)]*numberLocations
        else:
            assert len(self.costOfLocation) == numberLocations, "Location cost must be either an int/float or iterable of world length"
            locationCost = [-abs(locCost) for locCost in self.costOfLocation]
        return(locationCost)

# Grosse action utility - multiple agent actions
"""
    costOfLocation: list of dictionaries: list indices indicate agents [signaler, receiver], dictionaries indicate cost of actions {action key: action cost scalar}
    valueOfReward: scalar reward value for achieving each component of the intended goal
    nullAction: the representation of a null action (default = 'n')
"""
class ActionUtility_Grosse(object):
    def __init__(self, costOfLocation, valueOfReward, nullAction = 'n'):
        self.costOfLocation = costOfLocation
        self.valueOfReward = valueOfReward
        self.nullAction = nullAction

    def __call__(self, action, world, goal):
        assert self.isActionCongruous(action, world), 'action is not possible in this world'
        jointCost  = self.getActionCost(action)
        rewardAmount = self.getReward(action, goal)
        totalUtility = jointCost + rewardAmount
        return(totalUtility)
            
    def isActionCongruous(self, action, world):
        areActionsPossible = [agentAction in world for agentAction in action if agentAction != self.nullAction]
        return(all(areActionsPossible))
    
    #joint cost of action for all agents
    def getActionCost(self, action):
        signalerAction = action[0]
        signalerCost = -abs(self.costOfLocation[0][signalerAction])
        receiverAction = action[1]
        receiverCost = -abs(self.costOfLocation[1][receiverAction])
        jointActionCost = signalerCost + receiverCost
        return(jointActionCost)

    #total reward of action
    def getReward(self, action, goal):
        if goal == 'n':
            return(0)
        goalList = list(goal)
        reward = 0
        signalerAction = action[0]
        receiverAction = action[1]
        
        if signalerAction in goalList:
            reward += self.valueOfReward
            goalList.remove(signalerAction)
        if receiverAction in goalList:
            reward += self.valueOfReward
            goalList.remove(receiverAction)
        return(reward)



#p(a|w,g) component of the mind - Miysak
"""
    alpha: scalar rationality constant
    actionUtilityFunction: function that takes in (action, world, goal) and returns a scalar utlity
    softmax: boolean; False indicates strict maximization
"""
class ActionDistributionGivenWorldGoal(object):
    def __init__(self, alpha, actionUtilityFunction, softmax=False):
        self.alpha = alpha
        self.getUtilityOfAction = actionUtilityFunction
        self.softmax = softmax
        
    def __call__(self, actionSpace, world, goal):
    	#create a dataframe with indices actions
        actionSpaceDF = getMultiIndexMindSpace({'actions':actionSpace})

        #for each action, get the utility given goal, world; transform this into an action distribution
        getConditionActionUtility = lambda x: np.exp(self.alpha*self.getUtilityOfAction(x.index.get_level_values("actions")[0], world))
        utilities = actionSpaceDF.groupby(actionSpaceDF.index.names).apply(getConditionActionUtility)
    
    	#keep as softmax pdf or transform to strict maximization, normalize
        if self.softmax:         
            probabilities = normalizeValuesPdSeries(utilities)
        else:
            maxUtility = max(utilities)
            numberOfOccurances = utilities.value_counts().loc[maxUtility]
            getConditionProbability = lambda x: 1.0/numberOfOccurances if x == maxUtility else 0 
            probabilities = utilities.apply(getConditionProbability)
        
        actionSpaceDF['p(a|w,g)'] = actionSpaceDF.index.get_level_values(0).map(probabilities.get)
        return(actionSpaceDF)

#p(a|w,g) component of the mind - Grosse
"""
    alpha: scalar rationality constant
    actionUtilityFunction: function that takes in (action, world, goal) and returns a scalar utlity
    softmax: boolean; False indicates strict maximization
"""
# Inputs in the callable: actions as tuples indicating agent actions; i.e. (signaler action, receiver action)
class ActionDistributionGivenWorldGoal_Grosse(object):
    def __init__(self, alpha, actionUtilityFunction, softmax=False):
        self.alpha = alpha
        self.getUtilityOfAction = actionUtilityFunction
        self.softmax = softmax
        
    def __call__(self, actionSpace, world, goal):
        #create a dataframe with indices actions
        actionSpaceDF = getMultiIndexMindSpace({'actions':actionSpace})

        #for each action, get the utility given goal, world; transform this into an action distribution
        getConditionActionUtility = lambda x: np.exp(self.alpha*self.getUtilityOfAction(x.index.get_level_values('actions')[0], world, goal))
        utilities = actionSpaceDF.groupby(actionSpaceDF.index.names).apply(getConditionActionUtility)
        #keep as softmax pdf or transform to strict maximization
        if self.softmax:         
            probabilities = normalizeValuesPdSeries(utilities)
        else:
            maxUtility = max(utilities)
            numberOfOccurances = utilities.value_counts().loc[maxUtility]
            getConditionProbability = lambda x: 1.0/numberOfOccurances if x == maxUtility else 0 
            probabilities = utilities.apply(getConditionProbability)
            
        actionSpaceDF['p(a|w,g)'] = actionSpaceDF.index.get_level_values(0).map(probabilities.get)
        return(actionSpaceDF)
    

#mind generation (pdf over set of possible target minds)- specify each mind component function and combine for the full mind distribution
class GenerateMind(object):
    def __init__(self, getWorldProbability, getDesireProbability, getGoalProbability, getActionProbability):
        self.getWorldProbability = getWorldProbability
        self.getDesireProbability = getDesireProbability
        self.getGoalProbability = getGoalProbability
        self.getActionProbability = getActionProbability
        self.mindProbabilityLabel = 'p(mind)'
        
    def __call__(self, mindSpaceDictionary):
        jointMindSpace = getMultiIndexMindSpace(mindSpaceDictionary)
        getMindForCondition = lambda x: self.getConditionMind(x, mindSpaceDictionary)
        mindProbabilitySeries = jointMindSpace.groupby(jointMindSpace.index.names).apply(lambda x: self.getConditionMind(x, mindSpaceDictionary))
        mindProbability = pd.DataFrame(normalizeValuesPdSeries(mindProbabilitySeries))
        mindProbability.rename(columns={0 : self.mindProbabilityLabel}, inplace=True)
        return(mindProbability)
    
    def getConditionMind(self, oneMindCondition, mindSpace):
        world = oneMindCondition.index.get_level_values('worlds')[0]
        desire = oneMindCondition.index.get_level_values('desires')[0]
        goal = oneMindCondition.index.get_level_values('goals')[0]
        action = oneMindCondition.index.get_level_values('actions')[0]
        
        worldSpace = mindSpace['worlds']
        worldPDF = self.getWorldProbability(worldSpace)
        
        desireSpace = mindSpace['desires']
        desirePDF = self.getDesireProbability(desireSpace)
        
        goalSpace = mindSpace['goals']
        conditionalGoalPDF = self.getGoalProbability(goalSpace, world, desire)

        actionSpace = mindSpace['actions']
        conditionalActionPDF = self.getActionProbability(actionSpace, world, goal)

        mindProbability = worldPDF.loc[world].values[0]*desirePDF.loc[desire].values[0]*conditionalGoalPDF.loc[goal].values[0]*conditionalActionPDF.loc[action].values[0]
        
        if type(mindProbability) != float:
            mindProbability = mindProbability[0]
        #mindProbability = worldPDF.loc[world, 'p(w)'].values[0]*desirePDF.loc[desire, 'p(d)'].values[0]*conditionalGoalPDF.loc[goal, 'p(g|w,d)'].values[0]*conditionalActionPDF.loc[action, 'p(a|w,g)'].values[0]
        return(mindProbability)	