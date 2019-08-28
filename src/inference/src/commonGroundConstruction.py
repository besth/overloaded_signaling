import itertools

def getWorldSpace(wall, nBoxes, nRewards): #omega world, list output
    if wall:
        possibleWorlds = [w for w in itertools.product([1,0], repeat = nBoxes)] 
    else:
        possibleWorlds = [w for w in itertools.product([1,0], repeat = nBoxes) if list(w).count(1) == nRewards]
    return(possibleWorlds)

def getActionSpace(nBoxes, nReceiverChoices): #omega actions, list output
    possibleActions = [a for a in itertools.product([1,0], repeat = nBoxes) 
                       if sum(a) <= nReceiverChoices]
    return(possibleActions)

def getSignalSpace(nBoxes, nSignals): #omega singals, list output
    all_utterances = [c for c in itertools.product([1,0], repeat = nBoxes) 
                      if (sum(c) <= nSignals)] 
    return(all_utterances)

