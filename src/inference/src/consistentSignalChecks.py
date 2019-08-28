# Miysak - returns a boolean of whether the input signal is possible given the mind and signal category
def signalIsConsistent_Boxes(signal, mind, signalType, goToSignal = '1'):
    world = mind['worlds']
    signalTypeStr = str(signalType)
    
    if signalTypeStr == goToSignal:
        consistentSignals = [int(w == 1) for u, w in zip(signal, world) if u == 1]  
    else:
        consistentSignals = [int(w != 1) for u, w in zip(signal, world) if u == 1]
    return(consistentSignals.count(0) == 0)

# Grosse - returns a boolean of whether the input signal is possible given the mind (action and goal for Grosse) and signal category
def signalIsConsistent_Grosse(signal, mind, signalerType, nullSignal = 'n', receiverSignal = 'you'):
    action = mind['actions']
    goal = mind['goals']

    if signal == receiverSignal:
        #action must include the receiver (second position)
        isActionConsistent = (action[1] != nullSignal)
        isGoalConsistent = (goal != nullSignal)
    else:
        #action must NOT include the receiver
        isActionConsistent = (action[1] == nullSignal)
        isGoalConsistent = True
    return(all([isActionConsistent, isGoalConsistent]) )