def normalizeDictionary(unnormalizedDictionary, nested = False):
    if nested:
        totalSum = sum([sum(innerDict.values()) for outerKey, innerDict in unnormalizedDictionary.items()])
        normalizedDictionary = {outerKey : {innerKey : probability/totalSum for innerKey, probability in innerDict.items()}
                            for outerKey, innerDict in unnormalizedDictionary.items()}
    else:
        totalSum = sum(unnormalizedDictionary.values())
        normalizedDictionary = {originalKey: val/totalSum for originalKey, val in unnormalizedDictionary.items()}
    return(normalizedDictionary)

def removeZeroes(d):
    for key in d.copy():
        if type(d[key]) == dict:
            removeZeroes(d[key])
        elif d[key]== 0:
            del d[key]
    return(d)

def cleanDictionary(d): #remove 0s and empty dictionaries
    noZeroD = removeZeroes(d)
    for key in noZeroD.copy(): 
        if not noZeroD[key]:
            del noZeroD[key]
    return(noZeroD)

def viewDictionaryStructure(d, dictionaryType = "wap", indent=0):
    if dictionaryType == "wap":
        levels  = ["world", "action", "probability"]
    if dictionaryType == "wasp":
        levels  = ["world", "action", "signal type", "probability"]

    for key, value in d.items():
        print('\t' * indent + str(levels[indent]) + ": "+ str(key))
        if isinstance(value, dict):
            viewDictionaryStructure(value, dictionaryType, indent+1)
        else:
            print('\t' * (indent+1) + str(levels[indent+1])+ ": " + str(value))