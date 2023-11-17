import pickle




#unpickle loss results
def unpickleLossResults():
    with open('lossByEpoch.pickle', 'rb') as handle:
        lossResults = pickle.load(handle)
    return lossResults

#unpickle training times
def unpickleTrainingTimes():
    with open('timeTrainingByEpoch.pickle', 'rb') as handle:
        trainingTimes = pickle.load(handle)
    return trainingTimes

#unpickle total time
def unpickleTotalTime():
    with open('totalTime.pickle', 'rb') as handle:
        totalTime = pickle.load(handle)
    return totalTime

#unpickle inference results
def unpickleInferenceResults():
    with open('inferenceResults.pickle', 'rb') as handle:
        inferenceResults = pickle.load(handle)
    return inferenceResults

import matplotlib.pyplot as plt
#plot loss results
def plotLossResults(lossResults):
    print(lossResults)
    data_y = lossResults
    data_x = range(len(lossResults))
    plt.plot(data_x, data_y)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    
    plt.show()
    
#plot training times
def plotTrainingTimes(trainingTimes):
    plt.plot(trainingTimes)
    plt.ylabel('Training Time')
    plt.xlabel('Epoch')
    plt.show()
    
    
plotLossResults(unpickleLossResults())
plotTrainingTimes(unpickleTrainingTimes())
print(f"Total time: {unpickleTotalTime()}")