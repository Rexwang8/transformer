import pickle




#unpickle loss results
def unpickleLossResults():
    with open('transformer/picklejarImproved/lossByEpoch.pickle', 'rb') as handle:
        lossResults = pickle.load(handle)
    return lossResults

#unpickle training times
def unpickleTrainingTimes():
    with open('transformer/picklejarImproved/timeTrainingByEpoch.pickle', 'rb') as handle:
        trainingTimes = pickle.load(handle)
    return trainingTimes

#unpickle total time
def unpickleTotalTime():
    with open('transformer/picklejarImproved/totalTime.pickle', 'rb') as handle:
        totalTime = pickle.load(handle)
    return totalTime

#unpickle inference results
def unpickleInferenceResults():
    with open('transformer/picklejarImproved/inferenceResults.pickle', 'rb') as handle:
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
    #label = "Loss: " + str(lossResults[-1])
    #plt.annotate(label, (data_x[-1], data_y[-1]))
    
    #legend that shows the train/test split
    plt.legend(["Train", "Test"], loc ="upper right")
    
    
    
    #title
    plt.title("Loss by Epoch")
    
    
    plt.savefig('transformer/lossResults.png')
    plt.show()
    
    
    
#plot training times
def plotTrainingTimes(trainingTimes):
    print(trainingTimes)
    data_y = trainingTimes
    data_x = range(len(trainingTimes))
    plt.plot(data_x, data_y)
    plt.ylabel('Training Time')
    plt.xlabel('Epoch')
    
    #title
    plt.title("Training Time by Epoch")
    
    #legend
    plt.legend(["Train", "Test"], loc ="upper right")
    
    
    plt.savefig('transformer/trainingTimes.png')
    plt.show()
    
    
    
plotLossResults(unpickleLossResults())
plotTrainingTimes(unpickleTrainingTimes())
print(f"Total time: {unpickleTotalTime()}")