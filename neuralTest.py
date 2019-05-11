# -----------------------------------------------------------------------------
# Main source file to run neural network.
# If yoy want to use MNIST dataset to be able to train and test your network
# go here https://pjreddie.com/projects/mnist-in-csv/.
# -----------------------------------------------------------------------------
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import time
import pickle
import NeuralNetwork.neuralNetwork as nNetwork
import testDataLoader

# global variables.
gNeuralNetwork = None
base_path = Path(__file__).parent

def main():
    """ Call it to run main work loop. """
    print("Welcome to the testing neural network app!")

    while True:
        print("\nPlease, choose your option:")
        print("0. Load training data if it exist.")
        print("1. Train neural network with specific params.")
        print("2. Save training data if it exist.")
        print("3. Load tests and test neural network.")
        print("4. Run network backwards with given label.")
        print("5. Exit.")

        choose = input()
        handleAnswer = {
            0 : loadTrainingData,
            1 : trainNetwork,
            2 : saveTrainingData,
            3 : testNetwork,
            4 : testNetworkBackwards,
            5 : quit
        }

        if choose.isdigit():
            try:
                handleAnswer[int(choose)]()
            except KeyError as e:
                pass
        
def setTrainParams():
    """ Allows us to set up network init params before training. """

    trainParams = {
        "hidden_nodes" : 100,
        "learning_rate" : 0.2,
        "epochs" : 1
    }

    while True:
        print("\nPlease, set training params:")
        print("0. Set hidden nodes ( current value", trainParams["hidden_nodes"], ")")
        print("1. Set learning rate ( current value", trainParams["learning_rate"], ")")
        print("2. Set epochs ( current value", trainParams["epochs"], ")")
        print("3. Train network.")

        choose = int(input())

        if choose != 3:
            print("value: ", end="")
            x = input()
            
        if choose == 0:
            trainParams["hidden_nodes"] = int(x)
        elif choose == 1:
            trainParams["learning_rate"] = float(x)
        elif choose == 2:
            trainParams["epochs"] = int(x)
        elif choose == 3:
            return trainParams

def trainNetwork():
    """ Sets train params, creates new neural network object, loads train data
    from MNIST and trains network. """

    trainParams = setTrainParams()
    print("training...")
    
    input_nodes = 784
    hidden_nodes = trainParams["hidden_nodes"]
    output_nodes = 10

    learning_rate = trainParams["learning_rate"]

    global gNeuralNetwork
    gNeuralNetwork = nNetwork.neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # get absolute file path.
    file_path = (base_path / "mnist_dataset/mnist_train.csv").resolve()

    # load test MNIST data set
    try:
        training_data_file = open(file_path, "r")
    except FileNotFoundError:
        print("There isn't data to train with.")
        return
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    # neural network training process.
    epochs = trainParams["epochs"]

    start = time.time()

    for e in range(epochs):
        for record in training_data_list:
            all_values = record.split(",")
            # scaling input values to be able to train neural network.
            # appropriate range is from 0.01 to 0.99.
            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

            # create target output values (all equal to 0.01 except of desired marker value = 0.99)
            targets = np.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99

            gNeuralNetwork.train(inputs, targets)
            # uncomment this to be able to rotate current record to increase training complexity.
            """
            # create rotated variations.
            # rotated anticlockwize by x degrees.
            inputs_plusx_img = ndimage.interpolation.rotate(inputs.reshape(28, 28), 10, cval=0.01,
                order=1, reshape=False)
            gNeuralNetwork.train(inputs_plusx_img.reshape(784), targets)
            # rotated clockwise by x degrees.
            inputs_minusx_img = ndimage.interpolation.rotate(inputs.reshape(28, 28), -10, cval=0.01,
                order=1, reshape=False)
            gNeuralNetwork.train(inputs_minusx_img.reshape(784), targets)"""
    
    gNeuralNetwork.lastTrainEpochs = epochs
    end = time.time()

    print("training time:", end - start, "ms.")
    print("trained successfully")

def testNetwork():
    """ Tests neural network with test data (images or MNIST .csv file). """

    # uncomment this if you want to test with MNIST.
    """
    file_path = (base_path / "mnist_dataset/mnist_test.csv").resolve()
    test_data_file = open(file_path, "r")
    test_data_list = test_data_file.readlines()
    test_data_file.close()
    """

    print("testing")
    
    if gNeuralNetwork == None:
        print("You have to initialize neural network first (load existing data or train from scratch)")
        return

    file_path = (base_path / "images/2828_test_*[0-9].png").__str__()
    test_data_list = testDataLoader.loadTestImageData(file_path)

    scorecard = []

    for record in test_data_list:
        correct_label = int(record[0])

        # uncomment this if you want to test with MNIST.
        """
        # scaling input values to be able to train neural network.
        # appropriate range is from 0.01 to 0.99.
        #all_values = record.split(",")
        #inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        """
        inputs = record[1:]
        outputs = gNeuralNetwork.query(inputs)[1]

        label = np.argmax(outputs)

        print(outputs)
        print("network response:", label)
        print("correct value:", correct_label)

        if label == correct_label:
            scorecard.append(1)
        else:
            scorecard.append(0)

    print("\nscorecard:", scorecard)

    scorecard_array = np.asarray(scorecard)
    print("efficiency =", scorecard_array.sum() / scorecard_array.size)

def testNetworkBackwards():
    """ Allows us to look at how the neural network sees the number itself by passing label (0-9). """

    if gNeuralNetwork == None:
        print("You have to initialize neural network first (load existing data or train from scratch)")
        return

    while(True):
        print("\nPlease, enter the label: ", end="")
        label = input()

        if label.isdigit():
            break
    
    targets = np.zeros(gNeuralNetwork.onodes) + 0.01
    targets[int(label)] = 0.99

    image_data = gNeuralNetwork.backquery(targets)

    plt.imshow(image_data.reshape(28, 28), cmap="Greys", interpolation="None")
    plt.show()

def loadTrainingData():
    """ Loads previously trained neural network data to bypass long training process. """

    print("loading training data")

    file_path = (base_path / "neuralData.pkl").resolve()
    with open(file_path, "rb") as handle:
        data = pickle.load(handle)

    global gNeuralNetwork
    gNeuralNetwork = nNetwork.neuralNetwork(data["inodes"], data["hnodes"], data["onodes"], data["learningrate"],
        wih = data["wih"], who = data["who"])
    gNeuralNetwork.lastTrainEpochs = data["lastTrainEpochs"]

    print("loaded successfully\n\nNetwork data:")
    print("Input nodes:", gNeuralNetwork.inodes)
    print("Hidden nodes:", gNeuralNetwork.hnodes)
    print("Output nodes:", gNeuralNetwork.onodes)
    print("Learning rate:", gNeuralNetwork.lr)
    print("Last train epochs:", gNeuralNetwork.lastTrainEpochs)

def saveTrainingData():
    """ Saves neural network training data. """

    print("saving")

    if gNeuralNetwork == None:
        print("You have to initialize neural network first (load existing data or train from scratch)")
        return
    
    data = {
        "wih" : gNeuralNetwork.wih,
        "who" : gNeuralNetwork.who,
        "inodes" : gNeuralNetwork.inodes,
        "hnodes" : gNeuralNetwork.hnodes,
        "onodes" : gNeuralNetwork.onodes,
        "learningrate" : gNeuralNetwork.lr,
        "lastTrainEpochs" : gNeuralNetwork.lastTrainEpochs
    }

    file_path = (base_path / "neuralData.pkl").resolve()
    with open(file_path, "wb") as handle:
        pickle.dump(data, handle)
    print("saved successfully")

# -----------------------------------------------------------------------------
# Main func.
# -----------------------------------------------------------------------------
main()