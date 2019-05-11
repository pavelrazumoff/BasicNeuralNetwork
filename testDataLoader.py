# -----------------------------------------------------------------------------
# Helper to load test data from image files.
# Source code is a property of https://github.com/makeyourownneuralnetwork.
# -----------------------------------------------------------------------------
import imageio
# glob helps select multiple files using patterns.
import glob
import numpy

def loadTestImageData(filepathPattern):
    our_own_dataset = []

    for image_file_name in glob.glob(filepathPattern):
        print("loading ...", image_file_name)
        # use the filename to set the correct label
        label = int(image_file_name[-5:-4])
        # load image data from files into an array
        img_array = imageio.imread(image_file_name, as_gray=True)
        # reshape from 28x28 to list of 784 values, invert values
        img_data = 255.0 - img_array.reshape(784)
        # then scale data to range from 0.01 to 1.0
        img_data = (img_data / 255.0 * 0.99) + 0.01

        #append label and image data to test data set
        record = numpy.append(label, img_data)
        our_own_dataset.append(record)
    
    return our_own_dataset