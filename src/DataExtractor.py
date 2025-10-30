import numpy as np
import os


class DataExtractor():

    # class used to store an image in a usable way
    class imageData():
        # define the size of the images to be used
        image_size = np.array((32,32))

        def __init__(self, colour_data, filename="", classification=None):
            r,g,b = self.filterColourData(colour_data)
            self.red_channel = self.loadDataIntoArray(r).astype(np.uint8)
            self.green_channel = self.loadDataIntoArray(g).astype(np.uint8)
            self.blue_channel = self.loadDataIntoArray(b).astype(np.uint8)

            self.full_image = np.dstack((self.red_channel, self.green_channel, self.blue_channel))

            self.filename = filename
            self.classification = classification

        def filterColourData(self, colour_data):
            r = colour_data[0:1024]
            g = colour_data[1024:2048]
            b = colour_data[2048:]
            return r,g,b
        
        # turns the data from a 1D array to a 2D array 
        def loadDataIntoArray(self, data_array:np.array):
            output_array = data_array.reshape(self.image_size)
            return output_array
        
        def getImage(self):
            return self.full_image
        
        def getChannel(self, channel):
            if channel == 'r': return self.red_channel
            if channel == 'g': return self.green_channel
            if channel == 'b': return self.blue_channel
            return None
        
        def getClassification(self):
            return self.classification
        
        def getFilename(self):
            return self.filename

    
    # Gets the raw data from the file
    # Credit: Alex Krizhevsky
    # This code is taken from the CIFAR-10 dataset readme. It is used to load the images into a dict from the picked files. 
    def getUnpickled(self, filename:str):
        import pickle
        with open(filename, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    # read all data from the unpickled file and load it into an array of image
    # will default to "truck, ship, horse, cat, bird" if no other classes are given
    def readData(self, data_batch=1, working_classes = [9,8,7,3,2], test=False):
        if test:
            filename = "test_batch"
        else: 
            filename = "data_batch_" + str(data_batch)
        
        dir_path = os.path.dirname(os.path.realpath(__file__)) + "/dataset/" + filename
        

        data_dict = self.getUnpickled(dir_path)
        
        if data_dict == None:
            return
        

        images = []
        
        for i in range(len(data_dict[b'data'])):
            if data_dict[b'labels'][i]  in working_classes:
                images.append(self.imageData(data_dict[b'data'][i],
                                            data_dict[b'filenames'][i],
                                            data_dict[b'labels'][i]))
                
        return images
