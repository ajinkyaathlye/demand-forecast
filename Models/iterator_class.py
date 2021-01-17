import numpy as np

class Sanskriti_Data_Reader():

    # Load npz data into inputs and targets variables. Init the total number of batches (count) based on the batch size.
    def __init__(self, data, batch_size=None):

        npz = np.load('../Data/NPZ_data/Sanskriti_{0}_data.npz'.format(data))

        self.inputs, self.targets = npz['inputs'].astype(np.float64), npz['targets'].astype(np.float64)

        # counts number of batches based on given size
        if batch_size is None:
            self.batch_size = self.inputs.shape[0]
        else:
            self.batch_size = batch_size

        self.current_batch = 0
        self.batch_count = self.inputs.shape[0] // self.batch_size



    # function that loads the next batch.
    # Check whether the batch count (current batch you are on) is > number of batches. If yes, stop iteration.
    def next(self):
        if self.current_batch >= self.batch_count:
            self.current_batch = 0
            raise StopIteration()
        # Get next batch slice by returning the input and targets for the given batch (ex. from 1-10, or 11-20 etc)

        batch_slice = slice(self.current_batch * self.batch_size, (self.current_batch + 1) * self.batch_size)
        input_batch = self.inputs[batch_slice]
        target_batch = self.targets[batch_slice]
        self.current_batch += 1

        # The function will return the inputs batch
        return self.current_batch, input_batch, target_batch

    def __iter__(self):
        return self
