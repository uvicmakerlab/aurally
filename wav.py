import os
import numpy as np
from scipy.io import wavfile

class Wav:
    """
    Wrapper for operations on .wav files.
    """

    def __init__(self, filepath):
        """"
        Loads the .wav file from the given filepath.

        filepath should not be None, whitespace, or empty.
        """
        self.load(filepath)

    def load(self, filepath):
        """
        Loads the .wav file from the given filepath.

        filepath should not be None, whitespace, or empty.
        """

        assert filepath is not None, 'Filename cannot be None.'
        assert filepath.strip() != '', 'Filename cannot be empty or whitespace.'

        self.sample_rate, self.data = wavfile.read(filepath)

if __name__ == '__main__':
    help(Wav)
 
