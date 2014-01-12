import os
import copy
import struct
import tempfile
import subprocess
import numpy as np

class Wav:
    """
    Intended as a sort of 'struct' to hold WAV file data.

    Use the sound.load function to load a WAV file.
    """
    def __init__(self, numChannels=None, sampleRate=None, sampleData=None):
        self.num_channels = numChannels
        self.sample_rate = sampleRate
        self.sample_data = sampleData

def load(filepath):
    """
    Loads a .wav or .mp3 file from the given filepath.

    Returns instance of the Wav class.
    """

    error_message = 'filepath was ' + repr(filepath) + ', {0}'
    if filepath is None:
        raise ValueError(error_message.format('cannot be None.'))
    if filepath.strip() == '':
        raise ValueError(error_message.format('cannot be empty or whitespace.'))
    if os.path.isdir(filepath):
        raise ValueError(error_message.format('cannot be a directory.'))
    if not os.path.exists(filepath):
        raise ValueError(error_message.format('file does not exist.'))

    filename, extension = os.path.splitext(filepath)
    if extension.lower() == '.wav':
        return _load_wav(filepath)
    elif extension.lower() == '.mp3':
        with tempfile.TemporaryFile() as temp, open(os.devnull, 'w') as null:
            cmd = ['lame', '--decode', filepath, temp.name]
            if subprocess.call(cmd, stdout=null, stderr=null) != 0:
                raise ValueError(error_message.format('LAME could not read that .mp3 file.'))
        return _load_wav(temp.name)
    else:
        raise NotImplementedError(error_message, 'file must have \'.wav\' or \'.mp3\' extension.')

def _load_wav(filename):
    """
    Helper for sound.load. Needed number of samples from wavfile, which scypi.io.wavfile doesn't
    return. Please use the sound.load function for public file reading.
    """

    with open(filename, 'rb') as wav:
        riff_letters = wav.read(4)
        if riff_letters != 'RIFF':
            raise ValueError('Not a WAV file, no \'RIFF\' letters in header.')

        # Chunk size
        garbage = wav.read(4)

        wave_letters = wav.read(4)
        if wave_letters != 'WAVE':
            raise ValueError('Not a WAV file, no \'WAVE\' letters in header.')

        fmt_letters = wav.read(4)
        if fmt_letters != 'fmt ':
            raise ValueError('Not a wav file, no \'fmt \' letters in format subchunk.')

        # Format chunk size.
        garbage = wav.read(4)

        audio_format = wav.read(2)
        if audio_format != 1:
            ValueError('Compressed WAV file, cannot read.')

        num_channels = struct.unpack('h', wav.read(2))[0]
        sample_rate = struct.unpack('i', wav.read(4))[0]
        
        # 4 bytes (uint32):
        #     byte_rate == sample_rate * num_channels * bits_per_sample / 8
        # 2 bytes (uint16):
        #     block_align == num_channels * bits_per_sample / 8
        #     (num bytes for one frame, which is a sample that includes all channels)
        # 2 bytes (uint16):
        #     bits_per_sample == num bits in one sample (assumed to be 8)
        garbage = wav.read(4 + 2 + 2)

        data_letters = wav.read(4)
        if data_letters != 'data':
            raise ValueError('Not a WAV file, no \'data\' letters in data subchunk.')

        # Size of the rest of the file, the PCM data.
        data_size = struct.unpack('i', wav.read(4))[0]
        samples = np.fromstring(wav.read(data_size), dtype='int8')

        return Wav(num_channels, sample_rate, samples)

def flatten_to_mono(wav):
    """
    Flattens the PCM data in wav (a Wav class instance) to mono if it's not mono already.

    Does not modify wav, returns a new Wav instance.
    """

    num_channels = wav.num_channels
    if num_channels == 1:
        return wav

    sample_rate = wav.sample_rate
    num_frames = len(wav.sample_data) / num_channels
    samples = np.fromiter((np.sum(wav.sample_data[i * num_channels:i * num_channels + num_channels]) for i in range(num_frames)), dtype='int8')

    return Wav(1, sample_rate, samples)



import unittest

class SoundTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_load_filepathNone_throwsValueError(self):
        self.assertRaises(ValueError, load, None)

    def test_load_filepathEmpty_throwsValueError(self):
        self.assertRaises(ValueError, load, '')

    def test_load_filepathWhitespace_throwsValueError(self):
        self.assertRaises(ValueError, load, ' \t\r\n')

    def test_load_filepathDirectory_throwsValueError(self):
        self.assertRaises(ValueError, load, os.getcwd())

    def test_load_filepathNotExist_throwsValueError(self):
        self.assertRaises(ValueError, load, '/name/of/a/file/that/will/never/exist.wav')

    def test_load_filepathNotWavOrMp3Extension_throwsValueError(self):
        self.assertRaises(NotImplementedError, load, os.path.abspath(__file__))

    def test_load_filepathWavExtensionNotWavFile_throwsValueError(self):
        with tempfile.NamedTemporaryFile(suffix='.wav') as f:
            self.assertRaises(ValueError, load, f.name)

    def test_load_filepathMp3ExtensionNotMp3File_throwsValueError(self):
        with tempfile.NamedTemporaryFile(suffix='.mp3') as f:
            self.assertRaises(ValueError, load, f.name)

    def test_flattenToMono_arrayOfOnes_sumEqualsFrameWidth(self):
        channels = 2
        rate = 44100
        num_frames = 10
        num_samples = num_frames * channels
        samples = np.ones(shape=(num_samples,), dtype='int8')

        w = Wav(channels, rate, samples)
        m = flatten_to_mono(w)

        self.assertEquals(m.num_channels, 1)
        self.assertEquals(m.sample_rate, rate)
        self.assertEquals(len(m.sample_data), num_frames)
        self.assertTrue(np.array_equal(m.sample_data, channels * w.sample_data[::channels]))



if __name__ == '__main__':
    unittest.main()
    
