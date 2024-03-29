from __future__ import  division
import os
import copy
import debug
import struct
import logging
import tempfile
import subprocess
import numpy as np

class Wav:
    """
    Intended as a sort of 'struct' to hold WAV file data.

    Use the sound.load function to load a WAV file.
    """
    @debug.trace(logging.DEBUG)
    def __init__(self, numChannels=None, sampleRate=None, sampleData=None):
        self.num_channels = numChannels
        self.sample_rate = sampleRate
        self.sample_data = sampleData

@debug.trace(logging.DEBUG)
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

@debug.trace(logging.DEBUG)
def _load_wav(filename):
    """
    Helper for sound.load. Needed number of samples from wavfile, which scypi.io.wavfile doesn't
    return. Please use the sound.load function for public file reading.
    """
    with open(filename, 'rb') as wav:
        if wav.read(4) != b'RIFF':
            raise ValueError('Not a WAV file, no \'RIFF\' letters in header.')

        # Add 8 for the 8 bytes just read.
        file_size = struct.unpack('I', wav.read(4))[0] + 8

        if wav.read(4) != b'WAVE':
            raise ValueError('Not a WAV file, no \'WAVE\' letters in header.')

        if wav.read(4) != b'fmt ':
            raise ValueError('Not a wav file, no \'fmt \' letters in format subchunk.')

        fmt_chunk = struct.unpack('IHHIIHH', wav.read(20))
        size, audio_format, num_channels, sample_rate, _, _, bits_per_sample = fmt_chunk

        if audio_format != 1:
            raise ValueError('Compressed WAV file, cannot read.')

        if wav.read(4) != b'data':
            raise ValueError('Not a WAV file, no \'data\' letters in data subchunk.')

        # Size of the rest of the file, the PCM data.
        data_size = struct.unpack('I', wav.read(4))[0]
        samples = np.asarray(np.fromfile(wav, dtype='int8', count=data_size), dtype='float')

        return Wav(num_channels, sample_rate, samples)

@debug.trace(logging.DEBUG)
def flatten_to_mono(wav):
    """
    Flattens the PCM data in wav (a Wav class instance) to mono if it's not mono already.

    Does not modify wav, returns a new Wav instance.
    """
    num_channels = wav.num_channels
    if num_channels == 1:
        return wav

    sample_rate = wav.sample_rate
    num_frames = int(len(wav.sample_data) / num_channels)
    samples = np.fromiter((np.sum(wav.sample_data[i * num_channels:i * num_channels + num_channels]) for i in range(num_frames)), dtype='float')

    return Wav(1, sample_rate, samples)

@debug.trace(logging.DEBUG)
def normalize(wav):
    if wav.num_channels > 1:
        raise ValueError("Must be a mono wav file, flatten the channels first.")

    samples = wav.sample_data[:]
    peakVal = float(np.amax(np.absolute(samples)))

    if peakVal == 0:
        raise ValueError('Wav file is completely silent, nothing to normalize!')

    # Clip to range [-1, 1] to manage floating point errors.
    samples = np.clip(samples / peakVal, -1.0, 1.0)
    return Wav(wav.num_channels, wav.sample_rate, samples)

@debug.trace(logging.DEBUG)
def _h(x):
	return (x + abs(x)) / 2

@debug.trace(logging.DEBUG)
def spectral_difference(n, stft):
	abs_stft = np.absolute(stft(n))
	sd = _h(abs_stft(n) - abs_stft(n-1))**2
	return np.sum(sd[len(sd)/2 - 1:]) / N
	
def stft(x, fs, framesz, hop):
    framesamp = int(framesz*fs)
    hopsamp = int(hop*fs)
    w = scipy.hamming(framesamp)
    X = scipy.array([scipy.fft(w*x[i:i+framesamp]) 
                     for i in range(0, len(x)-framesamp, hopsamp)])
    return X

import unittest

class SoundTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(self.which("lame"), "LAME executable not found, need to install for MP3 conversion.")
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

    def test_load_noise_noErrors(self):
        sample_rate = 44100
        num_channels = 2
        duration = 1
        num_samples = sample_rate * duration
        data_size = num_samples * num_channels
        filepath = 'noise.wav'
        hiss = SoundTest.noise(sample_rate, duration, num_channels, filepath)
        w = load(filepath)
        self.assertEqual(sample_rate, w.sample_rate)
        self.assertEqual(num_channels, w.num_channels)
        self.assertEqual(data_size, w.sample_data.size)
        self.assertTrue(np.array_equal(hiss, w.sample_data))

    def test_flattenToMono_arrayOfOnes_sumEqualsFrameWidth(self):
        channels = 4
        rate = 44100
        num_frames = 10
        num_samples = num_frames * channels
        samples = np.ones(shape=(num_samples,), dtype='float')

        w = Wav(channels, rate, samples)
        m = flatten_to_mono(w)

        self.assertEquals(m.num_channels, 1)
        self.assertEquals(m.sample_rate, rate)
        self.assertEquals(len(m.sample_data), num_frames)
        self.assertTrue(np.array_equal(m.sample_data, channels * w.sample_data[::channels]))

    def test_flattenToMono_arrayOfZeros_sumStillZero(self):
        channels = 4
        rate = 44100
        num_frames = 10
        num_samples = num_frames * channels
        samples = np.zeros(shape=(num_samples,), dtype='float')

        w = Wav(channels, rate, samples)
        m = flatten_to_mono(w)

        self.assertEquals(m.num_channels, 1)
        self.assertEquals(m.sample_rate, rate)
        self.assertEquals(len(m.sample_data), num_frames)
        self.assertTrue(np.array_equal(m.sample_data, w.sample_data[::channels]))

    def test_flattenToMono_arrayAranged_equalsSumOfRange(self):
        channels = 4
        rate = 44100
        num_frames = 10
        num_samples = num_frames * channels
        samples = np.array([np.arange(channels) for i in range(num_frames)], dtype='float').flatten()

        w = Wav(channels, rate, samples)
        m = flatten_to_mono(w)

        self.assertEquals(m.num_channels, 1)
        self.assertEquals(m.sample_rate, rate)
        self.assertEquals(len(m.sample_data), num_frames)
        compare = sum(range(channels)) * np.ones(shape=(len(w.sample_data[::channels]),), dtype='float')
        self.assertTrue(np.array_equal(m.sample_data, compare))

    def test_normalize_nonMonoChannelWav_throwsValueError(self):
        w = Wav(2, 41100, np.ones(shape=(4096,), dtype='float'))
        self.assertRaises(ValueError, normalize, w)

    def test_normalize_arrayOfZeros_throwsValueError(self):
        w = Wav(1, 41100, np.zeros(shape=(4096,), dtype='float'))
        self.assertRaises(ValueError, normalize, w)

    def test_normalize_arrayOfOnes_unchanged(self):
        w = Wav(1, 44100, np.ones(shape=(4096,), dtype='float'))
        n = normalize(w)
        self.assertEqual(1, n.num_channels)
        self.assertEqual(44100, n.sample_rate)
        self.assertTrue(np.array_equal(n.sample_data, np.ones(shape=(w.sample_data.size,), dtype='float')))

    def test_normalize_noise_range0to1Afterwards(self):
        hiss = SoundTest.noise()
        w = Wav(1, 44100, hiss)
        n = normalize(w)
        self.assertEqual(1, n.num_channels)
        self.assertEqual(44100, n.sample_rate)
        self.assertEqual(w.sample_data.size, n.sample_data.size)
        self.assertTrue(np.all([abs(x) <= 1 for x in n.sample_data]))

    # Helper Methods
    @staticmethod
    def which(executable_path):
        def is_exe(filepath):
            return os.path.isfile(filepath) and os.access(filepath, os.X_OK)

        path, name = os.path.split(executable_path)
        if path and is_exe(executable_path):
            return executable_path
        else:
            for path in os.environ["PATH"].split(os.pathsep):
                path = path.strip()
                exe_file = os.path.join(path, name)
                if is_exe(exe_file):
                    return exe_file

        return None

    # Returns white noise hiss as numpy array.
    # frequency -- in Hz.
    # sample_rate -- in Hz.
    # coefficient -- i.e. a * sin(). float.
    # duration -- in seconds.
    # num_channels -- number of audio channels.
    # filepath -- if a filepath, samples will be saved filepath as a WAV file.
    #             if None (default), no file output.
    #             Returns samples in either case.
    @staticmethod
    def noise(sample_rate=44100, duration=4, num_channels=2, filepath=None):
        num_samples = sample_rate * duration
        hiss = np.fromstring(np.random.bytes(num_samples * num_channels), dtype='int8')

        if filepath is not None:
            import wave
            bytes_per_sample = 2 # 16 bits = 2 bytes
            f = wave.open(filepath, 'w')
            f.setparams((num_channels, bytes_per_sample, sample_rate, num_samples, 'NONE', 'not compressed'))
            f.writeframes(hiss.tostring())
            f.close()
        
        return hiss



if __name__ == '__main__':
    import sys
    import optparse
    opt_parser = optparse.OptionParser()
    opt_parser.add_option('-v', '--verbose', dest='verbose', action='count', help='Increase verbosity, -v < -vv < -vvv.')
    opts, args = opt_parser.parse_args()

    log_level = logging.WARNING
    if opts.verbose == 1:
        log_level = logging.INFO
    elif opts.verbose >= 2:
        log_level = logging.DEBUG

    debug.configure(level=log_level)

    unittest.main()

