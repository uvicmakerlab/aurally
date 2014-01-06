import os
import tempfile
import subprocess
import scipy.io.wavfile as wavfile

def load(filepath):
    """
    Loads a .wav or .mp3 file from the given filepath.

    Returns (sample rate, array of samples) tuple.
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
        return wavfile.read(filepath)
    elif extension.lower() == '.mp3':
        with tempfile.NamedTemporaryFile(suffix='.wav') as temp:
            with open(os.devnull, 'w') as fnull:
                cmd = ['lame', '--decode', filepath, temp.name]
                if subprocess.call(cmd, stdout=fnull, stderr=fnull) != 0:
                    raise ValueError(error_message.format('LAME could not read that .mp3 file.'))
            return wavfile.read(temp.name)
    else:
        raise NotImplementedError(error_message, 'file must have \'.wav\' or \'.mp3\' extension.')



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



if __name__ == '__main__':
    unittest.main()

