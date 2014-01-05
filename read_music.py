import numpy as np
import scipy as sp
import scipy.io.wavfile as wav

def read_mp3(path):
	import os
	oname = 'temp.wav'
	cmd = 'lame --decode {0} {1}'.format(path,oname )
	os.system(cmd)
	data = wav.read(oname)
	return data

def read_wave(path):
    return wav.read(path)