# generates an animated and synchronized video of RP features over time from an audio file

# call the python script as "python rpvideo.py test.wav" to generate test.avi with video & sound

from __future__ import print_function
#!/usr/bin/python

__author__ = "Jakob Abesser, 2015"


import sys
import time
# import warnings
# warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use("agg")
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append('../rp_extract')
from rp_plot import *
from rp_extract_python import rp_extract
from audiofile_read import *

# processing parameters
winLenSec = 6 # seconds
hopLenSec = 1 # seconds

# check input parameters
if len(sys.argv) == 1:
    raise Exception('Missing file name')
else:
    audiofile = sys.argv[1]

print("Process audio file ", audiofile)

# load audio file
samplerate, samplewidth, wavedata = audiofile_read(audiofile)
nsamples = wavedata.shape[0]
nchannels = wavedata.shape[1]

# downmix to mono
if wavedata.ndim == 2:
    samples = np.mean(wavedata, axis=1)

nSamples = len(samples)

# convert window size and hop size to samples
winLenSamples = round(samplerate*winLenSec)
hopLenSamples = round(samplerate*hopLenSec)

# sample start indices for each frame
frameStartIdx = np.arange(0, nSamples-winLenSamples-1, hopLenSamples)
nFrames = len(frameStartIdx)

# initialize matrix to store rhythm patterns
rp = np.zeros((nFrames, 24, 60))

t = time.time()
print("Compute rhythm patterns in ")
for f in range(nFrames):
    print('Frame {}/{}'.format(f, nFrames))
    currSamples = samples[frameStartIdx[f]:frameStartIdx[f]+winLenSamples]
    
    # extract rhythm patterns
    extracted_features = rp_extract(currSamples, samplerate, extract_rp=True)

    # store them
    rp[f,:,:] = extracted_features['rp'].reshape(24, 60, order='F')

print("Computed rhythm pattern descriptor in {:.4f} x realtime".format((time.time()-t)/(float(nSamples)/samplerate)))

# generate video

# set up a figure
fig = plt.figure()
# set up the plot to be animated
l = plt.imshow(np.random.random((24,60)), origin='lower', aspect='auto', interpolation='nearest')
plt.xlabel('Mod. Frequency Index')
plt.ylabel('Frequency [Bark]')

# update data within animation function
def animate(f):
    l.set_data(rp[f,:,:])
    return l,

# call animator
anim = animation.FuncAnimation(fig, animate, frames=nFrames)#, interval=hopLenSec*1000, blit=True)

# save video
fnVideo = audiofile.replace('.wav','.avi')
anim.save(fnVideo, fps=1./hopLenSec, writer="ffmpeg")#, codec="libx264")
os.system('ffmpeg -i {} -i {} -map 0:0 -map 1:0 -codec copy -shortest {}'.format(audiofile,fnVideo,fnVideo))
print("Video saved to {}".format(fnVideo))
