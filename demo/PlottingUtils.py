
import matplotlib.pyplot as plt

import pylab
import numpy as np

def plotmatrix(features):
    pylab.figure()
    pylab.imshow(features, origin='lower', aspect='auto',interpolation='nearest')
    plt.xlabel('Mod. Frequency Index')
    pylab.ylabel('Frequency [Bark]')
    pylab.show()


def plotrp(features, reshape=True):

    if reshape:
        features = features.reshape(24,60,order='F')

    plotmatrix(features)


def plotssd(features, reshape=True):

    if reshape:
        features = features.reshape(24,7,order='F')

    pylab.figure()
    pylab.imshow(features, origin='lower', aspect='auto',interpolation='nearest')
    pylab.xticks(range(0,7), ['mean', 'var', 'skew', 'kurt', 'median', 'min', 'max'])
    pylab.ylabel('Frequency [Bark]')
    pylab.show()

def plotrh(hist):
    plt.bar(range(0,60),hist) # 50, normed=1, facecolor='g', alpha=0.75)
    plt.xlabel('Mod. Frequency Index')
    #plt.ylabel('Probability')
    plt.title('Rhythm Histogram')
    plt.show()

def plot_waveform(samples, width=6, height=4, stereo=False):

    fig = plt.figure(num=None, figsize=(width, height), dpi=72, facecolor='w', edgecolor='k');

    if not stereo and samples.shape[1] == 2:
        samples_to_plot = samples.copy().mean(axis=1)
    else:
        samples_to_plot = samples

    channel_1 = fig.add_subplot(111);
    channel_1.set_ylabel('Channel 1');
    #channel_1.set_xlim(0,song_length) # todo
    channel_1.set_ylim(-32768,32768);
    channel_1.plot(samples_to_plot);

    if stereo:
        channel_2 = fig.add_subplot(212);
        channel_2.set_ylabel('Channel 2');
        channel_2.set_xlabel('Time (s)');
        channel_2.set_ylim(-32768,32768);
        #channel_2.set_xlim(0,song_length) # todo
        channel_2.plot(samples[:,1]);

    plt.show();
    plt.clf();



def plotstft(samples, samplerate, binsize=2**10, plotpath=None, colormap="jet", ax=None, fig=None, width=6, height=4, ignore=False):

    from Features import stft, logscale_spec

    if ignore:
        import warnings
        warnings.filterwarnings('ignore')

    s = stft(samples, binsize)

    sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)
    ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel

    timebins, freqbins = np.shape(ims)

    if ax is None:
        fig, ax = plt.subplots(1, 1, sharey=True, figsize=(width, height))

    #ax.figure(figsize=(15, 7.5))
    cax = ax.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
    #cbar = fig.colorbar(cax, ticks=[-1, 0, 1], cax=ax)
    #ax.set_colorbar()

    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (hz)")
    ax.set_xlim([0, timebins-1])
    ax.set_ylim([0, freqbins])

    xlocs = np.float32(np.linspace(0, timebins-1, 5))
    ax.set_xticks(xlocs, ["%.02f" % l for l in ((xlocs*len(samples)/timebins)+(0.5*binsize))/samplerate])
    ylocs = np.int16(np.round(np.linspace(0, freqbins-1, 10)))
    ax.set_yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])

    if plotpath:
        plt.savefig(plotpath, bbox_inches="tight")
    else:
        if ax==None:
            plt.show()

    #plt.clf();
    b = ["%.02f" % l for l in ((xlocs*len(samples)/timebins)+(0.5*binsize))/samplerate]
    return xlocs, b, timebins


def superimpose_timedomain_feature(wavedata, sample_rate, feature, block_length=1024, spectrogram=False, width=8, height=4, squared=False):


    if wavedata.shape[1] == 2:
        samples_to_plot = wavedata.copy().mean(axis=1)

    if feature == 'zcr':
        from Features import zero_crossing_rate
        feature_data, timestamps = zero_crossing_rate(samples_to_plot, block_length, sample_rate)

    if spectrogram:
        # plot spectrogram
        _,_,_ = plotstft(samples_to_plot, sample_rate, width=width, height=height);

    fig = plt.figure(num=None, figsize=(width, height), dpi=72, facecolor='w', edgecolor='k');
    channel_1 = fig.add_subplot(111);
    channel_1.set_ylabel('Channel 1');
    channel_1.set_xlabel('time');

    # plot waveform
    scaled_wf_y = ((np.arange(0,samples_to_plot.shape[0]).astype(np.float)) /sample_rate) * 1000.0

    if squared:
        scaled_wf_x = (samples_to_plot**2 / np.max(sample_rate**2))
    else:
        scaled_wf_x = (samples_to_plot / np.max(sample_rate) / 2.0 ) + 0.5


    plt.plot(scaled_wf_y, scaled_wf_x, color='lightgrey');

    # plot feature-data
    scaled_fd_y = timestamps * 1000.0
    scaled_fd_x = (feature_data / np.max(feature_data))

    plt.plot(scaled_fd_y, scaled_fd_x, color='r', linewidth=0.5);

    plt.show();
    plt.clf();



def superimpose_frequencydomain_feature(wavedata, sample_rate, feature, block_length=1024, width=8, height=4):

    if wavedata.shape[1] == 2:
        samples_to_plot = wavedata.copy().mean(axis=1)

    if feature == 'sc':
        from Features import spectral_centroid
        feature_data, timestamps = spectral_centroid(samples_to_plot, block_length, sample_rate)
    elif feature == 'sf':
        from Features import spectral_flux
        feature_data, timestamps = spectral_flux(samples_to_plot, block_length, sample_rate)

    fig, ax = plt.subplots(1, 1, sharey=True, figsize=(width, height), dpi=72, facecolor='w', edgecolor='k')

    # plot spectrogram
    xlocs, b, timebins = plotstft(samples_to_plot, sample_rate, ax=ax);

    #channel_1.set_ylabel('Channel 1');
    #channel_1.set_xlabel('time');

    # plot waveform
    scaled_wf_y = ((np.arange(0,samples_to_plot.shape[0]).astype(np.float)) /sample_rate) * 1000.0

    # plot feature-data
    scaled_fd_y = timestamps * 1000.0
    scaled_fd_x = (feature_data / np.max(feature_data)) * 1000

    print feature_data

    ax.plot(scaled_fd_y, scaled_fd_x, color='black', linewidth=0.5);

    plt.show();
    plt.clf();

    return feature_data, timebins, scaled_fd_y