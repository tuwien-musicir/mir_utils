
from IPython.display import HTML
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import io
import base64


# soundcloud player
embedded_player_soundcloud = '<iframe width="100%" height="{1}" scrolling="no" frameborder="no"' + \
                             'src="https://w.soundcloud.com/player/?url=https%3A//api.soundcloud.com/tracks/{0}&amp;' + \
                             'auto_play=false&amp;hide_related=true&amp;show_comments=false&amp;show_user=false&amp;' + \
                             'show_reposts=false&amp;visual={2}"></iframe>'

# free music archive player
embedded_player_fma =       "<object width='600' height='60'><param name='movie' value='http://freemusicarchive.org/swf/trackplayer.swf'/>" + \
                            "<param name='flashvars' value='track=http://freemusicarchive.org/services/playlists/embed/track/{0}.xml'/>" + \
                            "<param name='allowscriptaccess' value='sameDomain'/>" + \
                            "<embed type='application/x-shockwave-flash' src='http://freemusicarchive.org/swf/trackplayer.swf'" + \
                            "width='500' height='80' flashvars='track=http://freemusicarchive.org/services/playlists/embed/track/{0}.xml'" + \
                            "allowscriptaccess='sameDomain' /></object>"




class SoundcloudTracklist(list):

    def __init__(self, *args, **kwargs):
        super(SoundcloudTracklist, self).__init__(args[0])

        self.width  = kwargs['width']
        self.height = kwargs['height']

        if kwargs['visual']:
            self.visual = "true"
        else:
            self.visual = "false"


    def _repr_html_(self):
        html = ["<table width='{0}%' style='border:none'>".format(self.width)]

        for row in self:
            html.append("<tr style='border:none'>")
            html.append("<td style='border:none'>{0}</td>".format(embedded_player_soundcloud.format(row,
                                                                                                    self.height,
                                                                                                    self.visual)))
            html.append("</tr>")
        html.append("</table>")

        return ''.join(html)


class FMATracklist(list):


    def __init__(self, width=100, height=120, visual=False):
        super(SoundcloudTracklist, self).__init__()

        self.width  = width
        self.height = height

        if visual:
            self.visual = "true"
        else:
            self.visual = "false"


    def _repr_html_(self):
        html = ["<table width='{0}%' style='border:none'>".format()]
        for row in self:
            html.append("<tr style='border:none'>")
            html.append("<td style='border:none'>{0}</td>".format(embedded_player_soundcloud.format(row),
                                                                  self.height,
                                                                  self.visual))
            html.append("</tr>")
        html.append("</table>")
        return ''.join(html)


class compareSimilarityResults(list):


    def __init__(self, *args, **kwargs):
        super(compareSimilarityResults, self).__init__(args[0])

        self.width  = kwargs['width']
        self.height = kwargs['height']
        self.columns = kwargs['columns']

        if kwargs['visual']:
            self.visual = "true"
        else:
            self.visual = "false"


    def _repr_html_(self):

        data = np.asarray(self).T.tolist()


        html = ["<table width='{0}%' style='border:none'>".format(self.width)]

        # === titles ===
        html.append("<tr style='border:none'>")

        for col_name in self.columns:
            html.append("<td style='border:none'><center><b>{0}</b></center></td>".format(col_name))

        html.append("</tr>")

        for row in data:
            html.append("<tr style='border:none'>")

            for col in row:
                html.append("<td style='border:none'>{0}</td>".format(embedded_player_soundcloud.format(col,
                                                                                                        self.height,
                                                                                                        self.visual)))
            html.append("</tr>")
        html.append("</table>")

        return ''.join(html)


def get_rp_as_imagebuf(features, width=493, height=352, dpi=72, cmap='jet'):

    features = features.reshape(24,60,order='F')

    plt.ioff()
    fig = plt.figure(figsize=(int(width/dpi), int(height/dpi)), dpi=dpi);
    ax = fig.add_subplot(111)
    fig.suptitle('Rhythm Patterns')
    ax.imshow(features, origin='lower', aspect='auto',interpolation='nearest',cmap=cmap);
    ax.set_xlabel('Mod. Frequency Index');
    ax.set_ylabel('Frequency [Bark]');


    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format = 'png');
    img_buffer.seek(0)
    plt.close()
    plt.ion()
    return base64.b64encode(img_buffer.getvalue())


def get_rh_as_imagebuf(hist, width=493, height=352, dpi=72, normalize=True):

    if len(hist.shape) == 2:
        hist = hist[0]

    if normalize:
        hist /= np.sum(hist)

    plt.ioff()

    fig = plt.figure(figsize=(int(width/dpi), int(height/dpi)), dpi=dpi);
    #plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    ax = fig.add_subplot(111)
    fig.suptitle('Rhythm Histogram')
    ax.bar(np.arange(0,60) / 6.0,hist);
    ax.set_xlim([0.0, 10.0])
    ax.set_xlabel('Mod. Frequency Index');



    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format = 'png');
    img_buffer.seek(0)
    plt.close()
    plt.ion()
    return base64.b64encode(img_buffer.getvalue())


def get_ssd_as_imagebuf(features, width=493, height=352, dpi=72, cmap='jet', std=False):

    features = features.reshape(24,7,order='F')

    if std:
        features[:,1] = np.sqrt(features[:,1])

    plt.ioff()
    fig = plt.figure(figsize=(int(width/dpi), int(height/dpi)), dpi=dpi);

    ax = fig.add_subplot(111)
    fig.suptitle('Statistical Spectrum Descriptors')
    ax.imshow(features, origin='lower', aspect='auto',interpolation='nearest',cmap=cmap);
    ax.set_xticklabels(['','mean', 'var', 'skew', 'kurt', 'median', 'min', 'max'])
    ax.set_ylabel('Frequency [Bark]')

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format = 'png');
    img_buffer.seek(0)
    plt.close()
    plt.ion()
    return base64.b64encode(img_buffer.getvalue())



def show_rp_features_with_souncloud_player(scds, soundcloud_ids, feature_sets, width=900, margin=10):

    img_width = np.min([430, int((width - 2*margin) / len(soundcloud_ids))])
    img_height = int(img_width * (288.0/432.0))

    supported_features = ['rp', 'rh', 'ssd']

    html = ["<table width='100%' style='border:none'>"]

    if img_width <= 300:
        html.append("<tr style='border:none'>")

        for scid in soundcloud_ids:
            html.append("<td style='border:none;text-align:center'><center><b>{0}</b></center></td>".format(scds.getNameByID(scid)))

        html.append("</tr>")

    # === Soundcloud Players ===
    html.append("<tr style='border:none'>")

    for scid in soundcloud_ids:
        html.append("<td style='border:none;text-align:center'>{0}</td>".format(scds.getPlayerHTMLForID(scid, width=90, visual=False)))

    html.append("</tr>")

    # === feature-Plots ===
    for f_set in feature_sets:

        html.append("<tr style='border:none'>")

        for scid in soundcloud_ids:

            if f_set in supported_features:

                if f_set == 'rp':
                    features = scds.getFeaturesForID(scid, 'rp')
                    img_tag = "<img src='data:image/png;base64," + get_rp_as_imagebuf(features, width=img_width, height=img_height) + "'/>"
                elif f_set == 'rh':
                    features = scds.getFeaturesForID(scid, 'rh')
                    img_tag = "<img src='data:image/png;base64," + get_rh_as_imagebuf(features, width=img_width, height=img_height) + "'/>"
                elif f_set == 'ssd':
                    features = scds.getFeaturesForID(scid, 'ssd')
                    img_tag = "<img src='data:image/png;base64," + get_ssd_as_imagebuf(features, width=img_width, height=img_height, std=True) + "'/>"

                html.append("<td align='center' style='border:none;text-align:center'>{0}</td>".format(img_tag))
            else:
                html.append("<td align='center' style='border:none;text-align:center'>Featureset '{0}' not supported!</td>".format(f_set))

        html.append("</tr>")


    html.append("</table>")

    result = ''.join(html)
    return HTML(result)