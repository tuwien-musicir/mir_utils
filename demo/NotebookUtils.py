
from IPython.display import HTML
import numpy as np

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

        if kwargs['visual']:
            self.visual = "true"
        else:
            self.visual = "false"


    def _repr_html_(self):
        html = ["<table width='{0}%' style='border:none'>".format(self.width)]

        for row in self:
            html.append("<tr style='border:none'>")

            for col in row:
                html.append("<td style='border:none'>{0}</td>".format(embedded_player_soundcloud.format(col,
                                                                                                        self.height,
                                                                                                        self.visual)))
            html.append("</tr>")
        html.append("</table>")

        return ''.join(html)
