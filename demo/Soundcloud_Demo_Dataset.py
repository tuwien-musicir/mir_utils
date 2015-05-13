
import pandas as pd
import numpy as np
import os

import urllib2
import gzip
import StringIO

embedded_player_soundcloud = '<iframe width="{1}%" height="{2}" scrolling="no" frameborder="no"' + \
                             'src="https://w.soundcloud.com/player/?url=https%3A//api.soundcloud.com/tracks/{0}&amp;' + \
                             'auto_play=false&amp;hide_related=true&amp;show_comments=false&amp;show_user=false&amp;' + \
                             'show_reposts=false&amp;visual={3}"></iframe>'


DATA_PATH = "http://www.ifs.tuwien.ac.at/~schindler/data"


SUPPORTED_FEATURE_SETS = ['rp', 'rh', 'ssd', 'tssd', 'trh', 'mvd']


class SoundcloudDemodatasetHandler(object):

    def __init__(self, local_dataset_path, lazy=False):

        self.local_dataset_path  = local_dataset_path
        self.metadata = pd.read_csv(os.path.join(self.local_dataset_path, "soundcloud_dataset_metadata.csv"), sep=",", engine="python")
        self.features = {}
        self.lazy_loading = lazy

        if not self.lazy_loading:
            self.load_all_features()

    def load_featureset(self, feature_set):

        data_url = "%s/soundcloud_dataset.features.%s.csv.gz" % (DATA_PATH,feature_set)
        response = urllib2.urlopen(data_url)

        compressedFile = StringIO.StringIO()
        compressedFile.write(response.read())
        compressedFile.seek(0)

        decompressedFile = gzip.GzipFile(fileobj=compressedFile, mode='rb')

        data = pd.read_csv(decompressedFile, engine='python', header=None, sep=',')

        self.features[feature_set] = {}
        self.features[feature_set]["data"]   = data.iloc[:,:-2].values.astype(np.float32)
        self.features[feature_set]["labels"] = data.iloc[:,-2].values
        self.features[feature_set]["ids"]    = data.iloc[:,-1].values

    def load_all_features(self):

        for f_set in feature_sets:
            self.load_featureset(f_set)



    def getNameByID(self, id):
        return self.metadata["filename"][self.metadata["soundcloudid"] == id].values[0]

    def getMp3pathByID(self, id):
        return "%s/%s.mp3" % (self.local_dataset_path, self.metadata["filename"][self.metadata["soundcloudid"] == id].values[0])

    def getPlayerHTMLForID(self, id, width=100, height=120, visual=False):
        if visual:
            visual_str = "true"
        else:
            visual_str = "false"

        return embedded_player_soundcloud.format(id, width, height, visual_str)

    def getFeaturesForID(self, id, feature_set):

        if feature_set not in self.features.keys() and self.lazy_loading:
            self.load_featureset(feature_set)

        return self.features[feature_set]['data'][self.features[feature_set]["ids"] == id].copy()

    def getFeatures(self, feature_set):

        if feature_set not in self.features.keys() and self.lazy_loading:
            self.load_featureset(feature_set)

        return self.features[feature_set]['data'].copy()

    def getFeatureIndexByID(self, id, feature_set):
        return self.features[feature_set]["ids"] == id

    def getIdsByIndex(self, indeces, feature_set):
        return self.features[feature_set]["ids"][indeces]

    def getCombinedFeaturesets(self, feature_sets):

        # TODO: this only works for two feature-sets and only for rp_extract where all features have exactly the same order
        merged = np.concatenate([self.getFeatures(feature_sets[0]), self.getFeatures(feature_sets[1])], axis=1)
        return merged

