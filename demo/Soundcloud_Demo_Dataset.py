
import pandas as pd
import os

class SoundcloudDemodatasetHandler(object):

    def __init__(self, dataset_path):

        self.dataset_path  = dataset_path
        self.metadata = pd.read_csv(os.path.join(self.dataset_path, "soundcloud_dataset_metadata.csv"), sep=",", engine="python")


    def getNameByID(self, id):
        return self.metadata["filename"][self.metadata["soundcloudid"] == id].values

    def getMp3pathByID(self, id):
        return "%s/%s.mp3" % (self.dataset_path, self.metadata["filename"][self.metadata["soundcloudid"] == id].values[0])