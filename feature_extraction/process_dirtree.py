import numpy as np
import pandas as pd
import os
import glob

import warnings
warnings.filterwarnings('ignore')

import sys
#sys.path.append("D:/Work/PhD/Eclipse/rp_extract")
sys.path.append("C:/Work/IFS/rp_extract")

from FFmpeg import FFmpeg
from rp_extract_python import rp_extract


DATA_DIR = "E:/Data/MIR/EU_SOUNDS"
FEATURE_DIR = "E:/Data/MIR/EU_SOUNDS_FEATURES"


if __name__ == '__main__':
    
    mp3_decoder = FFmpeg("D:/Research/Tools/ffmpeg/bin/ffmpeg.exe")
    
    error_log = open("E:/Data/MIR/EU_SOUNDS_FEATURES/error.log", 'w')
    
    i = 1

    for dir_name in glob.glob(os.path.join(DATA_DIR, "*")):
        
        src_path = dir_name.replace("\\","/")
        dst_path = src_path.replace(DATA_DIR, FEATURE_DIR)
        
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
            
        for mp3_name in glob.glob(os.path.join(src_path, "*.mp3")):
            
            src_filename = mp3_name.replace("\\","/")
            dst_filename = src_filename.replace(DATA_DIR, FEATURE_DIR)
            
            if os.path.exists("%s.npz" % (dst_filename)):
                i += 1
                continue
            
            try:
                
                print ">", src_filename
                
                samplerate, wavedata = mp3_decoder.convertAndRead(src_filename)
                wavedata = wavedata / float(32768)
    
                extracted_features = rp_extract(wavedata,                            # the two-channel wave-data of the audio-file
                                                samplerate,                          # the samplerate of the audio-file
                                                extract_rp          = True,          # <== extract this feature!
                                                extract_rh          = True,
                                                extract_ssd         = True,
                                                extract_mvd         = True,
                                                extract_trh         = True,
                                                extract_tssd        = True,
                                                transform_db        = True,          # apply psycho-accoustic transformation
                                                transform_phon      = True,          # apply psycho-accoustic transformation
                                                transform_sone      = True,          # apply psycho-accoustic transformation
                                                fluctuation_strength_weighting=True, # apply psycho-accoustic transformation
                                                skip_leadin_fadeout = 0,             # skip lead-in/fade-out. value = number of segments skipped
                                                step_width          = 1)             # 
    
    
                np.savez(dst_filename,
                         rp = extracted_features["rp"],
                         rh = extracted_features["rh"],
                         trh = extracted_features["trh"],
                         ssd = extracted_features["ssd"],
                         tssd = extracted_features["tssd"],
                         mvd = extracted_features["mvd"])
                
                i += 1
                
                if i % 10000 == 0:
                    print i
                    
                if i % 100 == 0:
                    error_log.flush()
                
            except Exception as e:
                err_msg = "%s\t%s\n" % (src_filename, e)
                print "***", err_msg
                #print err_msg
                error_log.write(err_msg)

                
error_log.close()