'''
Created on 21.01.2016

@author: SchindlerA
'''

import sys
import os
import subprocess

import glob

import shutil

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

BENCHMARK_DIR_NAME = "D:/MIR/benchmarks"

DATASETS = ["D:/Research/Data/MIR/GTZAN",
            "D:/Research/Data/MIR/ISMIR_Genre",
            "D:/Research/Data/MIR/ISMIR_Rhythm",
            "D:/Research/Data/MIR/LatinMusicDatabase",
            "D:/Research/Data/MIR/Music_Audio_Benchmark_Data_Set",
            "D:/Research/Data/MIR/IEEE_AASP_CASA_Challenge_-_Public_Dataset_for_Scene_Classification_Task"]

RP_FEATURES      = ['rp','ssd','rh','tssd','trh','mvd']
MARSYAS_FEATURES = ["ZeroCrossings", "Centroid", "Rolloff", "Flux", "MFCC", "Chroma", "SCF", "SFM", "LSP", "LPCC"]

COMBINATIONS = [['rp','ssd'],['rp','tssd'],['rp','trh','tssd'],['ssd','rh'],['rp','mvd'],
                ["ZeroCrossings", "Centroid", "Rolloff", "Flux", "MFCC"], ["Centroid", "Rolloff", "Flux"],
                ['rp','MFCC'], ['rh', 'MFCC'], ['MFCC', 'Chroma'], ["rp", "Chroma"], ["rp", "MFCC", "Chroma"],
                ["LSP", "rp"], ["LSP", "ssd"]]
COMBINATIONS.extend([[feat] for feat in RP_FEATURES])
COMBINATIONS.extend([[feat] for feat in MARSYAS_FEATURES])


# CLASSIFIERS = {"NB": GaussianNB()}
CLASSIFIERS = {"KNN": KNeighborsClassifier(1),
               "SVM": SVC(kernel='linear'),
               "NB": GaussianNB()
               }

NUM_FOLDS = 10

TRAIN = 0
TEST  = 1

SKIP_EXISTING_FILES = True

HTML_CSS_FILE = "C:/Work/IFS/mir_utils/benchmarks/templates/benchmarks.css"


# === FEATURE EXTRACTORS ===

# rp_extract
sys.path.insert(0,'C:/Work/IFS/rp_extract')
os.environ['PATH'] += os.pathsep + "D:/Research/Tools/ffmpeg/bin/ffmpeg.exe"
import rp_extract_batch as rp_extract

# marsyas
MARSYAS_BEXTRACT_BIN = "D:/Research/Tools/marsyas/bextract.exe"




def create_filelists(dataset_path, config_path):
    
    import pandas as pd

    labels    = []
    filenames = []
    filenames_full_path = []

    for dir_name in glob.glob(os.path.join(dataset_path, "*")):
        
        dir_name = dir_name.replace("\\", "/")
        label    = dir_name.split("/")[-1]
        
        for f_name in glob.glob(os.path.join(dir_name, "*.*")):
            
            filenames.append( "%s/%s" % (label,f_name.replace("\\", "/").split("/")[-1] ))
            filenames_full_path.append( f_name.replace("\\", "/"))
            labels.append(label)
    
        
    filelist_data = pd.DataFrame({"filenames": filenames, "labels": labels})
    filelist_data.to_csv("%s/filelist.txt" % (config_path), sep="\t", header=None, index=None)
    
    filelist_fn_data = pd.DataFrame({"filenames": filenames_full_path, "labels": labels})
    filelist_fn_data.to_csv("%s/filelist_full_path.txt" % (config_path), sep="\t", header=None, index=None)
    

def create_partitions(work_dir_path):

    import cPickle
    import numpy as np

    from sklearn.preprocessing import LabelEncoder
    from sklearn.cross_validation import StratifiedKFold 

    # load data from one feature file
    npz    = np.load("%s/features/features.ssd.npz" % (work_dir_path))
    labels = npz["labels"]
    npz.close()
    
    # create labelencoder
    if not (os.path.exists("%s/labelencoder.pkl" % (work_dir_path)) and SKIP_EXISTING_FILES):
        
        le = LabelEncoder()
        le.fit(labels)
        
        with open("%s/labelencoder.pkl" % (work_dir_path), 'wb') as fid:
            cPickle.dump(le, fid)
            
    # create partitions
    if not (os.path.exists("%s/stratified_%dfold.pkl" % (work_dir_path, NUM_FOLDS)) and SKIP_EXISTING_FILES):
        
        cv = StratifiedKFold(le.transform(labels),
                             n_folds=NUM_FOLDS, 
                             shuffle=False)
        
        with open("%s/stratified_%dfold.pkl" % (work_dir_path, NUM_FOLDS), 'wb') as fid:
            cPickle.dump(cv, fid)

        
def create_npz_from_rpextract_format(input_filename, npz_dest_path, config_dir_path):
    
    import numpy as np
    import pandas as pd
    
    csv = pd.read_csv(input_filename, header=None)
    
    np.savez(npz_dest_path, 
             data      = csv[csv.columns[2:-1]].values, 
             filenames = csv[csv.columns[0]],
             labels    = csv[csv.columns[1]])
    
def create_npz_from_marsyas_arff(input_filename, work_dir_features_path, config_dir_path):
    
    import numpy as np
    import pandas as pd
    import arff
    
    fl = pd.read_csv("%s/filelist_full_path.txt" % (config_dir_path), header=None, sep="\t", )
    fl.columns = ["filename", "label"]
    
    
    arff_data = arff.load(open(input_filename, 'rb'))
    data      = pd.DataFrame(arff_data["data"], 
                             columns=[attr for attr, val in arff_data["attributes"]])
    
    for feat_name in MARSYAS_FEATURES:
    
        np.savez("%s/features.%s" % (work_dir_features_path, feat_name), 
                 data      = data[data.columns[data.columns.str.contains(feat_name)]].values, 
                 filenames = fl["filename"],
                 labels    = fl["label"])


def load_data(feat_file):

    import numpy as np

    # load data
    npz    = np.load(feat_file)
    data   = npz["data"]
    labels = npz["labels"]
    npz.close()
    
    return data, labels



def run_classifications(work_dir_path, work_dir_features_path, work_dir_results_path):
    
    import cPickle
    import numpy as np

    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.cross_validation import StratifiedKFold
    
    scaler = StandardScaler()
    
    # load partitions
    with open("%s/stratified_10fold.pkl" % (work_dir_path), 'rb') as fid:
        cross_validation = cPickle.load(fid)
    
    # load labelencoder
    with open("%s/labelencoder.pkl" % (work_dir_path), 'rb') as fid:
        le = cPickle.load(fid)
    
    for feat_combi in COMBINATIONS:
        
        if len(feat_combi) == 1:
            feat_name = feat_combi[0]
        else:
            feat_name   = "_".join(feat_combi)
            
        predictions_resultfile_name = "%s/predictions.%s.pkl" % (work_dir_results_path,feat_name)
        
        if os.path.exists(predictions_resultfile_name) and SKIP_EXISTING_FILES:
            continue
            
        if len(feat_combi) == 1:
            data, labels = load_data("%s/features.%s.npz" % (work_dir_features_path, feat_name))
        else:
            loaded_data = [load_data("%s/features.%s.npz" % (work_dir_features_path, f_name)) for f_name in feat_combi]
            data        = np.concatenate([vecs for vecs, labels in loaded_data], axis=1)
            labels      = loaded_data[0][1]
            del loaded_data
        
        
        results_for_feature = {}
        
        # normalize features
        data = scaler.fit_transform(data)
        
        for classifier_name, classifier in CLASSIFIERS.iteritems():
            
            print feat_name, classifier_name
            
            results_for_feature[classifier_name] = []

            for train, test in cross_validation:
            
                # train
                clf         = classifier.fit(data[train], le.transform(labels[train]))
                
                # test
                predictions = clf.predict(data[test])
                
                # store
                results_for_feature[classifier_name].append(predictions)
                
        # save results
        with open(predictions_resultfile_name, 'wb') as fid:
            cPickle.dump(results_for_feature, fid)
                
def write_output(result_table, dataset_output_results_path, filename):
    
    print "%s/%s.html" % (dataset_output_results_path,filename)
    
    output = result_table.sort_values("SVM", ascending=False)
    output.to_html("%s/%s.html" % (dataset_output_results_path,filename))
    

def evaluate_results_and_create_output(work_dir_path, work_dir_features_path, work_dir_results_path,dataset_output_results_path):
    
    import cPickle
    import numpy as np
    import pandas as pd

    from sklearn.preprocessing import LabelEncoder
    from sklearn.cross_validation import StratifiedKFold
    import sklearn.metrics as sk_metrics
    
    # load partitions
    with open("%s/stratified_10fold.pkl" % (work_dir_path), 'rb') as fid:
        cross_validation = cPickle.load(fid)
        cv_folds         = [[train,test] for train,test in cross_validation]
    
    # load labelencoder
    with open("%s/labelencoder.pkl" % (work_dir_path), 'rb') as fid:
        le = cPickle.load(fid)
    
    dataset_mean_accuracies = []
    featuresets = []
    cms = []
    dataset_predictions = []
    
    for feat_combi in COMBINATIONS:
    #for feat_result_file in glob.glob(os.path.join(work_dir_results_path, "*.pkl")):
        
        if len(feat_combi) == 1:
            feat_name       = feat_combi[0]
            feat_label_file = "%s/features.%s.npz" % (work_dir_features_path, feat_name)
        else:
            feat_name       = "_".join(feat_combi)
            feat_label_file = "%s/features.%s.npz" % (work_dir_features_path, feat_combi[0])
        
        feat_result_file = "%s/predictions.%s.pkl" % (work_dir_results_path,feat_name)
        featuresets.append(feat_name)
        
        # load data
        npz       = np.load(feat_label_file)
        labels    = npz["labels"]
        filenames = npz["filenames"]
        npz.close()

        # load predictions
        with open(feat_result_file, 'rb') as fid:
            predictions = cPickle.load(fid)
        
        feat_mean_accuracies = []
        
        for classifier_name in predictions.keys():
            
            # accuracies
            accuracies = [sk_metrics.accuracy_score(le.transform(labels[cv_folds[fold][TEST]]), predictions[classifier_name][fold] ) for fold in range(NUM_FOLDS)]
            feat_mean_accuracies.append( np.mean(accuracies) )

            all_test_ids    = np.concatenate([cv_folds[fold][TEST] for fold in range(NUM_FOLDS)], axis=0)
            all_test        = np.concatenate([le.transform(labels[cv_folds[fold][TEST]]) for fold in range(NUM_FOLDS)], axis=0)
            all_predictions = np.concatenate([predictions[classifier_name][fold] for fold in range(NUM_FOLDS)], axis=0)
            
            true_positives  = np.where(all_predictions == all_test)
            pred = all_predictions.copy()
            pred[true_positives] = -1
            
            dataset_predictions.append(pred)
            
            cm = sk_metrics.confusion_matrix(all_test,all_predictions)
            cms.append(cm)
            
            

        dataset_mean_accuracies.append(feat_mean_accuracies)

    result_table_mean_accuracies = pd.DataFrame(dataset_mean_accuracies, columns=predictions.keys())
    result_table_mean_accuracies.insert(0, "featureset", featuresets)

    # persist table    
    result_table_mean_accuracies.to_hdf("%s/result_table_mean_accuracies.h5" % (work_dir_path), "results")
    
    # write output
    write_output(result_table_mean_accuracies, dataset_output_results_path, "mean_accuracies")

    #print np.sum(cms,axis=0)
    #print le.classes_
    
    dataset_predictions = np.asarray(dataset_predictions).T
    
    
    bins = [-1]
    bins.extend(range(len(le.classes_)+1))
    dataset_predictions = np.apply_along_axis(lambda x: np.histogram(x, bins=bins)[0], 1, dataset_predictions)
    
    columns= ["true_positive"]
    columns.extend(le.classes_)
    columns = np.asarray(columns)

    miss =  pd.DataFrame(dataset_predictions, columns=columns)
    
    sort_order = [0]
    sort_order.extend((np.argsort(miss.max(axis=0)[1:].values)[::-1] + 1).tolist()) 
    
    miss["filenames"] = filenames[all_test_ids]
    miss.sort_values(columns[sort_order].tolist()).to_html("%s/classification_frequencies.html" % (dataset_output_results_path), index=None)


    #exit()
    
    return result_table_mean_accuracies


def copy_auxillary_files(global_output_path):
    
    shutil.copy(HTML_CSS_FILE, "%s/%s" % (global_output_path, HTML_CSS_FILE.split("/")[-1]))



def write_summary_table(result_tables, global_output_path):
    
    html = "<html><body>"
    
    html = "%s<head>" % (html)
    html = "%s<link rel=\"stylesheet\" type=\"text/css\" href=\"%s\">" % (html,HTML_CSS_FILE.split("/")[-1])
    html = "%s</head>" % (html)
    html = "%s<body>" % (html)
    html = "%s<table class=\"resulttable\">" % (html)
    
    for dataset_name, res_table in result_tables:
        
        html = "%s<tr class=\"datasetname\"><th colspan=\"%d\"><center><a href='%s/mean_accuracies.html'>%s</a></center></th><tr>" % (html, 
                                                                                                                             res_table.columns.shape[0], 
                                                                                                                             dataset_name,
                                                                                                                             dataset_name)
                
        for col in res_table.columns:
            html = "%s<td class=\"classifiername\">%s</td>" % (html, col)
            
        html = "%s</tr>" % (html)
        
        for row in res_table.sort_values("SVM", ascending=False).iterrows():
            html = "%s<tr>" % (html)
            
            col_nr = 1
            
            for col in res_table.columns:
                if col_nr == 1:
                    html = "%s<td class=\"featurename\"><a href=\"feature_summary_%s.html\">%s</a></td>" % (html, row[1][col], row[1][col])
                else:
                    html = "%s<td class=\"resultvalue\">%.2f</td>" % (html, row[1][col] * 100)
                    
                col_nr += 1
            html = "%s</tr>" % (html)

    
    html = "%s</table>" % (html)
    
    html = "%s</body></html>" % (html)

    f = open("%s/summary_table.html" % (global_output_path), 'w')
    f.write(html)
    f.close()
    
    
def write_feature_summary(result_tables, global_output_path):
    
    for feat_combi in COMBINATIONS:
        
        feat_name       = "_".join(feat_combi)
    
        html = "<html><body>"
        
        html = "%s<head>" % (html)
        html = "%s<link rel=\"stylesheet\" type=\"text/css\" href=\"%s\">" % (html,HTML_CSS_FILE.split("/")[-1])
        html = "%s</head>" % (html)
        html = "%s<body>" % (html)
        html = "%s<table class=\"resulttable\">" % (html)
        
        
        html = "%s<tr>" % (html)
        html = "%s<td class=\"classifiername\">Dataset</td>" % (html)
        
        for col in result_tables[0][1].columns[1:]:
            html = "%s<td class=\"classifiername\">%s</td>" % (html, col)
            
        html = "%s</tr>" % (html)
        
        for dataset_name, res_table in result_tables:
        
            html = "%s<tr>" % (html)
            html = "%s<td>%s</td>" % (html, dataset_name
                                      )
            for row in res_table.loc[res_table['featureset'] == feat_name].iterrows():
                
                col_nr = 1
                
                for col in res_table.columns[1:]:
                    html = "%s<td class=\"resultvalue\">%.2f</td>" % (html, row[1][col] * 100)
                        
                    col_nr += 1
                html = "%s</tr>" % (html)
    
        
        html = "%s</table>" % (html)
        
        html = "%s</body></html>" % (html)
    
        f = open("%s/feature_summary_%s.html" % (global_output_path, feat_name), 'w')
        f.write(html)
        f.close()

if __name__ == '__main__':
    
    result_tables = []
    
    for dataset_path in DATASETS:
        
        dataset_path = dataset_path.replace("\\", "/")
        dataset_name = dataset_path.split("/")[-1]
        
        # =============================
        # === Variables Definitions ===
        # =============================
        
        # input
        input_audio_path               = "%s/audio" % (dataset_path)

        # features
        output_feature_path_rpextract  = "%s/features/rp_extract_python/rp_extract" % (dataset_path)
        output_feature_path_marsyas    = "%s/features/marsyas/marsyas.arff" % (dataset_path)

        # config
        config_dir_path                = "%s/datasets/%s/config" % (BENCHMARK_DIR_NAME, dataset_name)

        # work dir
        work_dir_path                  = "%s/datasets/%s/intermediate_data" % (BENCHMARK_DIR_NAME, dataset_name)
        work_dir_features_path         = "%s/features" % (work_dir_path)
        work_dir_results_path          = "%s/results" % (work_dir_path)

        # output
        global_output_path             = "%s/output" % (BENCHMARK_DIR_NAME)
        dataset_output_results_path    = "%s/%s" % (global_output_path, dataset_name)
        
        # === create dirs ===
        if not os.path.exists(config_dir_path):               os.makedirs(config_dir_path)
        if not os.path.exists(output_feature_path_rpextract): os.makedirs(output_feature_path_rpextract)
        if not os.path.exists(work_dir_path):                 os.makedirs(work_dir_path)
        if not os.path.exists(work_dir_features_path):        os.makedirs(work_dir_features_path)
        if not os.path.exists(work_dir_results_path):         os.makedirs(work_dir_results_path)
        if not os.path.exists(dataset_output_results_path):   os.makedirs(dataset_output_results_path)
        
        
        # ========================
        # === start processing ===
        # ========================
         
        # create filelist
        filelist = create_filelists(input_audio_path, config_dir_path)
         

        # extract features
        if not (os.path.exists("%s.ssd" % (output_feature_path_rpextract)) and SKIP_EXISTING_FILES): 
        
            print "* extracting rp_extract: %s" % (dataset_name)
            # rp_extract
            rp_extract.extract_all_files_in_path(input_audio_path, 
                                                 output_feature_path_rpextract,
                                                 RP_FEATURES, 
                                                 ('.wav','.mp3','.aif','.aiff','.m4a','.au'),
                                                 True,
                                                 verbose=False,
                                                 filelist_path="%s/filelist.txt" % (config_dir_path))
            
            # marsyas
            
            
        if not (os.path.exists(output_feature_path_marsyas) and SKIP_EXISTING_FILES):
            
            print "* extracting marsyas: %s" % (dataset_name)
            marsyas_arff_filename = output_feature_path_marsyas.replace("\\","/").split("/")[-1]
            output_feature_path_marsyas_dir = output_feature_path_marsyas.replace("\\","/").replace(marsyas_arff_filename,"")
            
            if not os.path.exists(output_feature_path_marsyas_dir):   os.makedirs(output_feature_path_marsyas_dir)
            cmd = [MARSYAS_BEXTRACT_BIN, 
                   "-sv", "-l", "30", "-fe", "-mfcc", "-chroma", "-ctd", "-rlf", "-flx", "-zcrs", "-sfm", "-scf", "-lsp", "-lpcc",
                   "%s/filelist_full_path.txt" % (config_dir_path),
                   "-w", output_feature_path_marsyas]
            
            print cmd
            subprocess.call(cmd)
            # bextract  -sv -l 30 -fe -mfcc -chroma -ctd -rlf -flx -zcrs -sfm -scf -lsp -lpcc D:\MIR\benchmarks\datasets\GTZAN\config\filelist_full_path.txt -w E:\test\GTZAN.arff 
            
        
        # create npzs
        
        # from rp_extract
        for feat in RP_FEATURES:
             
            feature_file_path = "%s.%s" % (output_feature_path_rpextract, feat)
            npz_dest_path     = "%s/features.%s" % (work_dir_features_path, feat)
             
            if not (os.path.exists("%s.npz" % (npz_dest_path)) and SKIP_EXISTING_FILES): 
                create_npz_from_rpextract_format(feature_file_path, npz_dest_path, config_dir_path)
                
        # from marsyas
        create_npz_from_marsyas_arff(output_feature_path_marsyas, work_dir_features_path, config_dir_path)
        
        # create consistent stratified partitions
        create_partitions(work_dir_path)
        
        # run classification
        run_classifications(work_dir_path, work_dir_features_path, work_dir_results_path)

        # evaluate classification results
        res_table = evaluate_results_and_create_output(work_dir_path, 
                                                       work_dir_features_path, 
                                                       work_dir_results_path, 
                                                       dataset_output_results_path)
        
        result_tables.append([dataset_name, res_table])
        
        #break
        
    
    #copy_auxillary_files(global_output_path)
    
    write_summary_table(result_tables, global_output_path)
    write_feature_summary(result_tables, global_output_path)
        
        
        
        
        
        