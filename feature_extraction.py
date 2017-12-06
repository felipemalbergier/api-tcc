# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 22:16:59 2017

@author: nazli
"""
import pickle
import json
from sklearn import metrics
import pandas as pd
from os import listdir
from os.path import isfile, join
import time
import predict_audio
import statistics as st
import numpy as np


with open('config.json') as json_data:
    config = json.load(json_data)

with open('features.json') as json_data:
    feat_config = json.load(json_data)
    
    
def get_files(output_path, algo): 
    files = [f for f in listdir(output_path) if isfile(join(output_path, f))]
    data = pd.DataFrame()
    for file in files:
        #Get feature name: only get features that matter for this algorithm
        feature = (file.split('.wav.')[1].split('.csv')[0])

        if feature in feat_config[algo]:
            df = pd.read_csv(output_path + file, skiprows=range(0, 4), header = 0).reset_index()
            
            if('index' in df.columns):
                df = df.drop('index', 1)
            
            #Add columns names
            columns = []
            for i in range(0, len(df.columns)):
                columns.append('{}_{}'.format(feature, i))
            df.columns = columns

            
            #Filter unnecessary columns for some features
            if feature in ['mfcc', 'spectralShapeStatistics', 'lpc']:
                key = 'keep_{}_{}'.format(feature, algo)
                keep_cols = feat_config[key]
                
                #Filter columns
                df = df[keep_cols]
        
            data = pd.concat([data, df], axis=1)

    #Make sure all values are numeric
    columns = data.columns
    for i in range(0, len(columns)):
        data[columns[i]] = data[columns[i]].apply(pd.to_numeric)
    
    return data
    
def get_1sec_frames(data, features, algo):
    frame_size = 100
    rows = round(len(data.index) / frame_size)
    processed_df = pd.DataFrame(index=range(0, rows), columns = features)
    
    point = 0
    
    while point < rows:
        frame = pd.DataFrame()
        frame = data.iloc[point * frame_size:(point + 1) * frame_size, :]

        results = get_sample_avg_var(frame, algo)
        
        processed_df.loc[point] = list(map(lambda x: results[x] if x != 'autocorrelation' else results['rms'], features))

        point += 1
    
    return processed_df

def autocorr(x, mode):
    result = np.correlate(x, x, mode=mode)
    return result[int(len(result)/2):]

def get_sample_avg_var(frame, algo):
    results = {}
    if algo in ['svm', 'knn', 'nn']:
        for feature in feat_config[algo + '_mean']:
            mean = st.mean(frame[feature])
            results[feature + '_mean'] = mean
        for feature in feat_config[algo + '_var']:
            var = st.variance(frame[feature])
            results[feature + '_var'] = var
        for feature in feat_config[algo + '_mean_var']:
            mean = st.mean(frame[feature])
            results[feature + '_mean'] = mean
            var = st.variance(frame[feature])
            results[feature + '_var'] = var
    
    
    ### Get energy results ###
    col = frame['energy_0']
    col = col.values
    mean = st.mean(col)
    var = st.variance(col)
    count = 0
    
    for i in range(0, len(col)):
        if(col[i] < 0.5 * mean):
            count += 1
            
    results['rms'] = var / (mean * mean)
    results['low_energy_proportion'] = count / len(col)
    results['energy_0_mean'] = mean
    results['energy_0_var'] = var
    
    return results

def remove_outliers(prediction):
    for i in range(1, len(prediction) - 1):
        if(prediction[i - 1] != prediction[i] and prediction[i - 1] == prediction[i + 1]):
            prediction[i] = prediction[i - 1]
    return prediction

# clf may be: 'svm', 'knn', 'nn', 'voter', 'svme', 'knne', 'nne', 'votere'
def main(output_path, features, clf):
    data_f = get_files(output_path, clf)
    data_f = data_f.dropna(0)
    
    data = get_1sec_frames(data_f, features, clf)    
    
    try:
        a = autocorr(data['autocorrelation'], 'full')
        data['autocorrelation'] = pd.DataFrame(a)
    except KeyError:
        print('No autocorrelation')

    return data   


if __name__ == '__main__':   
    output_path = input('Input output_path ')
    features = input('Input features ')
    clf = input('Input classifier ')
    data = main(output_path, features, clf)
    print ("Returned data:")
    print (data)
