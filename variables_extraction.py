# coding: utf-8
from os import rename
import os
from os.path import basename,isdir
import shutil
from yaafelib import AudioFileProcessor, FeaturePlan, Engine
import json

with open ('config.json') as f:
    config = json.load(f)
with open('features.json') as f:
    feat_json = json.load (f)

# Function to extract features from an array of audio files using AudioFileProcessor and write results to csv files
def process_audio(audioFile, engine):
    rename(audioFile,basename(audioFile))
    afp = AudioFileProcessor()
    afp.setOutputFormat('csv', config['OUTPUT_FOLDER'], {'Precision':'8'})
    # afp.setOutputFormat('h5', OUTPUT_FOLDER_NAME, {'mode':'overwrite'})
    afp.processFile(engine, basename(audioFile))
    rename(basename(audioFile),audioFile)

    return 

def main(audio_file, variables_to_extract):
    # Delete old files and recriate folder
    if isdir('output'): shutil.rmtree('output')
    os.mkdir('output')
    
    # Build a DataFlow object using FeaturePlan
    fp = FeaturePlan(sample_rate = 44100)
    for variable in variables_to_extract:
        fp.addFeature(feat_json['variables'][variable])

    df = fp.getDataFlow()
    
    # configure an Engine
    engine = Engine()
    engine.load(df)

    # Make csv files with audio variables
    process_audio(audio_file, engine)
    
    return

if __name__ == '__main__':   
    audio_file = input ('Input audio file: ')
    variable = input('Input variable to extract ')
    features = main(audio_file,variable)
    print (features)
