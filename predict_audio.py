
import json
import pickle
import time
from os.path import basename

# from yaafelib import *

import featureExtraction as fex
import variablesExtractionScript_OnlyOne as vex

with open('features.json') as json_data:
    feat_config = json.load(json_data)

with open('config.json') as json_data:
    config = json.load(json_data)

def remove_outliers(prediction):
    for i in range(1, len(prediction) - 1):
        if(prediction[i - 1] != prediction[i] and prediction[i - 1] == prediction[i + 1]):
            prediction[i] = prediction[i - 1]
    return prediction

def predict_audio(clf_and_scaler_folder, used_features, clf_to_use):
    output_path = config['OUTPUT_FOLDER']
    # From csv files get features from 1 second frames
    # features = fex.main(output_path, used_features, clf_time) 
    features = fex.main(output_path, used_features, clf_to_use)

    scaler_file = 'scaler.sav'
    scaler_file = clf_and_scaler_folder + '/' + scaler_file
    scaler = pickle.load(open(scaler_file, 'rb'))
    
    clf_file = 'finalized_model.sav'
    clf_file = clf_and_scaler_folder + '/' + clf_file
    clf = pickle.load(open(clf_file, 'rb'))
    
    
    print('Scaler {}\nModelo {}'.format(scaler_file, clf_file))
    print (features.shape)
    scaled_data = scaler.transform(features)

    predicted = clf.predict(scaled_data)

    new_predicted = remove_outliers(predicted)

    new_predicted = list(new_predicted)

    likely_to_be_music = sum(new_predicted)/len(new_predicted)

    return (likely_to_be_music,new_predicted)

def voter_result (results_voter):
    # Fazendo votador
    results_final = [sum(x) for x in zip(*results_voter)]
    for i, result in enumerate(results_final):
        if result >= int(len(results_voter)/2.0 + 0.5) :
           results_final[i] = 1
        else:
            results_final[i] = 0
    likely_to_be_music = sum(results_final)/len(results_final)
    
    return likely_to_be_music    

def main(audio_file, algo):
    variables_to_extract = feat_config[algo]

    # Create list of csv files created
    vex.main(audio_file, variables_to_extract)
    
    sec_by_sec_prediction_array = []
    likely_to_be_music_array = []

    if algo.find('voter') == -1:
        # If not a voter
        used_clf_list = [algo]
        use_voter = False
    else:
        # If a voter
        used_clf_list = feat_config['{}_algo'.format(algo)]
        use_voter = True

    for clf in used_clf_list:
        results = predict_audio(feat_config[clf + '_folder'], feat_config[clf+ '_features'], clf)
        likely_to_be_music, prediction_sec_by_sec = results

        sec_by_sec_prediction_array.append(prediction_sec_by_sec)
        likely_to_be_music_array.append(likely_to_be_music)
        
        print ('Used Classifier: {}\nProbability to be music: {}%'.format(clf, likely_to_be_music * 100))
    
    if use_voter:
        likely_to_be_music_voter = voter_result(sec_by_sec_prediction_array)

        print ('Final Result with Voter:\nProbability to be music: {}%'.format(likely_to_be_music_voter * 100))
    
    #create results to return
    to_return = {'audio_file': basename(audio_file)}
    for i, value in enumerate(sec_by_sec_prediction_array):
        to_return[used_clf_list[i]] = {'sec_by_sec_music_prediction':value, "likely_to_be_music": likely_to_be_music_array[i]}
    if use_voter:
        to_return['voter'] = likely_to_be_music_voter
        
    return json.dumps(to_return, ensure_ascii=False)


if __name__ == '__main__':
    a = time.time()
    clf = 'svm'#input('Used classifier: ')
    # audio_file = '/home/felipe/Downloads/80s_2017_11_22__12_36_00.wav'
    audio_file = '/home/felipe/Downloads/80s_2017_11_22__12_13_57.wav'#input('Audio file path: ')
    print (main(audio_file, clf))
    print ('Final time '+ str(time.time()-a) )
