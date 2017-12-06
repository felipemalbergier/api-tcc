from flask import Flask, jsonify, request, flash, redirect, url_for, send_from_directory, session
import os
import predict_audio
from datetime import datetime


app = Flask(__name__)

ALLOWED_EXTENSIONS = set(['txt', 'wav'])
UPLOAD_FOLDER = os.getcwd()+'/files/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.isdir(UPLOAD_FOLDER): os.mkdir(UPLOAD_FOLDER)

startup_time = datetime.now()
@app.route("/")
def hello():
    global startup_time
    return "Api is up since:{}".format(startup_time)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

# classificadores knn, svm, nn, knne,svme, nne, voter, votere
@app.route('/speechmusic/<string:used_clf>', methods=['GET', 'POST'])
def upload_file(used_clf):
    
    if request.method == 'POST':
        
        print (request.files)
        file = request.files['file']
        # check if the post request has the file part
        if 'file' not in request.files:
            print('No file')            
            flash('No file part')
            return redirect(request.url)
        
        # check if file is empty
        if file.filename == '':
            print('Empty File')
            flash('No selected file')
            return redirect(request.url)
            
        # check if file is allowed
        if file and allowed_file(file.filename): 
            filename = file.filename
            
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print ('SALVOU EM:{}'.format(os.path.join(app.config['UPLOAD_FOLDER'], filename)))
            # predict if is speech or music
            response = predict_audio.main(UPLOAD_FOLDER + filename, used_clf)

            return response

    else: return '''
    <!doctype html>
    <title>Faça upload do arquivo de áudio que quiser</title>
    <h1>Upload new File</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''

if __name__ == '__main__':
    app.secret_key = "uashudhaflhafjklsdhjfkabnjkbak389493yrfh378h443h87gf87"
    # app.run(host='10.128.0.4',port=80)
    app.run(debug = True)
    
