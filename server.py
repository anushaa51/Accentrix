import os
from flask import Flask, request, render_template
from flask import jsonify
from Accentrix import accentrix

from flask_cors import CORS

app = Flask(__name__)
CORS(app)

ALLOWED_EXTENSIONS = {'wav'}

app.config['UPLOAD_FOLDER'] = "./Accentrix/"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    res = { 'message': 'OK' }
    return res

@app.route('/lang')
def lang():
    supported_lang = {
        'from': [{
            'name': 'English (India)',
            'code': 'en-in'
        }],
        'to':
        [{
            'name': 'English (US)',
            'code': 'en-us'
        }]
    }
    message = {
        'status': 200,
        'message': 'OK',
        'lang': supported_lang
    }
    resp = jsonify(message)
    resp.status_code = 200
    return resp


@app.route('/process', methods=['GET', 'POST'])
def process():

    try:

        file = request.files['audio_file']
        from_accent = request.form['from']
        to_accent = request.form['to']

        if not allowed_file(file.filename):
            print("File type not allowed")
            raise Exception("File type not allowed")
            

        file.save(os.path.join(app.config['UPLOAD_FOLDER'], "temp.wav"))


        results_dict = accentrix.get_results("./Accentrix/temp.wav", from_accent, to_accent)

        results_dict['mfcc_input'] = results_dict['mfcc_input'].decode('utf-8')
        results_dict['mfcc_output'] = results_dict['mfcc_output'].decode('utf-8')
        results_dict['classifier1'] = results_dict['classifier1'].decode('utf-8')
        results_dict['classifier2'] = results_dict['classifier2'].decode('utf-8')
        results_dict['converter1'] = results_dict['converter1'].decode('utf-8')
        results_dict['converter2'] = results_dict['converter2'].decode('utf-8')

        return (jsonify(results_dict))

    except Exception as e:
        return {'failed': True, 'reason': str(e)}



if __name__ == '__main__':
    app.run(debug = False, threaded = False)