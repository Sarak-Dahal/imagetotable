#__init__ will bring our application together

from flask import Flask
app = Flask(__name__,template_folder='../templates',static_folder='../static')
app.secret_key='developedbysarakdahal9861392262'
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['MAX_CONTENT_LENGTH'] = 16*1024*1024
UPLOAD_FOLDER = 'static/uploads/'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

from app import main

