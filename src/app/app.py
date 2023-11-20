from __future__ import absolute_import



import os

from flask import Flask
from flask_cors import CORS

import metabolitics

app = Flask(__name__)

CORS(app)

config = {
    "development": "app.config.DevelopmentConfig",
    "testing": "app.config.TestingConfig",
    "production": "app.config.ProductionConfig"
}

app.config.from_object(config[os.getenv('FLASK_CONFIGURATION', 'development')])
#
# from .celery2 import celery
# from .models import *
# from .auth import *
# from .schemas import *
# from .views import *
# from .admin import *

def change_content(file1, file2):
    with open(file1, 'r') as f:
        cont = f.read()
    with open(file2, 'w') as f:
        f.write(cont)

def update_recon3D():
    path = metabolitics.__file__
    path = path[:path.rindex('/')] + '/datasets'
    change_content('../datasets/assets/metabolitics/recon3D.json', path + '/network_models/recon2.json')
    change_content('../datasets/assets/metabolitics/cheBl-mapping.json', path + '/naming/cheBl-mapping.json')
    change_content('../datasets/assets/metabolitics/hmdb-mapping.json', path + '/naming/hmdb-mapping.json')
    change_content('../datasets/assets/metabolitics/kegg-mapping.json', path + '/naming/kegg-mapping.json')
    change_content('../datasets/assets/metabolitics/pubChem-mapping.json', path + '/naming/pubChem-mapping.json')
    change_content('../datasets/assets/metabolitics/toy-mapping.json', path + '/naming/toy-mapping.json')

update_recon3D()