from __future__ import absolute_import



import os

from flask import Flask
from flask_cors import CORS

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
