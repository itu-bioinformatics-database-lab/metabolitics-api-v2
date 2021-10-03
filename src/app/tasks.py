import datetime
import pickle

from metabolitics.preprocessing import MetaboliticsPipeline
import celery
from .models import db, Analysis, Dataset
from .services.mail_service import *


@celery.task()
def save_analysis(analysis_id, concentration_changes,registered=True,mail='none',study2='none'):

    with open('../models/api_model.p', 'rb') as f:
        reaction_scaler = pickle.load(f)

    pathway_scaler = MetaboliticsPipeline([
        'pathway-transformer',
        'transport-pathway-elimination'
    ])
    # print ("-----------------------1")
    results_reaction = reaction_scaler.transform([concentration_changes])
    results_pathway = pathway_scaler.transform(results_reaction)

    analysis = Analysis.query.get(analysis_id)

    analysis.results_reaction = analysis.clean_name_tag(results_reaction)
    analysis.results_pathway = analysis.clean_name_tag(results_pathway)
    study = Dataset.query.get(analysis.dataset_id)
    study.status = True
    analysis.end_time = datetime.datetime.now()

    db.session.commit()

    if registered != True:
        message = 'Hello, \n you can find your analysis results in the following link: \n http://metabolitics.itu.edu.tr/past-analysis/'+str(analysis_id)
        send_mail(mail,study2+' Analysis Results',message)



