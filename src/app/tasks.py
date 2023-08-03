import datetime
import pickle

from metabolitics.preprocessing import MetaboliticsPipeline
import celery
from .models import db, Analysis, Dataset
from .services.mail_service import *
import json
import requests
from libchebipy import ChebiEntity


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

@celery.task()
def enhance_synonyms(data):
    print("Enhancing synonyms...")
    with open('../datasets/assets/synonyms_v.0.4.json') as f:
        synonyms_json = json.load(f)
    for key, value in data['analysis'].items():
        metabolites = value['Metabolites']
        for metabolite in metabolites:
            bigg_id = metabolite[:metabolite.rindex('_')]
            bigg_url = 'http://bigg.ucsd.edu/api/v2/universal/metabolites/' + bigg_id
            try:
                bigg_response = requests.get(bigg_url).json()
                bigg_compartments = bigg_response['compartments_in_models']
                compartments = set()
                for bigg_compartment in bigg_compartments:
                    compartments.add(bigg_compartment['bigg_id'])
                bigg_ids = []
                for compartment in compartments:
                    bigg_ids.append(bigg_id + '_' + compartment)
                chebi_links = bigg_response['database_links']['CHEBI']
                for link in chebi_links:
                    chebi_id = link['id']
                    chebi_entity = ChebiEntity(chebi_id)
                    chebi_synonyms = chebi_entity.get_names()
                    for synonym in chebi_synonyms:
                        synonym = synonym.get_name()
                        if not synonym in synonyms_json.keys():
                            synonyms_json.update({synonym:bigg_ids})
            except:
                pass
    with open('../datasets/assets/synonyms_v.0.4.json', 'w') as o:
        json.dump(synonyms_json, o) 
    print("Enhancing synonyms done.")
