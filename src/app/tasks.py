import datetime
import pickle

from metabolitics.preprocessing import MetaboliticsPipeline
import celery
from .models import db, Analysis, Dataset, MetabolomicsData, Disease
from .services.mail_service import *
import json
import requests
from libchebipy import ChebiEntity
import os
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from collections import OrderedDict
from sklearn.model_selection import cross_val_score, StratifiedKFold


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
    print('Enhancing synonyms...')
    with open('../datasets/assets/synonyms.json') as f:
        synonyms = json.load(f, object_pairs_hook=OrderedDict)
    with open('../datasets/assets/recon2.json') as f:
        recon2 = json.load(f)
    recon2_metabolites = recon2['metabolites'].keys()
    recon2_metabolites = list(recon2_metabolites)
    for key, value in data['analysis'].items():
        metabolites = value['Metabolites']
        for metabolite in metabolites:
            try:
                if '_' not in metabolite:
                    continue
                bigg_id = metabolite[:metabolite.rindex('_')]
                bigg_url = 'http://bigg.ucsd.edu/api/v2/universal/metabolites/' + bigg_id
                bigg_response = requests.get(bigg_url).json()
                bigg_compartments = bigg_response['compartments_in_models']
                compartments = set(bigg_compartment['bigg_id'] for bigg_compartment in bigg_compartments)
                bigg_ids = [bigg_id + '_' + compartment for compartment in compartments if bigg_id + '_' + compartment in recon2_metabolites]
                chebi_links = bigg_response['database_links']['CHEBI']
                for link in chebi_links:
                    chebi_id = link['id']
                    chebi_entity = ChebiEntity(chebi_id)
                    chebi_synonyms = chebi_entity.get_names()
                    for synonym in chebi_synonyms:
                        synonym = synonym.get_name()
                        if not synonym in synonyms.keys() and len(bigg_ids) != 0:
                            synonyms.update({synonym : bigg_ids})
            except:
                pass
    with open('../datasets/assets/synonyms.json', 'w') as f:
        json.dump(synonyms, f, indent=4) 
    print("Enhancing synonyms done.")

@celery.task(name='train_save_model')
def train_save_model():
    print('Training and saving models...')
    dataset_ids = db.session.query(Analysis.dataset_id).filter(Analysis.label != 'not_provided').distinct()
    for dataset_id, in dataset_ids:
        path = '../trained_models/analysis' + str(dataset_id) + '_model.p'
        if os.path.isfile(path):
            continue
        metabolomics_data_ids = db.session.query(Analysis.metabolomics_data_id).filter(Analysis.dataset_id == dataset_id)
        metabolomics_data_ids = metabolomics_data_ids.filter(Analysis.label.notlike('%Group Avg%')).filter(Analysis.label.notlike('%label avg%')).all()
        metabolomics_datum = db.session.query(MetabolomicsData.metabolomics_data).filter(MetabolomicsData.id.in_(metabolomics_data_ids)).all()
        X = [metabolomics_data[0] for metabolomics_data in metabolomics_datum]
        labels = db.session.query(Analysis.label).filter(Analysis.dataset_id == dataset_id)
        labels = labels.filter(Analysis.label.notlike('%Group Avg%')).filter(Analysis.label.notlike('%label avg%')).all()
        y = [label[0] for label in labels]
        disease_id, = db.session.query(Dataset.disease_id).filter(Dataset.id == dataset_id).first()
        disease, = db.session.query(Disease.name).filter(Disease.id == disease_id).first()
        try:
            pipe = Pipeline([
                ('vect', DictVectorizer(sparse=False)),
                ('pca', PCA()),
                ('clf', LogisticRegression(C=0.3e-6, random_state=43))
            ])
            model = pipe.fit(X, y)
            kf = StratifiedKFold(n_splits=2, shuffle=True, random_state=43)
            scores = cross_val_score(pipe, X, y, cv=kf, n_jobs=None, scoring='f1_micro')
            score = scores.mean().round(3)
            save = {}
            save['disease'] = disease
            save['model'] = model
            save['score'] = score
            with open(path, 'wb') as f:
                pickle.dump(save, f)
        except Exception as e:
            print(e)
    print('Training and saving models done.')