import datetime
import pickle

from metabolitics3d.preprocessing import MetaboliticsPipeline
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

    reaction_scaler['metabolitics-transformer'].analyzer.model.solver = 'cplex'

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
    with open('../datasets/assets/recon3D.json') as f:
        recon3d = json.load(f)
    recon3d_metabolites = recon3d['metabolites'].keys()
    recon3d_metabolites = list(recon3d_metabolites)
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
                bigg_ids = [bigg_id + '_' + compartment for compartment in compartments if bigg_id + '_' + compartment in recon3d_metabolites]
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
    with open('../datasets/assets/recon3D.json') as f:
        recon3d = json.load(f)
    reactions = recon3d['reactions']
    disease_ids = db.session.query(Dataset.disease_id).filter(Dataset.group != 'not_provided').filter(Dataset.method_id == 1).distinct()
    for disease_id in disease_ids:
        disease_name = Disease.query.get(disease_id).name
        dataset_ids = db.session.query(Dataset.id).filter(Dataset.disease_id == disease_id).filter(
            Dataset.group != 'not_provided').filter(Dataset.method_id == 1).all()
        results_reactions_labels = db.session.query(Analysis).filter(Analysis.label.notlike('%label avg%')).filter(Analysis.label.notlike('%Group Avg%')).filter(
            Analysis.dataset_id.in_(dataset_ids)).filter(Analysis.results_reaction != None).with_entities(
                Analysis.results_reaction, Analysis.label).all()
        results_reactions = [value[0][0] for value in results_reactions_labels]
        labels = [value[1] for value in results_reactions_labels]
        groups = db.session.query(Dataset.group).filter(Dataset.id.in_(dataset_ids)).all()
        groups = [group[0] for group in groups]
        labels = ['healthy' if label in groups else label for label in labels]
        path = '../trained_models/' + disease_name.replace(' ', '_') + '_' + str(disease_id[0]) + '_model.p'
        try:
            X_train = []
            for results_reaction in results_reactions:
                sample = []
                for reaction in reactions:
                    if reaction in results_reaction:
                        sample.append(results_reaction[reaction])
                    else:
                        sample.append(0)
                X_train.append(sample)
            pipe = Pipeline([
                ('pca', PCA()),
                ('clf', LogisticRegression(C=0.3e-6, random_state=43))
            ])
            model = pipe.fit(X_train, labels)
            kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=43)
            scores = cross_val_score(pipe, X_train, labels, cv=kfold, n_jobs=None, scoring='f1_micro')
            score = scores.mean().round(3)
            save = {}
            save['disease_name'] = disease_name
            save['model'] = model
            save['score'] = score
            with open(path, 'wb') as f:
                pickle.dump(save, f)
        except Exception as e:
            print(e)
    print('Training and saving models done.')