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
import sys
from .dpm import *
from .pe import *


@celery.task()
def save_analysis(analysis_id, concentration_changes,registered=True,mail='none',study2='none'):

    analysis = Analysis.query.get(analysis_id)
    analysis.start_time = datetime.datetime.now()
    db.session.commit()
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
def save_dpm(analysis_id, concentration_changes):

    analysis = Analysis.query.get(analysis_id)
    analysis.start_time = datetime.datetime.now()
    db.session.commit()
    
    
    analysis_runs = DirectPathwayMapping(concentration_changes)  # Forming the instance
    # fold_changes
    analysis_runs.run()  # Making the analysis
    analysis.results_pathway = [analysis_runs.result_pathways]
    analysis.results_reaction = [analysis_runs.result_reactions]
    analysis.end_time = datetime.datetime.now()

    db.session.commit()

@celery.task()
def save_pe(analysis_id, concentration_changes):

    analysis = Analysis.query.get(analysis_id)
    analysis.start_time = datetime.datetime.now()
    db.session.commit()
    
    
    analysis_runs = PathwayEnrichment(concentration_changes)  # Forming the instance
    # fold_changes
    analysis_runs.run()  # Making the analysis
    analysis.results_pathway = [analysis_runs.result_pathways]
    analysis.results_reaction = [analysis_runs.result_reactions]
    analysis.end_time = datetime.datetime.now()

    db.session.commit()

@celery.task()
def enhance_synonyms(metabolites):
    print('Enhancing synonyms...')
    with open('../datasets/assets/synonyms.json') as f:
        synonyms = json.load(f, object_pairs_hook=OrderedDict)
    with open('../datasets/assets/refmet_recon3d.json') as f:
        refmet_recon3d = json.load(f, object_pairs_hook=OrderedDict)
    try:
        metabolite_name = '\n'.join(metabolites)
        params = {
            "metabolite_name": metabolite_name
        }
        res = requests.post("https://www.metabolomicsworkbench.org/databases/refmet/name_to_refmet_new_min.php", data=params).text.split('\n')
        res.pop(0)
        for line in res:
            if line == '':
                continue
            line = line.split('\t')
            met = line[0]
            ref = line[1]
            if ref in refmet_recon3d.keys():
                rec_id = refmet_recon3d[ref]
                if met not in synonyms.keys():
                    synonyms.update({met : rec_id})
    except Exception as e:
        print(e)
    with open('../datasets/assets/synonyms.json', 'w') as f:
        json.dump(synonyms, f, indent=4) 
    print("Enhancing synonyms done.")

@celery.task(name='train_save_model')
def train_save_model():
    print('Training and saving models...')
    disease_ids = db.session.query(Dataset.disease_id).filter(Dataset.group != 'not_provided').filter(Dataset.method_id == 1).distinct()
    for disease_id in disease_ids:
        disease_name = Disease.query.get(disease_id).name
        disease_synonym = Disease.query.get(disease_id).synonym
        dataset_ids = db.session.query(Dataset.id).filter(Dataset.disease_id == disease_id).filter(
            Dataset.group != 'not_provided').filter(Dataset.method_id == 1).all()
        results_reactions_labels = db.session.query(Analysis).filter(Analysis.label.notlike('%label avg%')).filter(
            Analysis.dataset_id.in_(dataset_ids)).filter(Analysis.results_reaction != None).with_entities(
                Analysis.results_reaction, Analysis.label).all()
        results_reactions = [value[0][0] for value in results_reactions_labels]
        labels = [value[1] for value in results_reactions_labels]
        groups = db.session.query(Dataset.group).filter(Dataset.id.in_(dataset_ids)).all()
        groups = [group[0] for group in groups]
        labels = ['healthy' if label in groups else label for label in labels]
        path = '../trained_models/' + disease_name.replace(' ', '_') + '_' + str(disease_id[0]) + '_model.p'
        try:
            pipe = Pipeline([
                ('vect', DictVectorizer(sparse=False)),
                ('pca', PCA()),
                ('clf', LogisticRegression(C=0.3e-6, random_state=43, solver='lbfgs'))
            ])
            model = pipe.fit(results_reactions, labels)
            kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=43)
            scores = cross_val_score(pipe, results_reactions, labels, cv=kfold, n_jobs=None, scoring='f1_micro')
            score = scores.mean().round(3)
            save = {}
            save['disease_name'] = str(disease_name) + ' (' + disease_synonym + ')'
            save['model'] = model
            save['score'] = score
            with open(path, 'wb') as f:
                pickle.dump(save, f)
        except Exception as e:
            print(e)
    print('Training and saving models done.')