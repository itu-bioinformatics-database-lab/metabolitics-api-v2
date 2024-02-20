import datetime
import pickle

from metabolitics3d.preprocessing import MetaboliticsPipeline
import celery
from .models import db, Analysis, Dataset, MetabolomicsData, Disease, DiseaseModel
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
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_validate
from sklearn.metrics import f1_score
import sys
from .dpm import *
from .pe import *
from sklearn_utils.preprocessing import *
from sklearn.feature_selection import VarianceThreshold, SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import random
import numpy as np
import math


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
    for disease_id, in disease_ids:
        seed = 41
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        disease_name = Disease.query.get(disease_id).name
        disease_synonym = Disease.query.get(disease_id).synonym
        dataset_ids = db.session.query(Dataset.id).filter(Dataset.disease_id == disease_id).filter(
            Dataset.group != 'not_provided').filter(Dataset.method_id == 1).all()
        results_reactions_labels = db.session.query(Analysis).filter(Analysis.type == 'public').filter(
            Analysis.dataset_id.in_(dataset_ids)).filter(Analysis.results_reaction != None).with_entities(
                Analysis.results_reaction, Analysis.label).all()
        results_reactions = [value[0][0] for value in results_reactions_labels]
        if len(results_reactions) < 12:
            continue
        labels = [value[1] for value in results_reactions_labels]
        groups = db.session.query(Dataset.group).filter(Dataset.id.in_(dataset_ids)).all()
        def is_healthy(label):
            for group, in groups:
                if group.lower() + ' label avg' == label or group == label:
                    return True
            return False
        labels = [0 if is_healthy(label) else 1 for label in labels]
        num_healthy = labels.count(0)
        num_disease = labels.count(1)
        file_path = '../trained_models/' + disease_name.replace(' ', '_') + '_' + str(disease_id) + '_model.p'
        try:
            fs = ('fs', Pipeline([
                            ('vt', DictInput(VarianceThreshold(0.01), feature_selection=True)),
                            ('skb', DictInput(SelectKBest(k=100), feature_selection=True))
                        ]))
            vect = ('vect', DictVectorizer(sparse=True))
            lr = ('lr', LogisticRegression(penalty='l1', tol=0.015, C=0.0008, intercept_scaling=0.3, solver='liblinear', max_iter=100000))
            lr_pipe = Pipeline([fs, vect, lr])
            rfc = ('rfc', RandomForestClassifier(n_estimators=100))
            rfc_pipe = Pipeline([fs, vect, rfc])
            svc = ('svc', SVC(gamma='auto', probability=True))
            svc_pipe = Pipeline([fs, vect, svc])
            pipes = [('Logistic Regression ', lr_pipe), ('Random Forest Classification', rfc_pipe), ('Support Vector Classification', svc_pipe)]
            models = []
            for algorithm, pipe in pipes:
                model = {}
                model['model'] = pipe.fit(results_reactions, labels)
                if min(num_healthy, num_disease) < 10:
                    kf = StratifiedKFold(n_splits=5)
                    fold_number = 5
                else:
                    kf = StratifiedKFold(n_splits=10)
                    fold_number = 10
                scoring = ['f1', 'precision', 'recall']
                scores = cross_validate(estimator=pipe, X=results_reactions, y=labels, scoring=scoring, cv=kf, return_train_score=False)
                f1_scores = scores['test_f1']
                precision_scores = scores['test_precision']
                recall_scores = scores['test_recall']
                model['f1_score'] = f1_scores[f1_scores != 0].mean()
                model['precision_score'] = precision_scores[precision_scores != 0].mean()
                model['recall_score'] = recall_scores[recall_scores != 0].mean()
                model['algorithm'] = algorithm
                if not math.isnan(model['f1_score']) and not math.isnan(model['precision_score']) and not math.isnan(model['recall_score']):
                    models.append(model)
            models = sorted(models, key=lambda model: model['f1_score'], reverse=True)
            model = models[0]
            if model['f1_score'] > 0.7:
                save = {}
                save['disease'] = str(disease_name) + ' (' + disease_synonym + ')'
                save['model'] = model['model']
                save['fold_number'] = fold_number
                save['f1_score'] = model['f1_score']
                save['precision_score'] = model['precision_score']
                save['recall_score'] = model['recall_score']
                save['algorithm'] = model['algorithm']
                disease_model = DiseaseModel(
                    disease_id=disease_id,
                    fold_number=fold_number,
                    f1_score=model['f1_score'],
                    precision_score=model['precision_score'],
                    recall_score=model['recall_score'],
                    creation_date=datetime.datetime.now(),
                    file_path=file_path,
                    algorithm=model['algorithm']
                )
                exists = db.session.query(DiseaseModel.id).filter_by(disease_id=disease_id).first() is not None
                if exists:
                    DiseaseModel.query.filter_by(disease_id=disease_id).delete()
                    db.session.commit()
                db.session.add(disease_model)
                db.session.commit()
                with open(file_path, 'wb') as f:
                    pickle.dump(save, f)
        except Exception as e:
            print(e)
    print('Training and saving models done.')