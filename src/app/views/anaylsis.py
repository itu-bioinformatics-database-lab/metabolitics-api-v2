from functools import reduce
from flask import jsonify, request
from flask_jwt import jwt_required, current_identity
from sqlalchemy import and_, or_
from sqlalchemy.types import Float
from ..utils import similarty_dict
from ..visualization import HeatmapVisualization
import time
from ..app import app
from ..schemas import *
from ..models import db, User, Analysis, MetabolomicsData, Method, Dataset, Disease
from ..tasks import save_analysis, enhance_synonyms, save_dpm, save_pe
from ..base import *
from ..dpm import *
import datetime
from ..services.mail_service import *
import os
import pickle
from ..pe import *
from metabolitics3d.preprocessing import MetaboliticsPipeline
import sys



@app.route('/analysis/fva', methods=['POST'])
@jwt_required()
def fva_analysis():
    """
    FVA analysis
    ---
    tags:
      - analysis
    parameters:
        -
          name: authorization
          in: header
          type: string
          required: true
        - in: body
          name: body
          schema:
            id: AnalysisInput
            required:
              - name
              - concentration_changes
            properties:
              name:
              name:
                  type: string
                  description: name of analysis
              concentration_changes:
                  type: object
                  description: concentration changes of metabolitics
    responses:
      200:
        description: Analysis info
      404:
        description: Analysis not found
      401:
        description: Analysis is not yours
    """

    (data, error) = AnalysisInputSchema().load(request.json)
    if error:
        return jsonify(error), 400
    if not request.json:
        return "", 404

    # if 'metabolites' in data:
    #     enhance_synonyms.delay(data['metabolites'])

    data = checkMapped(data)


    user = User.query.filter_by(email=str(current_identity)).first()
    #

    if len(data['analysis']) == 0:
        return jsonify({'id': 'mapping_error'})

    else:

        disease = Disease.query.get(request.json['disease'])
        study = Dataset(
            name=request.json['study_name'],
            method_id=1,
            group=request.json["group"],
            disease_id=disease.id,
            disease=disease)
        db.session.add(study)
        db.session.commit()

        analysis_id = 0
        healthy_data = None
        for key,value in data['analysis'].items():
            if len(value['Metabolites']) > 0:
                if value['Label'] == data['group'].lower() + ' label avg':
                    healthy_data = value['Metabolites']
        for key,value in data["analysis"].items():  # user as key, value {metaboldata , label}
            if len(value['Metabolites']) > 0:
                if healthy_data != None:
                    pipe = MetaboliticsPipeline([
                        'fold-change-scaler',
                    ])
                    for k, v in value["Metabolites"].items():
                        value["Metabolites"][k] = v if v != 0 else sys.float_info.min
                    for k, v in healthy_data.items():
                        healthy_data[k] = v if v != 0 else sys.float_info.min
                    X_t = pipe.fit_transform([value["Metabolites"], healthy_data], [value['Label'], 'healthy'])[0]
                metabolomics_data = MetabolomicsData(
                    metabolomics_data = value["Metabolites"] if healthy_data == None else X_t,
                    owner_email = str(user),
                    is_public = True if request.json['public'] else False
                )

                metabolomics_data.disease_id = disease.id
                metabolomics_data.disease = disease
                db.session.add(metabolomics_data)
                db.session.commit()


                analysis = Analysis(name=key, user=user)
                analysis.label = value['Label']
                analysis.name = key
                analysis.type = 'public' if request.json['public'] else "private"


                analysis.owner_user_id = user.id
                analysis.owner_email = user.email
                analysis.metabolomics_data_id = metabolomics_data.id
                analysis.dataset_id = study.id

                db.session.add(analysis)
                db.session.commit()
                save_analysis.delay(analysis.id, value["Metabolites"] if healthy_data == None else X_t)
                analysis_id = analysis.id

        return jsonify({'id': analysis_id})



###############

# Analysis FVA Public

@app.route('/analysis/fva/public', methods=['POST'])
def fva_analysis_public():

    (data, error) = AnalysisInputSchema2().load(request.json)
    if error:
        return jsonify(error), 400
    if not request.json:
        return "", 404
    # print(request.json)

    counter = 1
    check_value = len(list(request.json['analysis'].keys()))

    # if 'metabolites' in data:
    #     enhance_synonyms.delay(data['metabolites'])

    data = checkMapped(data)

    user = User.query.filter_by(email='tajothman@std.sehir.edu.tr').first()
    if len(data['analysis']) == 0:
        return jsonify({'id': 'mapping_error'})

    else:


        disease = Disease.query.get(request.json['disease'])
        study = Dataset(
            name=request.json['study_name'],
            method_id=1,
            group=request.json["group"],
            disease_id=disease.id,
            disease=disease)
        db.session.add(study)
        db.session.commit()

    #
        for key, value in data["analysis"].items():  # user as key, value {metaboldata , label}
            if len(value['Metabolites']) > 0:
                check_value -=1

                metabolomics_data = MetabolomicsData(
                    metabolomics_data=value["Metabolites"],
                    owner_email=request.json["email"],
                    is_public=True
                )

                metabolomics_data.disease_id = disease.id
                metabolomics_data.disease = disease
                db.session.add(metabolomics_data)
                db.session.commit()

                analysis = Analysis(name=key, user=user)
                analysis.label = value['Label']
                analysis.name = key
                analysis.type = 'public'
                analysis.start_time = datetime.datetime.now()

                analysis.owner_user_id = user.id
                analysis.owner_email = request.json["email"]
                analysis.metabolomics_data_id = metabolomics_data.id
                analysis.dataset_id = study.id

                db.session.add(analysis)
                db.session.commit()

                if check_value == counter:
                    save_analysis.delay(analysis.id, value["Metabolites"],registered=False,mail=request.json["email"],study2=request.json['study_name'])
                else:
                    counter+=1
                    save_analysis.delay(analysis.id, value["Metabolites"])


        return jsonify({'id': analysis.id})
        # return jsonify({1:1})



#### direct pathway analysis

@app.route('/analysis/direct-pathway-mapping', methods=['GET', 'POST'])
@jwt_required()
def direct_pathway_mapping():

    (data, error) = AnalysisInputSchema().load(request.json)
    if error:
        return jsonify(error), 400
    if not request.json:
        return "", 404

    # if 'metabolites' in data:
    #     enhance_synonyms.delay(data['metabolites'])

    data = checkMapped(data)

    user = User.query.filter_by(email=str(current_identity)).first()

    if len(data['analysis']) == 0:
        return jsonify({'id': 'mapping_error'})

    else:


        disease = Disease.query.get(request.json['disease'])
        study = Dataset(
            name=data['study_name'],
            method_id=2,
            status=True,
            group=data["group"],
            disease_id=disease.id,
            disease=disease)
        db.session.add(study)
        db.session.commit()
        analysis_id = 0
        healthy_data = None
        for key,value in data['analysis'].items():
            if len(value['Metabolites']) > 0:
                if value['Label'] == data['group'].lower() + ' label avg':
                    healthy_data = value['Metabolites']
        for key,value in data["analysis"].items():  # user as key, value {metaboldata , label}

            if len(value['Metabolites']) > 0:

                if healthy_data != None:
                    pipe = MetaboliticsPipeline([
                        'fold-change-scaler',
                    ])
                    for k, v in value["Metabolites"].items():
                        value["Metabolites"][k] = v if v != 0 else sys.float_info.min
                    for k, v in healthy_data.items():
                        healthy_data[k] = v if v != 0 else sys.float_info.min
                    X_t = pipe.fit_transform([value["Metabolites"], healthy_data], [value['Label'], 'healthy'])[0]
                metabolomics_data = MetabolomicsData(
                    metabolomics_data = value["Metabolites"] if healthy_data == None else X_t,
                    owner_email = str(user),
                    is_public = True if request.json['public'] else False
                )

                metabolomics_data.disease_id = disease.id
                metabolomics_data.disease = disease
                db.session.add(metabolomics_data)
                db.session.commit()

                analysis = Analysis(name =key, user = user)
                analysis.label = value['Label']
                analysis.name = key
                # analysis.status = True
                analysis.type = 'public' if request.json['public'] else "private"

                analysis.owner_user_id = user.id
                analysis.owner_email = user.email

                analysis.metabolomics_data_id = metabolomics_data.id
                analysis.dataset_id = study.id

                db.session.add(analysis)
                db.session.commit()
                save_dpm.delay(analysis.id, value["Metabolites"] if healthy_data == None else X_t)
                analysis_id = analysis.id

        return jsonify({'id': analysis_id})




### direct pathway analysis public

@app.route('/analysis/direct-pathway-mapping/public', methods=['GET', 'POST'])
def direct_pathway_mapping2():
    # print(request.json)
    (data, error) = AnalysisInputSchema2().load(request.json)
    if error:
        return jsonify(error), 400
    if not request.json:
        return "", 404

    # if 'metabolites' in data:
    #     enhance_synonyms.delay(data['metabolites'])

    data = checkMapped(data)
    user = User.query.filter_by(email='tajothman@std.sehir.edu.tr').first()
    if len(data['analysis']) == 0:
        return jsonify({'id':'mapping_error'})

    else:
        disease = Disease.query.get(request.json['disease'])
        study = Dataset(
            name=data['study_name'],
            method_id=2,
            status=True,
            group=data["group"],
            disease_id=disease.id,
            disease=disease)
        db.session.add(study)
        db.session.commit()

        analysis_id = 0
        for key,value in data["analysis"].items():  # user as key, value {metaboldata , label}

            if len(value['Metabolites']) > 0:
                metabolomics_data = MetabolomicsData(
                    metabolomics_data = value["Metabolites"],
                    owner_email = str(user),
                    is_public = True
                )
                print('ok')

                metabolomics_data.disease_id = disease.id
                metabolomics_data.disease = disease
                db.session.add(metabolomics_data)
                db.session.commit()

                analysis = Analysis(name =key, user = user)
                analysis.label = value['Label']
                analysis.name = key
                # analysis.status = True
                analysis.type = 'public'
                analysis.start_time = datetime.datetime.now()

                analysis.owner_user_id = user.id
                analysis.owner_email = request.json["email"]

                analysis.metabolomics_data_id = metabolomics_data.id
                analysis.dataset_id = study.id
                analysis_runs = DirectPathwayMapping(value["Metabolites"])  # Forming the instance
                # fold_changes
                analysis_runs.run()  # Making the analysis
                analysis.results_pathway = [analysis_runs.result_pathways]
                analysis.results_reaction = [analysis_runs.result_reactions]
                analysis.end_time = datetime.datetime.now()

                db.session.add(analysis)
                db.session.commit()
                analysis_id = analysis.id

        message = 'Hello, \n you can find your analysis results in the following link: \n http://metabolitics.itu.edu.tr/past-analysis/' + str(analysis_id)
        send_mail( request.json["email"], request.json['study_name'] + ' Analysis Results', message)
        return jsonify({'id': analysis_id})

#### pathway enrichment analysis

@app.route('/analysis/pathway-enrichment', methods=['GET', 'POST'])
@jwt_required()
def pathway_enrichment():

    (data, error) = AnalysisInputSchema().load(request.json)
    if error:
        return jsonify(error), 400
    if not request.json:
        return "", 404

    # if 'metabolites' in data:
    #     enhance_synonyms.delay(data['metabolites'])

    data = checkMapped(data)

    user = User.query.filter_by(email=str(current_identity)).first()

    if len(data['analysis']) == 0:
        return jsonify({'id': 'mapping_error'})

    else:


        disease = Disease.query.get(request.json['disease'])
        study = Dataset(
            name=data['study_name'],
            method_id=3,
            status=True,
            group=data["group"],
            disease_id=disease.id,
            disease=disease)
        db.session.add(study)
        db.session.commit()
        analysis_id = 0
        healthy_data = None
        for key,value in data['analysis'].items():
            if len(value['Metabolites']) > 0:
                if value['Label'] == data['group'].lower() + ' label avg':
                    healthy_data = value['Metabolites']
        for key,value in data["analysis"].items():  # user as key, value {metaboldata , label}

            if len(value['Metabolites']) > 0:
                if healthy_data != None:
                    pipe = MetaboliticsPipeline([
                        'fold-change-scaler',
                    ])
                    for k, v in value["Metabolites"].items():
                        value["Metabolites"][k] = v if v != 0 else sys.float_info.min
                    for k, v in healthy_data.items():
                        healthy_data[k] = v if v != 0 else sys.float_info.min
                    X_t = pipe.fit_transform([value["Metabolites"], healthy_data], [value['Label'], 'healthy'])[0]
                metabolomics_data = MetabolomicsData(
                    metabolomics_data = value["Metabolites"] if healthy_data == None else X_t,
                    owner_email = str(user),
                    is_public = True if request.json['public'] else False
                )

                metabolomics_data.disease_id = disease.id
                metabolomics_data.disease = disease
                db.session.add(metabolomics_data)
                db.session.commit()

                analysis = Analysis(name =key, user = user)
                analysis.label = value['Label']
                analysis.name = key
                # analysis.status = True
                analysis.type = 'public' if request.json['public'] else "private"

                analysis.owner_user_id = user.id
                analysis.owner_email = user.email

                analysis.metabolomics_data_id = metabolomics_data.id
                analysis.dataset_id = study.id

                db.session.add(analysis)
                db.session.commit()
                save_pe.delay(analysis.id, value["Metabolites"] if healthy_data == None else X_t)                
                analysis_id = analysis.id



        return jsonify({'id': analysis_id})

### pathway enrichment analysis public

@app.route('/analysis/pathway-enrichment/public', methods=['GET', 'POST'])
def pathway_enrichment2():
    # print(request.json)
    (data, error) = AnalysisInputSchema2().load(request.json)
    if error:
        return jsonify(error), 400
    if not request.json:
        return "", 404

    # if 'metabolites' in data:
    #     enhance_synonyms.delay(data['metabolites'])

    data = checkMapped(data)
    user = User.query.filter_by(email='tajothman@std.sehir.edu.tr').first()
    if len(data['analysis']) == 0:
        return jsonify({'id':'mapping_error'})

    else:
        disease = Disease.query.get(request.json['disease'])
        study = Dataset(
            name=data['study_name'],
            method_id=3,
            status=True,
            group=data["group"],
            disease_id=disease.id,
            disease=disease)
        db.session.add(study)
        db.session.commit()

        analysis_id = 0
        for key,value in data["analysis"].items():  # user as key, value {metaboldata , label}

            if len(value['Metabolites']) > 0:
                metabolomics_data = MetabolomicsData(
                    metabolomics_data = value["Metabolites"],
                    owner_email = str(user),
                    is_public = True
                )
                print('ok')

                metabolomics_data.disease_id = disease.id
                metabolomics_data.disease = disease
                db.session.add(metabolomics_data)
                db.session.commit()

                analysis = Analysis(name =key, user = user)
                analysis.label = value['Label']
                analysis.name = key
                # analysis.status = True
                analysis.type = 'public'
                analysis.start_time = datetime.datetime.now()

                analysis.owner_user_id = user.id
                analysis.owner_email = request.json["email"]

                analysis.metabolomics_data_id = metabolomics_data.id
                analysis.dataset_id = study.id
                analysis_runs = PathwayEnrichment(value["Metabolites"])  # Forming the instance
                # fold_changes
                analysis_runs.run()  # Making the analysis
                analysis.results_pathway = [analysis_runs.result_pathways]
                #analysis.results_reaction = [analysis_runs.result_reactions]
                analysis.end_time = datetime.datetime.now()

                db.session.add(analysis)
                db.session.commit()
                analysis_id = analysis.id

        message = 'Hello, \n you can find your analysis results in the following link: \n http://metabolitics.itu.edu.tr/past-analysis/' + str(analysis_id)
        send_mail( request.json["email"], request.json['study_name'] + ' Analysis Results', message)
        return jsonify({'id': analysis_id})


###############################################################################
###############################################################################

@app.route('/analysis/set', methods=['POST'])
def user_analysis_set():

    data = request.json['data']
    # print(data)
    analyses = Analysis.get_multiple(data.values())
    # for i in analyses:
        # print(i.results_pathway[0])
    # X = [i.results_pathway for i in analyses]
    # y = [i.name for i in analyses]

    return AnalysisSchema(many=True).jsonify(analyses)
# ///////////////////////
@app.route('/analysis/visualization', methods=['POST'])
def analysis_visualization():
    """
    List of analysis of user
    ---
    tags:
        - analysis
    parameters:
        -
          name: authorization
          in: header
          type: string
          required: true
    """

    data = request.json['data']
    # print(data)
    analyses = Analysis.get_multiple(data.values())
    # print(analyses)
    # for i in analyses:
        # print(i.results_pathway[0])
    X = [i.results_pathway[0] for i in analyses]
    y = [Disease.query.get(Dataset.query.get(i.dataset_id).disease_id).name.title() for i in analyses]

    return jsonify(HeatmapVisualization(X, y).clustered_data())
    # return AnalysisSchema(many=True).jsonify(analyses)




@app.route('/analysis/most-similar-diseases/<id>')
def most_similar_diseases(id: int):
    """
    Calculates most similar disease for given disease id
    ---
    tags:
      - analysis
    parameters:
      -
        name: authorization
        in: header
        type: string
        required: true
      -
        name: id
        in: path
        type: integer
        required: true
    responses:
      200:
        description: Most similar diseases
      404:
        description: Analysis not found
      401:
        description: Analysis is not yours
    """
    analysis = Analysis.query.get(id)
    if not analysis:
        return '', 404
    if not analysis.authenticated():
        return '', 401
    analysis_method_id = Dataset.query.get(analysis.dataset_id).method_id
    groups = db.session.query(Dataset.group).all()
    groups = [group[0].lower() + ' label avg' for group in groups]
    public_analyses = db.session.query(Analysis).join(Dataset).join(Disease).filter(
        Analysis.type == 'public').filter(Dataset.method_id == analysis_method_id).filter(
            Analysis.results_pathway != None).filter(
                or_(Analysis.label == 'not_provided', and_(Analysis.label.like('%label avg%'), ~Analysis.label.in_(groups)))).with_entities(
                    Disease.name, Analysis.results_pathway, Disease.synonym).all()
    diseases = [i[0] + ' (' + i[2] + ')' for i in public_analyses]
    results_pathways = [i[1][0] for i in public_analyses]
    similarities = similarty_dict(analysis.results_pathway[0], results_pathways)
    dis_sim = zip(diseases, similarities)
    dis_sim_dict = {}
    for i in dis_sim:
        if i[0] not in dis_sim_dict:
            dis_sim_dict[i[0]] = []
        dis_sim_dict[i[0]].append(i[1])
    for i in dis_sim_dict:
        dis_sim_dict[i] = sum(dis_sim_dict[i]) / len(dis_sim_dict[i])
    top_five = sorted(dis_sim_dict.items(), key=lambda x: x[1], reverse=True)[:5]
    return jsonify(dict(top_five))

@app.route('/analysis/disease-prediction/<id>')
def disease_prediction(id: int):
    """
    Disease prediction for given analysis id using trained models
    ---
    tags:
      - analysis
    parameters:
      -
        name: authorization
        in: header
        type: string
        required: true
      -
        name: id
        in: path
        type: integer
        required: true
    responses:
      200:
        description: Disease predictions
      404:
        description: Analysis not found
      401:
        description: Analysis is not yours
    """
    analysis = Analysis.query.get(id)
    if not analysis:
        return '', 404
    if not analysis.authenticated():
        return '', 401
    results_reaction = analysis.results_reaction[0]
    dir = '../trained_models'
    preds = []
    for file in os.listdir(dir):
        if file == '.keep':
            continue
        file_path = os.path.join(dir, file)
        if os.path.isfile(file_path):
            try:
                saved = pickle.load(open(file_path, 'rb'))
                disease = saved['disease']
                model = saved['model']
                pred = model.predict([results_reaction])[0]
                pred_score = model.predict_proba([results_reaction])[0]
                pred_score = max(pred_score)
                if pred != 0:
                    preds.append({'disease' : disease, 'pred_score': round(pred_score, 3)})
            except Exception as e:
                print(e)
    return jsonify(sorted(preds, key=lambda p: p['pred_score'], reverse=True))

@app.route('/analysis/<type>')
def analysis_details(type):
    data = Dataset.query.all()
    returned_data = []
    for item in data:
        analyses = Analysis.query.filter_by(type='public', dataset_id=item.id).with_entities(
            Analysis.id, Analysis.name, Analysis.dataset_id, Analysis.start_time, Analysis.end_time)
        method = Method.query.get(item.method_id)
        disease = Disease.query.get(item.disease_id)
        group = item.group
        if len(list(analyses)) > 0:
            avg_id = -1
            analysis_data = []
            starts = []
            ends = []
            for analysis in analyses:
                if group != 'not_provided':
                    if str(group).lower() + ' label avg' == analysis[1]:
                        continue
                analysis_data.append({'id': analysis[0], 'name': analysis[1], "start": analysis[3], "end": analysis[4]})
                if analysis[3] != None:
                    starts.append(analysis[3])
                if analysis[4] != None:
                    ends.append(analysis[4])
                if group != 'not_provided':
                    if ' label avg' in analysis[1]:
                        avg_id = analysis[0]
            if len(starts) > 0:
                start = min(starts)
            else:
                start = None
            if len(ends) == len(analysis_data):
                end = max(ends)
            else:
                end = None
            returned_data.append({
                'id': item.id,
                'name': item.name,
                'analyses': analysis_data,
                'method': method.name,
                'disease': disease.name,
                'start': start,
                'end': end,
                'avg_id': analysis_data[0]['id'] if avg_id == -1 else avg_id,
                'progress': round(len(ends) / len(analysis_data) * 100) 
            })
    # print(returned_data)
    return jsonify(returned_data)

@app.route('/analysis/list')
@jwt_required()
def user_analysis():
    """
    List of analysis of user
    ---
    tags:
        - analysis
    parameters:
        -
          name: authorization
          in: header
          type: string
          required: true
    """
    data = Dataset.query.all()
    returned_data = []

    if 'Authorization Required' not in str(current_identity.id):
        for item in data:
            analyses = Analysis.query.filter_by(owner_user_id=current_identity.id, type='private', dataset_id=item.id).with_entities(
            Analysis.id, Analysis.name, Analysis.dataset_id, Analysis.start_time, Analysis.end_time)
            method = Method.query.get(item.method_id)
            disease = Disease.query.get(item.disease_id)
            group = item.group
            if len(list(analyses)) > 0:
                avg_id = -1
                analysis_data = []
                starts = []
                ends = []
                for analysis in analyses:
                    if group != 'not_provided':
                        if str(group).lower() + ' label avg' == analysis[1]:
                            continue
                    analysis_data.append({'id': analysis[0], 'name': analysis[1], 'start': analysis[3], 'end': analysis[4]})
                    if analysis[3] != None:
                        starts.append(analysis[3])
                    if analysis[4] != None:
                        ends.append(analysis[4])
                    if group != 'not_provided':
                        if ' label avg' in analysis[1]:
                            avg_id = analysis[0]
                if len(starts) > 0:
                    start = min(starts)
                else:
                    start = None
                if len(ends) == len(analysis_data):
                    end = max(ends)
                else:
                    end = None
                returned_data.append({
                    'id': item.id,
                    'name': item.name,
                    'analyses': analysis_data,
                    'method': method.name,
                    'disease': disease.name,
                    'start': start,
                    'end': end,
                    'avg_id': analysis_data[0]['id'] if avg_id == -1 else avg_id,
                    'progress': round(len(ends) / len(analysis_data) * 100) 
                })

    return jsonify(returned_data)

@app.route('/analysis/detail/<id>')
def analysis_detail(id):
    analysis = Analysis.query.get(id)
    metabolomics_data = MetabolomicsData.query.get(analysis.metabolomics_data_id)
    study = Dataset.query.get(analysis.dataset_id)
    group = study.group
    method = Method.query.get(study.method_id)
    disease = Disease.query.get(study.disease_id)
    data = {
        'case_name': analysis.name,
        'status': study.status,
        'results_pathway': analysis.results_pathway,
        'results_reaction': analysis.results_reaction,
        'method': method.name,
        'fold_changes': metabolomics_data.metabolomics_data,
        'study_name': study.name,
        'analyses': [],
        'disease': disease.name
    }
    analyses = Analysis.query.filter_by(dataset_id=study.id)
    for analysis in analyses:
        if analysis.label == str(group).lower() + ' label avg':
            healthy = {'id': analysis.id, 'name': analysis.name, 'label': 'Healthy'}
            continue
        data['analyses'].append({
            'id': analysis.id,
            'name': analysis.name,
            'label': disease.name if analysis.label != group or analysis.label == 'not_provided' else 'healthy'
        })
    if group != 'not_provided':
        data['analyses'].sort(key=lambda s: (len(s['name']), s['name']))
        for i, a in enumerate(data['analyses']):
            if ' label avg' in a['name']:
                index = i
        avg = data['analyses'][index]
        avg['Label'] = disease.name
        data['analyses'].pop(index)
        data['analyses'].insert(0, avg)
        data['analyses'].insert(1, healthy)
    return jsonify(data)




@app.route('/analysis/search-by-change', methods=['POST'])
def search_analysis_by_change():
    """
    Search query in db
    ---
    tags:
        - analysis
    parameters:
        -
          name: query
          in: url
          type: string
          required: true
    """
    (data, error) = PathwayChangesScheme().load(request.json, many=True)
    if error:
        return jsonify(error), 400
    analyses = Analysis.query.filter_by_change_many(data).filter_by_change_amount_many(data).filter_by_authentication().with_entities(Analysis.id, Analysis.name, Analysis.dataset_id)
    temp_data = {}
    for analysis in analyses:
        temp_data.setdefault(analysis.dataset_id, [])
        temp_data[analysis.dataset_id].append((analysis.id, analysis.name))
    returned_data = {}
    c = 0
    for item in temp_data:
        study = Dataset.query.get(item)
        method = Method.query.get(study.method_id)
        for (id, name) in temp_data[item]:
            returned_data[c] = {'anlysisId':study.id, 'name': study.name, 'case': id ,"method":method.name}
        c+=1

    return returned_data





@app.route('/diseases/all', methods=['GET', 'POST'])
def get_diseases():
    data = Disease.query.all()
    returned_data = []
    for item in data:
        returned_data.append({
            "id": item.id,
            "name": item.name,
            "synonym": item.synonym
        })
    return jsonify(returned_data)


############################################################# deployed but new
@app.route('/analysis/search-by-metabol', methods=['POST'])
def search_analysis_by_metabol():
    """
    Search query in db
    ---
    tags:
        - analysis
    parameters:
        -
          name: query
          in: url
          type: string
          required: true
    """
    filtered_ids = {}
    c = 0
    metabolite_name = request.json["metabol"]
    # print(metabolite_name)
    # metabolite_name = "C01507_c"
    # metabolite_measurment = 10246.0

    # change = "+"## represent up to
    # change = "-" ## represents at least
    # change = "=" ## represents around -10/+10

    ids = db.session.query(MetabolomicsData.id).all()
    for i in ids:  # loop over the Ids
        data = MetabolomicsData.query.filter_by(id=i[0]).first();
        metabolites_data = data.metabolomics_data
        if metabolite_name in list(metabolites_data) :
            analysis = Analysis.query.filter_by(metabolomics_data_id=i[0]).first();
            temp = {"anlysisId":analysis.dataset.id,'study':analysis.dataset.name,"method":analysis.dataset.method.name,'case':analysis.metabolomics_data_id,'name':metabolite_name}
            filtered_ids[c] = temp
            c+=1
    # print(filtered_ids)

    return (filtered_ids)

# if change == "+" and metabolites_data[metabolite_name] <= metabolite_measurment:
#     # print (i[0],metabolites_data[metabolite_name])
#     filtered_ids.append(i[0])
# elif change == "-" and metabolites_data[metabolite_name] >= metabolite_measurment:
#     # print (i[0],metabolites_data[metabolite_name])
#     filtered_ids.append(i[0])
# elif change == "=" and metabolites_data[metabolite_name] < metabolite_measurment+11 and metabolites_data[metabolite_name] > metabolite_measurment-11 :


################## New
def checkMapped(data):
    '''

    :param data: our data strcuture for multi cases inputs
    :return: same data strcuture but removing the unmapped metabolites
    '''

    output = {}
    output['group'] = data['group']
    output['study_name'] = data['study_name']
    output['public'] = data['public']
    output['disease'] = data['disease']
    output['study_name'] = data['study_name']
    if 'email' in data.keys():
        output['email'] = data['email']

    output.setdefault('analysis', {})

    if 'isMapped' in data.keys():

        isMapped = data['isMapped']
        for case in data['analysis'].keys():
            temp = {}
            metabolites = data['analysis'][case]['Metabolites']
            label = data['analysis'][case]['Label']
            temp['Label'] = label
            temp.setdefault('Metabolites', {})

            for i in metabolites.keys():
                if i in isMapped and isMapped[i]['isMapped'] is True:
                    temp['Metabolites'][i] = metabolites[i]
            # print(len(temp['Metabolites']))
            if len(temp['Metabolites']) > 0:
                output['analysis'][case] = temp

    # {'public': True, 'analysis': {'NIDDK1': {'Label': 'not_provided', 'Metabolites': {}}},
    #  'study_name': 'LIPID MAPS Lipidomics studies', 'group': 'not_provided', 'email': 'tajothman@std.sehir.edu.tr',
    #  'disease': 147}

        return output
    else:
        mapping_metabolites = {}

        # mapping_data = open("../datasets/assets/mapping_all.txt", "r+").readlines()
        # for line in mapping_data:
        #     tempo = line.split(",")
        #     mapping_metabolites[tempo[0].strip()] = tempo[1].strip()
        #
        with open('../datasets/assets/recon3D.json') as f:
            mapping_data1 = json.load(f)
            mapping_data1 = mapping_data1["metabolites"]

        with open('../datasets/assets/synonyms.json') as f:
            mapping_data2 = json.load(f)

        for case in data['analysis'].keys():
            temp = {}
            metabolites = data['analysis'][case]['Metabolites']
            label = data['analysis'][case]['Label']
            temp['Label'] = label
            temp.setdefault('Metabolites', {})

            for i in metabolites.keys():

                if i in mapping_data2.keys():
                    print(type(metabolites[i]))
                    temp['Metabolites'][mapping_data2[i]] = float(str(metabolites[i]).strip())

                if i in mapping_data1.keys():
                    if metabolites[i] != '':
                        print(type(metabolites[i]))
                        temp['Metabolites'][i] = float(str(metabolites[i]).strip())

                # elif i in mapping_metabolites.keys():
                #     temp['Metabolites'][mapping_metabolites[i]] = metabolites[i]

            if len(temp['Metabolites']) > 0:
                output['analysis'][case] = temp
        print(output)
        return output

@app.route('/models/scores', methods=['GET'])
def get_model_scores():
    scores = {}
    dir = '../trained_models'
    for file in os.listdir(dir):
        if file == '.keep':
            continue
        file_path = os.path.join(dir, file)
        if os.path.isfile(file_path):
            try:
                saved = pickle.load(open(file_path, 'rb'))
                disease = saved['disease']
                fold_number = saved['fold_number']
                f1_score = saved['f1_score']
                precision_score = saved['precision_score']
                recall_score = saved['recall_score']
                algorithm = saved['algorithm']
                scores[disease] = {'fold_number': fold_number, 'f1_score': f1_score, 'precision_score': precision_score, 'recall_score': recall_score, 'algorithm': algorithm}
            except Exception as e:
                print(e)
    return jsonify(scores)