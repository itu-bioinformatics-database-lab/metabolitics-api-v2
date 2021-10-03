from functools import reduce
from flask import jsonify, request
from flask_jwt import jwt_required, current_identity
from sqlalchemy import and_
from sqlalchemy.types import Float
from ..utils import similarty_dict
from ..visualization import HeatmapVisualization
import time
from ..app import app
from ..schemas import *
from ..models import db, User, Analysis, MetabolomicsData, Method, Dataset, Disease
from ..tasks import save_analysis
from ..base import *
from ..dpm import *
import datetime
from ..services.mail_service import *



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
        for key,value in data["analysis"].items():  # user as key, value {metaboldata , label}
            if len(value['Metabolites']) > 0:
                metabolomics_data = MetabolomicsData(
                    metabolomics_data = value["Metabolites"],
                    owner_email = str(user),
                    is_public = True if request.json['public'] else False
                )

                metabolomics_data.disease_id = disease.id
                metabolomics_data.disease = disease
                db.session.add(metabolomics_data)
                db.session.commit()


                analysis = Analysis(name=key, user=user)
                analysis.name = key
                analysis.type = 'public' if request.json['public'] else "private"
                analysis.start_time = datetime.datetime.now()


                analysis.owner_user_id = user.id
                analysis.owner_email = user.email
                analysis.metabolomics_data_id = metabolomics_data.id
                analysis.dataset_id = study.id

                db.session.add(analysis)
                db.session.commit()
                save_analysis.delay(analysis.id, value["Metabolites"])
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
        for key,value in data["analysis"].items():  # user as key, value {metaboldata , label}

            if len(value['Metabolites']) > 0:

                metabolomics_data = MetabolomicsData(
                    metabolomics_data = value["Metabolites"],
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
                analysis.start_time = datetime.datetime.now()
                analysis.end_time = datetime.datetime.now()

                analysis.owner_user_id = user.id
                analysis.owner_email = user.email

                analysis.metabolomics_data_id = metabolomics_data.id
                analysis.dataset_id = study.id
                analysis_runs = DirectPathwayMapping(value["Metabolites"])  # Forming the instance
                # fold_changes
                analysis_runs.run()  # Making the analysis
                analysis.results_pathway = [analysis_runs.result_pathways]
                analysis.results_reaction = [analysis_runs.result_reactions]

                db.session.add(analysis)
                db.session.commit()
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
                analysis.end_time = datetime.datetime.now()

                analysis.owner_user_id = user.id
                analysis.owner_email = request.json["email"]

                analysis.metabolomics_data_id = metabolomics_data.id
                analysis.dataset_id = study.id
                analysis_runs = DirectPathwayMapping(value["Metabolites"])  # Forming the instance
                # fold_changes
                analysis_runs.run()  # Making the analysis
                analysis.results_pathway = [analysis_runs.result_pathways]
                analysis.results_reaction = [analysis_runs.result_reactions]

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
    y = [i.name for i in analyses]

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

    row_disease_analyses = Analysis.query.filter_by(
        type='public').with_entities(Analysis.name,
                                      Analysis.results_pathway).all()

    names, disease_analyses = zip(*[(i[0], i[1][0])
                                    for i in row_disease_analyses])

    sims = similarty_dict(analysis.results_pathway[0], list(disease_analyses))
    top_5 = sorted(zip(names, sims), key=lambda x: x[1], reverse=True)[:5]
    return jsonify(dict(top_5))


@app.route('/analysis/<type>')
def analysis_details(type):
    # print('reached')
    idss = [] # for expanding menu
    data = Dataset.query.all()
    returned_data = []
    for item in data:
        analyses = Analysis.query.filter_by(type='public', dataset_id=item.id).with_entities(
            Analysis.id, Analysis.name, Analysis.dataset_id)
        method = Method.query.get(item.method_id)
        if len(list(analyses)) > 0:
            analysis_data = []
            for analysis in analyses:
                analysis_data.append({'id': analysis[0], 'name': analysis[1]})
                idss.append({'id':analysis[0]})
            returned_data.append({
                'id': item.id,
                'name': item.name,
                'analyses': analysis_data,
                'method': method.name,
                'disease': 'Breast Cancer',
                'id2':idss
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
            Analysis.id, Analysis.name, Analysis.dataset_id)
            method = Method.query.get(item.method_id)
            if len(list(analyses)) > 0:
                analysis_data = []
                for analysis in analyses:
                    analysis_data.append({'id': analysis[0], 'name': analysis[1]})
                returned_data.append({
                    'id': item.id,
                    'name': item.name,
                    'analyses': analysis_data,
                    'method': method.name,
                    'disease': 'Breast Cancer'
                })

    return jsonify(returned_data)

@app.route('/analysis/detail/<id>')
def analysis_detail(id):
    analysis = Analysis.query.get(id)
    metabolomics_data = MetabolomicsData.query.get(analysis.metabolomics_data_id)
    study = Dataset.query.get(analysis.dataset_id)
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
        data['analyses'].append({
            'id': analysis.id,
            'name': analysis.name
        })
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
        with open('../datasets/assets/recon2.json') as f:
            mapping_data1 = json.load(f)
            mapping_data1 = mapping_data1["metabolites"]

        with open('../datasets/assets/synonyms_v.0.4.json') as f:
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
                    print(type(metabolites[i]))
                    temp['Metabolites'][i] = float(str(metabolites[i]).strip())

                # elif i in mapping_metabolites.keys():
                #     temp['Metabolites'][mapping_metabolites[i]] = metabolites[i]

            if len(temp['Metabolites']) > 0:
                output['analysis'][case] = temp
        print(output)
        return output





