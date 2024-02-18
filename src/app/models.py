import uuid
import datetime
import json

from sqlalchemy import and_, or_
from sqlalchemy.types import Float
from sqlalchemy.dialects.postgresql import JSON
from flask_sqlalchemy import SQLAlchemy, BaseQuery
from flask_jwt import jwt_required, current_identity, _jwt_required

from .app import app

db = SQLAlchemy(app)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255))
    surname = db.Column(db.String(255))
    email = db.Column(db.String(255), unique=True)
    affiliation = db.Column(db.String(255))
    password = db.Column(db.String(255))  # TODO: hash password
    analysis = db.relationship(
        "Analysis", back_populates="user", lazy='dynamic')

    def __repr__(self):
        return self.email

class Method(db.Model):
    __tablename__ = 'methods'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String())

    def __repr__(self):
        return '<Method %r>' % self.name

class MetabolomicsData(db.Model):
    __tablename__ = 'metabolomicsdata'
    id = db.Column(db.Integer, primary_key=True)
    metabolomics_data = db.Column(JSON)
    owner_email = db.Column(db.String())
    owner_user_id = db.Column(db.Integer, nullable=True)
    is_public = db.Column(db.Boolean)
    disease_id = db.Column(db.Integer, db.ForeignKey('diseases.id'), nullable=True)
    disease = db.relationship('Disease')

    def __repr__(self):
        return '<MetabolomicsData %r>' % self.owner_email

class Disease(db.Model):
    __tablename__ = 'diseases'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String())
    parent_id = db.Column(db.Integer, db.ForeignKey('diseases.id'), nullable=True)
    disease = db.relationship('Disease')
    synonym = db.Column(db.String())

    def __repr__(self):
        return '<Disease %r>' % self.name

class Dataset(db.Model):
    __tablename__ = 'datasets'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String())
    method_id = db.Column(db.Integer, db.ForeignKey('methods.id'))
    method = db.relationship('Method')
    status = db.Column(db.Boolean)
    group = db.Column(db.String())
    disease_id = db.Column(db.Integer, db.ForeignKey('diseases.id'), nullable=True)
    disease = db.relationship('Disease')

    def __repr__(self):
        return '<Disease %r>' % self.name

class Analysis(db.Model):
    __tablename__ = 'analysis'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255))
    # status = db.Column(db.Boolean)
    type = db.Column(db.String(255))
    start_time = db.Column(db.DateTime, nullable=True)
    end_time = db.Column(db.DateTime, nullable=True)
    results_pathway = db.Column(JSON)
    results_reaction = db.Column(JSON)
    owner_user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    user = db.relationship("User")
    owner_email = db.Column(db.String(255))
    metabolomics_data_id = db.Column(db.Integer, db.ForeignKey('metabolomicsdata.id'))
    metabolomics_data = db.relationship('MetabolomicsData')
    # method_id = db.Column(db.Integer, db.ForeignKey('methods.id'))
    # method = db.relationship('Method')
    dataset_id = db.Column(db.Integer, db.ForeignKey('datasets.id'))
    dataset = db.relationship('Dataset')
    label = db.Column(db.String())


    class AnalysisQuery(BaseQuery):
        def get_pathway_score(self, pathway):
            return Analysis.results_pathway[0][pathway].astext.cast(Float)

        def filter_by_change(self, pathway, change):
            score = self.get_pathway_score(pathway)
            return self.filter(score > 0 if change >= 0 else score < 0)

        def for_many(self, iterable, func):
            f = self
            for i in iterable:
                f = func(i)
            return f

        def filter_by_change_many(self, data):
            return self.for_many(
                data,
                lambda x: self.filter_by_change(x['pathway'], x['change']))

        def filter_by_change_amount(self, pathway, qualifier, amount):
            if not (qualifier and amount):
                return self

            score = self.get_pathway_score(pathway)
            if qualifier == 'lt':
                return self.filter(amount >= score)
            elif qualifier == 'gt':
                return self.filter(score >= amount)
            elif qualifier == 'eq':
                return self.filter(
                    or_(score + 10 >= amount, score - 10 <= amount))
            else:
                raise ValueError(
                    'qualifier should be lt, gt or eq but not %s ' % qualifier)

        def filter_by_change_amount_many(self, data):
            return self.for_many(
                data,
                lambda x: self.filter_by_change_amount(x['pathway'], x['qualifier'], x['amount'])
            )

        def filter_by_authentication(self):
            filter_type = Analysis.type.in_(['public', 'disease'])

            try:
                _jwt_required(app.config['JWT_DEFAULT_REALM'])
            except:
                pass

            if not current_identity:
                return self.filter(filter_type)
            return self.filter(
                or_(filter_type, Analysis.user.has(id=current_identity.id)))

    query_class = AnalysisQuery

    def __init__(self, name, user, status=False, type='private'):
        self.name = name
        self.status = status
        self.type = type
        self.user = user

    def clean_name_tag(self, dataset):
        cleaned_dataset = list()
        for d in dataset:
            cleaned_dataset.append({k: v for k, v in d.items()})
        return cleaned_dataset

    def authenticated(self):
        if self.type in ['private', 'noise']:
            _jwt_required(app.config['JWT_DEFAULT_REALM'])
            return self.owner_user_id == current_identity.id
        return True

    @staticmethod
    def get_multiple(ids):
        return Analysis.query.filter(
            Analysis.id.in_(ids)).filter_by_authentication()

    def __repr__(self):
        return '<Analysis %r>' % self.name

class DiseaseModel(db.Model):
    __tablename__ = 'diseasemodels'
    id = db.Column(db.Integer, primary_key=True)
    disease_id = db.Column(db.Integer, db.ForeignKey('diseases.id'))
    disease = db.relationship('Disease')
    fold_number = db.Column(db.Integer)
    f1_score = db.Column(db.Float())
    precision_score = db.Column(db.Float())
    recall_score = db.Column(db.Float())
    creation_date = db.Column(db.DateTime)
    file_path = db.Column(db.String())
    algorithm = db.Column(db.String())

    def __repr__(self):
        return '<DiseaseModel %r>' % self.id