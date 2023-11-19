import json


class MetaboliticsBase:
    """Base class of Metabolitics """

    def __init__(self, dataset='recon3D', file_type='.json'):
        self.dataset = dataset
        self.file_type = file_type
        self.data = {}
        self.fill_data()

    def get_reactions_by_metabolite(self, metabolite):
        """Getting reactions by metabolite name"""
        try:
            return self.data['metabolites'][metabolite]['reactions']
        except:
            print("There is no key existing as " + metabolite)
            # raise KeyError

    def get_metabolites_by_reaction(self, reaction):
        """Getting metabolites by reaction name"""
        try:
            return self.data['reactions'][reaction]['metabolites'].keys()
        except:
            print("There is no key existing as " + reaction)

    def get_metabolite_names(self):
        """Getting metabolite names"""
        return self.data['metabolites'].keys()

    def get_reaction_names(self):
        """Getting reaction names"""
        return self.data['reactions'].keys()
    
    def get_pathway_names(self):
        """Getting pathway names"""
        return self.data['pathways'].keys()

    def get_metabolites_by_pathway(self, pathway):
        """
        Getting metabolites of the given pathway
            - params:
                pathway : string
            - response:
                a list of metabolites belong to given pathway
        """
        try:
            reactions = self.data['pathways'][pathway]
            metabolites = []
            for reaction in reactions:
                metabolites_by_reaction = self.data['reactions'][reaction]['metabolites']
                for metabolite in metabolites_by_reaction:
                    if not metabolite in metabolites:
                        metabolites.append(metabolite)
            return metabolites
        except:
            print("There is no key existing as " + pathway)

    def get_reactions_by_pathway(self, pathway):
        """
        Getting reactions of the given pathway
         - params:
                pathway : string
            - response:
                a list of reactions belong to given pathway
        """
        try:
            return self.data['pathways'][pathway]
        except:
            print("There is no key existing as " + pathway)

    def get_common_metabolites_for_analysis(self, pathway, fold_changes):
        """
        Getting mutual metabolites in the analysis compared to given pathway
         - params:
                pathway : string
            - response:
                a list of metabolites belong to given pathway in the analysis
        """
        metabolites = self.get_metabolites_by_pathway(pathway)
        common = [key for key in fold_changes if key in metabolites]
        return common
        

    def fill_data(self):
        """Filling the data referenced from the user"""
        with open('../datasets/assets/' + str(self.dataset) + str(self.file_type)) as data:
            if self.file_type == ".json":
                json_str = data.read()  # Processing RECON Data
                db = json.loads(json_str)  # Processing RECON Data
                self.data = db


tst = MetaboliticsBase()
# print(tst.get_metabolites_by_reaction('SO4CLtex2'))
