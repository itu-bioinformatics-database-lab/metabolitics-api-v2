from .base import MetaboliticsBase

class DirectPathwayMapping():
    def __init__(self, concentration_table):
        self.name = "Direct Pathway Mapping"
        self.fold_changes = concentration_table
        self.base = MetaboliticsBase()
        self.result_pathways = {}
        self.result_reactions = {}

    def score_reactions(self):
        """ Scoring all the reactions 
            response:
                - dict
        """
        reactions = self.base.get_reaction_names()  # Getting reaction names
        reaction_scores = {}  # Scores will be stored
        for reaction in reactions:
            metabolites = self.base.get_metabolites_by_reaction(reaction)  # Get metabolites
            # Total is the sum of fold changes
            # Denominator is the number of participating metabolites
            total, denominator = 0, 0
            for metabolite in metabolites:
                if metabolite in self.fold_changes:
                    total += self.fold_changes[metabolite]
                    denominator += 1.0
            if denominator != 0:
                reaction_scores[reaction] = total/denominator
        return reaction_scores
    
    def score_pathways(self):
        pathways_scores, pathway_metabolites = {}, {}    # Scores will be stored
        metabolites = self.fold_changes.keys()  # Getting metabolite names of user-uploaded data
        for metabolite in metabolites:
            if metabolite in self.base.data['metabolites']:
                reactions = self.base.get_reactions_by_metabolite(metabolite)  # Getting reactions
                pathways = []
                for reaction in reactions:
                    if 'subsystem' in self.base.data['reactions'][reaction]:
                        pathways.append(self.base.data['reactions'][reaction]['subsystem'])
                subsystems = set(pathways)  # Removing duplicates
                for subsystem in subsystems:
                    pathways_scores.setdefault(subsystem, 0)
                    pathways_scores[subsystem] += self.fold_changes[metabolite]  # Scoring
                    pathway_metabolites.setdefault(subsystem, [])
                    pathway_metabolites[subsystem].append(metabolite)
        # Taking the average of pathway scores
        for pathway in pathways_scores:
            pathways_scores[pathway] = pathways_scores[pathway]/len(pathway_metabolites[pathway])
        return pathways_scores
    
    def display_pathway_scores(self):
        if len(self.result_pathways) != 0:
            for pathway, score in self.result_pathways.items():
                print("Pathway: " + pathway + " --- Score: " + str(score))
        else:
            print("No result found!")
    
    def display_reaction_scores(self):
        if len(self.result_reactions) != 0:
            for reaction, score in self.result_reactions.items():
                print("Reaction: " + reaction + " --- Score: " + str(score))
        else:
            print("No result found!")
    
    def run(self):
        self.result_pathways, self.result_reactions = self.score_pathways(), self.score_reactions()