from .base import MetaboliticsBase
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

class PathwayEnrichment():
    def __init__(self, concentration_table):
        self.name = "Pathway Enrichment"
        self.fold_changes = concentration_table
        self.base = MetaboliticsBase()
        self.result_pathways = {}
        self.result_reactions = {}

    def score_reactions(self):
        """ Scoring all the reactions 
            response:
                - dict
        """
        reactions_scores = {}
        all_metabolites = self.base.get_metabolite_names()
        N = len(all_metabolites)
        all_reactions = self.base.get_reaction_names()
        data_metabolites = self.fold_changes.keys()
        n = len(data_metabolites)

        p_values = []
        for reaction in all_reactions:
            reaction_metabolites = self.base.get_metabolites_by_reaction(reaction)
            K = len(reaction_metabolites)
            common_metabolites = [value for value in data_metabolites if value in reaction_metabolites]
            k = len(common_metabolites)
            pval = hypergeom.sf(k-1, N, K, n)
            #reactions_scores[reaction] = pval
            p_values.append(pval)

        q_values = multipletests(p_values, method="fdr_bh")[1]

        for reaction, q_value in zip(all_reactions, q_values):
            reactions_scores[reaction] = q_value
        return reactions_scores
    
    def score_pathways(self):
        """ Scoring all the pathways 
            response:
                - dict
        """
        pathways_scores = {}
        all_metabolites = self.base.get_metabolite_names()
        N = len(all_metabolites)
        all_pathways = self.base.get_pathway_names()
        data_metabolites = self.fold_changes.keys()
        n = len(data_metabolites)

        p_values = []
        for pathway in all_pathways:
            pathway_metabolites = self.base.get_metabolites_by_pathway(pathway)
            K = len(pathway_metabolites)
            common_metabolites = [value for value in data_metabolites if value in pathway_metabolites]
            k = len(common_metabolites)
            pval = hypergeom.sf(k-1, N, K, n)
            #pathways_scores[pathway] = pval
            p_values.append(pval)
        
        q_values = multipletests(p_values, method="fdr_bh")[1]
        
        for reaction, q_value in zip(all_pathways, q_values):
            pathways_scores[reaction] = q_value
        
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
