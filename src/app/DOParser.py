import urllib.request
import json


class DOParser:
    def __init__(self):
        '''
            - Disease Ontology - File Names
        '''
        self.file_names = ["DO_FlyBase_slim",
                           "DO_AGR_slim",
                           "DO_GXD_slim",
                           "DO_IEDB_slim",
                           "DO_MGI_slim",
                           "DO_cancer_slim",
                           "DO_rare_slim",
                           "GOLD",
                           "NCIthesaurus",
                           "TopNodes_DOcancerslim",
                           "gram-negative_bacterial_infectious_disease",
                           "gram-positive_bacterial_infectious_disease",
                           "sexually_transmitted_infectious_disease",
                           "tick-borne_infectious_disease",
                           "zoonotic_infectious_disease"]
        self.diseases = {} # {file_name: [Disease Name]}
        self.disease_synonym = []

    def parse(self, name):
        '''
            params: 
                    - name: file name
            description:
                    This function parses the data from webpage of Disease Ontology with the given name and adds to the class' itself.
        '''
        url = "https://raw.githubusercontent.com/DiseaseOntology/HumanDiseaseOntology/master/src/ontology/subsets/" + name + ".json"
        response = urllib.request.urlopen(url)
        data = json.loads(response.read().decode('utf-8'))
        nodes = data['graphs'][0]['nodes']
        for node in nodes:
            if 'synonyms' in node['meta'].keys():
                for synonym in node['meta']['synonyms']:
                    self.diseases.setdefault(name, [])
                    if synonym['val'] not in self.diseases and (node['lbl'] + synonym['val']) not in self.disease_synonym:
                        self.diseases[name].append({'name': node['lbl'], 'synonym': synonym['val']})
                        self.disease_synonym.append(node['lbl'] + synonym['val'])

    def start(self):
        '''
            params:
                    -
            description:
                    - This is a method used as a gateway through parsing the data
        '''
        for file_name in self.file_names:
            self.parse(file_name)


# do_parser = DOParser()
# do_parser.start()
# print(do_parser.diseases["DO_AGR_slim"])
# for key in do_parser.diseases:
#     print(key, len(do_parser.diseases[key]))
