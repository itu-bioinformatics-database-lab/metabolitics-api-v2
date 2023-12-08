import requests
import xlsxwriter

study = 'ST000329'
group = 'Sample type:Control'
disease = 'Sample type:FSGS'
summary = requests.get('https://www.metabolomicsworkbench.org/rest/study/study_id/' + study + '/summary').json()
study_title = summary['study_title']
workbook = xlsxwriter.Workbook(study + '_' + study_title.replace(' ', '_') + '.xlsx')
data_worksheet = workbook.add_worksheet('data')
data = requests.get('https://www.metabolomicsworkbench.org/rest/study/study_id/' + study + '/data').json()
sample_ids = data['1']['DATA'].keys()
sample_ids = list(sample_ids)
sample_ids.sort()
factors = requests.get('https://www.metabolomicsworkbench.org/rest/study/study_id/' + study + '/factors').json()
factors_dict = {}
for value in factors.values():
    local_sample_id = value['local_sample_id']
    if local_sample_id in sample_ids:
        factors_dict[local_sample_id] = value['factors'].split(' | ')[0]
sample_ids = [sample_id for sample_id in sample_ids if factors_dict[sample_id] == group or factors_dict[sample_id] == disease]
factors_dict = {key: value for key, value in factors_dict.items() if value == group or value == disease}
row = 0
col = 1
for sample_id in sample_ids:
    data_worksheet.write(row, col, sample_id)
    col += 1
row = 1
for value in data.values():
    metabolite_name = value['refmet_name'] if value['refmet_name'] != '' else value['metabolite_name']
    metabolite_datum = value['DATA']
    data_worksheet.write(row, 0, metabolite_name)
    col = 1
    for sample_id in sample_ids:
        metabolite_data = metabolite_datum[sample_id]
        metabolite_data = float(metabolite_data) if metabolite_data != None else metabolite_data
        data_worksheet.write(row, col, metabolite_data)
        col += 1
    row += 1
meta_worksheet = workbook.add_worksheet('meta')
meta_worksheet.write(0, 0, 'study name')
meta_worksheet.write(0, 1, study_title)
meta_worksheet.write(1, 0, 'control/healthy/wildtype group')
meta_worksheet.write(1, 1, group)
meta_worksheet.write(2, 0, 'subject id')
meta_worksheet.write(2, 1, 'group')
row = 3
for sample_id in sample_ids:
    meta_worksheet.write(row, 0, sample_id)
    meta_worksheet.write(row, 1, factors_dict[sample_id])
    row += 1
workbook.close()
print(len(set(factors_dict.values())))
print(len(sample_ids))
