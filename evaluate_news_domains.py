import json
import pandas as pd
import numpy as np
import os

def open_json(path):
    with open(path) as json_file:
        content = json.load(json_file)
        return content

def save_string_to_file(text, file_path):
    file = open(file_path, 'w')
    file.write(text)
    file.close()

def create_latex_table(data_frame):

    output =  ""
    for index, row in data_frame.iterrows():

        values = data_frame.loc[index][0:6]
        values = [int(i) for i in values]
        max_eng_value = np.max(values[0:3])
        max_deu_value = np.max(values[3:])

        output += index
        for v in values:
            if values.index(v) < 3 and v == max_eng_value:
                output += "&  \\textbf{"+str(v)+"}"
            elif values.index(v) >= 3 and v == max_deu_value:
                output += "&  \\textbf{"+str(v)+"}"
            else:
                output += "&  " + str(v)
        output += "\\\\ \hline\n"

    return output


    # for domain in domains:
    #     data_row = data_frame.loc[event, similarity_labels]


domain_events = {}
domain_events["Politics"] = ["2016_United_States_presidential_election", "Impeachment_of_Donald_Trump", "European_Union–Turkey_relations", "War_in_Donbass", "Brexit", "Cyprus–Turkey_maritime_zones_dispute"]
domain_events["Environment"] = ["Global_warming", "Water_scarcity", "2019–20_Australian_bushfire_season", "Indonesian_tsunami", "Water_scarcity_in_Africa", "2018_California_wildfires", "Palm_oil_production_in_Indonesia"]
domain_events["Finance"] = ["Financial_crisis_of_2007–08", "Greek_government-debt_crisis", "Volkswagen_emissions_scandal"]
domain_events["Health"] = ["Coronavirus", "Ebola_virus_disease", "Zika_fever", "Avian_influenza", "Swine_influenza"]
domain_events["Sport"] = ["2016_Summer_Olympics", "2018_FIFA_World_Cup", "2020_Summer_Olympics", "2022_FIFA_World_Cup"]

measurement_labels = ['sim_avg_text', 'sim_avg_visual', 'sim_avg_total']

deu_data_frame = pd.read_csv('evaluation_files/deu_v2_evaluation.csv', index_col='event')
eng_data_frame = pd.read_csv('evaluation_files/eng_v2_evaluation.csv', index_col='event')


all_values = []
for domain in domain_events:

    domain_row = np.zeros(6)
    for event in domain_events[domain]:
        eng_values = eng_data_frame.loc[event, measurement_labels]
        eng_values = [int(i * 100) for i in eng_values]
        deu_values = deu_data_frame.loc[event, measurement_labels]
        deu_values = [int(i * 100) for i in deu_values]

        event_row = np.zeros(6)
        event_row[0:3] = eng_values
        event_row[3:] = deu_values
        domain_row = np.add(domain_row, event_row)

    domain_row /= len(domain_events[domain])

    domain_values = [domain]
    domain_values.extend(domain_row)

    all_values.append(domain_values)

columns = ['domain', 'eng_sim_avg_text', 'eng_sim_avg_visual', 'eng_sim_avg_total', 'deu_sim_avg_text', 'deu_sim_avg_visual', 'deu_sim_avg_total']
domain_evaluation_data = pd.DataFrame(all_values, columns=columns)
domain_evaluation_data.set_index('domain', inplace=True)
if not os.path.exists('evaluation_files'):
    os.mkdir('evaluation_files')
domain_evaluation_data.to_csv('evaluation_files/domain_evaluation.csv', sep=',', encoding='utf-8')

latex_output = create_latex_table(domain_evaluation_data)

save_string_to_file(latex_output, "evaluation_files/domain_latex_table.txt")
