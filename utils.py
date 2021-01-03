import json
import os
import numpy as np

def save_file(fileName, file):
    with open(fileName, 'w') as outfile:
        json.dump(file, outfile)

def open_json(fileName):
    try:
        with open(fileName,encoding='utf8') as json_data:
            d = json.load(json_data)
    except Exception as s:
        d=s
        print(d)
    return d

def get_file_name(path):
    head, file_name = os.path.split(path)
    return file_name

def create_latex_table(evaluation_data, domain_events, similarity_labels):

    output = ""
    for domain in domain_events.keys():

        for event in domain_events[domain]:
            data_row = evaluation_data.loc[event, similarity_labels]

            values = [int(i * 100) for i in data_row]
            max_value = np.max(values)


            if domain_events[domain].index(event) == 0:
                table_row = "\hline \multirow{"+str(len(domain_events[domain]))+"}{*}{\\rotatebox{90}{\\textbf{"+domain+"}}} & "+event.replace("_", " ")
            else:
                table_row = "\cline{2-10}  & "+event.replace("_", " ")

            for value in values:

                if value == max_value:
                    table_row += "& \\textbf{" +str(value)+"}"
                else:
                    table_row += "& " + str(value)
            table_row += " \\\\"

            output += table_row + "\n"
        output += "\hline\n"

    return output

