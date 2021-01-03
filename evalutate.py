from sklearn.metrics import average_precision_score
import pandas as pd
import operator
from utils import *
from args import get_parser
args = get_parser().parse_args()

def save_string_to_file(text, file_path):
    file = open(file_path, 'w')
    file.write(text)
    file.close()

def evaluate_ranked_results(expected, ranked_results):
    c = 0
    for i in range(0, len(ranked_results)):
        (article_id, similarity) = ranked_results[i]
        if article_id in expected:
            c+=1

    precision = c/float(len(ranked_results))
    return precision

def compute_precision(query, event, similarity_measure):

    expected = list()
    for q in query['similarity']:
        if q['event'] == event:
            expected.append(q['article_id'])

    retrieved = {}
    for q in query['similarity']:
        retrieved[q['article_id']] = q[similarity_measure]

    ranked_results = sorted(retrieved.items(), reverse=True, key=operator.itemgetter(1))

    precision_values = []
    # take each k and compute precision
    for i in range(1, len(expected)+1):
        results_at_k = ranked_results[0:i]
        precision_at_k = evaluate_ranked_results(expected, results_at_k)
        precision_values.append(precision_at_k)

    avg_prec = np.mean(precision_values)
    return avg_prec

def compute_recall(query, event, similarity_measure):

    expected = list()
    for q in query['similarity']:
        if q['event'] == event:
            expected.append(q['article_id'])

    retrieved = {}
    for q in query['similarity']:

        sim_measure = q[similarity_measure]
        if np.isnan(sim_measure):
            sim_measure = 0

        retrieved[q['article_id']] = sim_measure


    # sort the documents by similarity score
    ranked_results = sorted(retrieved.items(), reverse=True, key=operator.itemgetter(1))

    c = 0
    top_k_ranked_results = ranked_results[0:len(expected)]
    for i in range(0, len(top_k_ranked_results)):
        (article_id, similarity) = top_k_ranked_results[i]
        if article_id in expected:
            c +=1

    recall = c/float(len(expected))
    return recall

def convert_to_class(event, vector, selected_feature):
    sims = vector['similarity']
    scores = np.zeros(len(sims))
    ret = np.zeros(len(sims))

    for s in range(len(sims)):
        sim = sims[s]
        if sim['event'] == event:
            ret[s] = 1
        if np.isnan(sim[selected_feature]):
            scores[s] = 0
        else:
           scores[s] = sim[selected_feature]

    return ret, scores

def get_valid_similarity_score( event, similarity_measures):
    similarity_scores = []
    for k in similarity_measures.keys():

        event_id = k.split("/")[0]

        if event == event_id:
            similarity_scores.append(similarity_measures[k])
    return similarity_scores

lang = args.language
data_path = args.docs_path
similarity_measures = open_json('similarity/'+ lang + '_similarity.json')
articles = open_json(data_path+'/nel_' + lang + '.json')
events = articles.keys()
avg_event = {}

domain_events = {}
domain_events["Politics"] = ["2016_United_States_presidential_election", "Impeachment_of_Donald_Trump", "European_Union–Turkey_relations", "War_in_Donbass", "Brexit", "Cyprus–Turkey_maritime_zones_dispute"]
domain_events["Environment"] = ["Global_warming", "Water_scarcity", "2019–20_Australian_bushfire_season", "Indonesian_tsunami", "Water_scarcity_in_Africa", "2018_California_wildfires", "Palm_oil_production_in_Indonesia"]
domain_events["Finance"] = ["Financial_crisis_of_2007–08", "Greek_government-debt_crisis", "Volkswagen_emissions_scandal"]
domain_events["Health"] = ["Coronavirus", "Ebola_virus_disease", "Zika_fever", "Avian_influenza", "Swine_influenza"]
domain_events["Sport"] = ["2016_Summer_Olympics", "2018_FIFA_World_Cup", "2020_Summer_Olympics", "2022_FIFA_World_Cup"]

measurement_labels = ['sim_bert', 'sim_entity', 'sim_avg_text', 'sim_obj', 'sim_scene', 'sim_loc', 'sim_avg_visual', 'sim_avg_total']


evaluation_result = []


for event in events:
            query_scores = get_valid_similarity_score( event, similarity_measures)
            event_score = [event]
            for sim_label in measurement_labels:
                scores = []

                processed_article_ids = set()
                for query in query_scores:

                    if query['article_id'] in processed_article_ids:
                        continue
                    # Compute Recall
                    # score = compute_recall(query, event, sim_label)
                    y_true, y_pred = convert_to_class(event, query, sim_label)
                    score = average_precision_score(y_true, y_pred)
                    scores.append(score)

                    processed_article_ids.add(query['article_id'])

                event_avg_score = np.mean(scores)
                event_score.append(event_avg_score)
            evaluation_result.append(event_score)

columns = ['event']
columns.extend(measurement_labels)

event_evaluation_data = pd.DataFrame(evaluation_result, columns=columns)

event_evaluation_data.set_index('event', inplace=True)

latex_output = create_latex_table(event_evaluation_data, domain_events, measurement_labels)
if not os.path.exists('evaluation_files'):
    os.mkdir('evaluation_files')
save_string_to_file(latex_output, 'evaluation_files/' +lang+'_latex_table.txt')
event_evaluation_data.to_csv('evaluation_files/' +lang+'_evaluation.csv', sep=',', encoding='utf-8')

print(lang+' evaluation done!')

