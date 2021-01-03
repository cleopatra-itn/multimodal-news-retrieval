import json



def open_json(path):
    with open(path, encoding='utf-8') as json_file:
        content = json.load(json_file)
        return content


def save_file(fileName, file):
    with open(fileName, 'w') as outfile:
        json.dump(file, outfile)


ls_features = ['sim_bert', 'sim_entity', 'sim_obj', 'sim_loc', 'sim_scene', 'sim_avg_text', 'sim_avg_visual',
               'sim_avg_total'

               ]
langs = ['deu', 'eng']

for lang in langs:
    file_name = lang + "_total_similarity_file" + ".json"
    sims_file = open_json(file_name)  # compute similarities
    keys = sims_file.keys()

    dic_events = {}

    for feature in ls_features:

        for event in keys:

            sorted_event_articles = {}
            query = sims_file[event]
            query_id = query['article_id']
            sims = query['similarity']
            sorted_event_articles0 = {}

            for s in sims:

                id = s['article_id']
                if id != query_id:
                    sim_extracted = s[feature]
                    sorted_event_articles0[s['event'] + '/' + str(id)] = sim_extracted

            sorted_event_articles = {}

            for k, v in sorted(sorted_event_articles0.items(), key=lambda item: item[1], reverse=True):
                sorted_event_articles[k] = v
                # print("%s: %s" % (k, v))
            dic_events[event] = sorted_event_articles

        save_file('sorted_results/' + lang + '/' + lang + '_' + feature + '.json', dic_events)


