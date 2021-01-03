import numpy as np
from scipy import spatial
from utils import open_json, save_file
import os
from args import get_parser
args = get_parser().parse_args()


def save_similarity_per_feature(whole_featurs, lang):
    n = len(whole_featurs)
    s_scene = np.zeros([n, n])
    s_object = np.zeros([n, n])
    s_location = np.zeros([n, n])
    s_bert = np.zeros([n, n])
    s_entity_overlap = np.zeros([n, n])

    for i in range(len(whole_featurs)):
        for j in range(len(whole_featurs)):
            print(i, j)
            article_features = whole_featurs[i]['features']
            query_article_features = whole_featurs[j]['features']
            try:
                s_scene[i, j] = 1 - spatial.distance.cosine(query_article_features['v_scene'], article_features['v_scene'])
                s_object[i, j] = 1 - spatial.distance.cosine(query_article_features['v_object'], article_features['v_object'])
                if whole_featurs[i]['article_id'] not in ['486810751', '1436821318', '435220228']:
                    if whole_featurs[j]['article_id'] not in ['486810751', '1436821318', '435220228']:
                         s_location[i, j] = 1 - spatial.distance.cosine(query_article_features['v_location'], article_features['v_location'])

                a = np.array([np.asarray(aa) for aa in query_article_features['t_bert']])
                b = np.array([np.asarray(bb) for bb in article_features['t_bert']])
                s = spatial.distance.cdist(a, b, 'cosine')
                s_bert[i, j] = 1 - np.mean(s)

                s_entity_overlap[i, j] = 1 - spatial.distance.cosine(article_features['t_entity_overlap'], query_article_features['t_entity_overlap'])
                if np.isnan(s_entity_overlap[i, j]):
                        s_entity_overlap[i, j] = 0

            except Exception as e:
                s_bert[i, j] = 0

    if not os.path.exists('similarity'):
        os.mkdir('similarity')

    np.save("similarity/" + lang + "/s_scene.npy", s_scene)
    np.save("similarity/" + lang + "/s_object.npy", s_object)
    np.save("similarity/" + lang + "/s_locaction.npy", s_location)
    np.save("similarity/" + lang + "/s_bert.npy", s_bert)
    np.save("similarity/" + lang + "/s_entity_overlap.npy", s_entity_overlap)


def save_similarities_in_json(whole_featurs, lang):
    s_scene = np.load("similarity/" + lang + "/s_scene.npy")
    s_object = np.load("similarity/" + lang + "/s_object.npy")
    s_location = np.load("similarity/" + lang + "/s_locaction.npy")
    s_bert = np.load("similarity/" + lang + "/s_bert.npy")
    s_entity_overlap = np.load("similarity/" + lang + "/s_entity_overlap.npy")

    dic_query_comparisons_with_the_rest = {}

    print("saving similarities to json... ")
    for i in range(len(whole_featurs)):
        query = whole_featurs[i]
        avg_total = []

        for j in range(0, len(whole_featurs), 1):
            print(i, j)
            art = whole_featurs[j]
            sim_obj = s_object[i, j]
            sim_scene = s_scene[i, j]
            sim_loc = s_location[i, j]
            sim_bert = s_bert[i, j]
            sim_entity = s_entity_overlap[i, j]
            sim_avg_text = (sim_bert + sim_entity) / 2
            sim_avg_visual = (sim_obj + sim_loc + sim_scene) / 3
            sim_avg_total = (sim_bert + sim_entity + sim_obj + sim_loc + sim_scene) / 5

            if np.isnan(sim_entity):
                print("error")
            avg_total.append({'article_id': art['article_id'], 'event': art['event'],
                              'sim_bert': sim_bert, 'sim_entity': sim_entity, 'sim_obj': sim_obj, 'sim_loc': sim_loc,
                              'sim_scene': sim_scene,
                              'sim_avg_text': sim_avg_text, 'sim_avg_visual': sim_avg_visual,
                              'sim_avg_total': sim_avg_total
                              })

        dic_query_comparisons_with_the_rest[query['event'] + "/" + query['article_id']] = {
            'article_id': query['article_id'], 'similarity': avg_total}
    file_name = lang + "_similarity"+".json"
    save_file("similarity/" + file_name, dic_query_comparisons_with_the_rest)
    return file_name


def convert_zero_similarities_to_random_retrieval(lang):
    ls_features = ['sim_bert', 'sim_entity', 'sim_obj', 'sim_loc', 'sim_scene', 'sim_avg_text', 'sim_avg_visual',
                   'sim_avg_total']

    similarity_measures = open_json('similarity/' + lang + '.json')

    for (key, sims) in similarity_measures.items():

        for feat in ls_features:

            min_val_sim = 1

            for s in sims['similarity']:
                if s[feat] < min_val_sim:
                    if s[feat] != 0:
                        min_val_sim = s[feat]
            if min_val_sim == 1:  # this means min_val=0
                min_val_sim = 0.01

            for s in sims['similarity']:
                sfeat = s[feat]
                if sfeat == 0.0:
                    s[feat] = np.random.random_sample() * min_val_sim
    save_file('similarity/' + lang + '_random_retrieval_replaced_by_zero.json', similarity_measures)


lang = args.language
whole_featurs = np.load("features/" + "features_"+lang+".npy", allow_pickle=True)  # saved features from get_features.py
save_similarity_per_feature(whole_featurs, lang)
save_similarities_in_json(whole_featurs, lang)
convert_zero_similarities_to_random_retrieval(lang)

