# from utils import *
#
#
# def convert_zero_to_random(similarity_measures, lang):
#     ls_features = ['sim_bert', 'sim_entity', 'sim_obj', 'sim_loc', 'sim_scene', 'sim_avg_text', 'sim_avg_visual',
#                    'sim_avg_total']
#
#     similarity_measures = open_json('similarity/' + lang + '.json')
#
#     for (key, sims) in similarity_measures.items():
#
#         for feat in ls_features:
#
#             min_val_sim = 1
#
#             for s in sims['similarity']:
#                 if s[feat] < min_val_sim:
#                     if s[feat] != 0:
#                         min_val_sim = s[feat]
#             if min_val_sim == 1:  # this means min_val=0
#                 min_val_sim = 0.01
#
#             for s in sims['similarity']:
#                 sfeat = s[feat]
#                 if sfeat == 0.0:
#                     s[feat] = np.random.random_sample() * min_val_sim
#
#     save_file('similarity/' + lang + '_random_retrieval_replaced_by_zero.json', similarity_measures)
#
