from args import get_parser
args = get_parser().parse_args()

from nltk import tokenize
from flair.embeddings import FlairEmbeddings, BertEmbeddings

from flair.data import Sentence
import glob
from utils import *
import os

from object_classification.object_embedding import ObjectDetector


class features:
    def __init__(self):

        model_path_sc = args.model_path_scene
        model_path_obj = args.model_path_object
        model_path_loc = args.model_path_location
        hierarchy = args.scene_hierarchy
        labels = args.scene_labels

        self.scence_obj = SceneClassificator(model_path=model_path_sc, labels_file=labels, hierarchy_file=hierarchy)
        self.object_obj = ObjectDetector(model_path=model_path_obj)
        self.location_obj = GeoEstimator(model_path_loc, use_cpu=True)

        self.flair_forward_embedding = FlairEmbeddings('multi-forward')
        self.flair_backward_embedding = FlairEmbeddings('multi-backward')
        self.bert_embedding = BertEmbeddings('bert-base-multilingual-cased')

    def save_void_file_features(self, articles, lang):
        whole_features = []
        events = articles.keys()
        for event in events:
            docs = articles[event]
            for doc in docs:
                id = doc['uri']
                whole_features.append( {'article_id':id, 'features':{} } )
        np.save('features/features_'+lang, whole_features)

    def entities_main_vector(self, articles_in, nel_tool, ner_tool):
        vector = []
        events = articles_in.keys()
        for event in events:
            articles = articles_in[event]
            for art in articles:
                for ent in art['nel'][nel_tool][ner_tool]:
                    if ent not in vector:
                        vector.append(ent)
        return vector

    def save_bert(self, name_whole_features, articles):

            if os.path.exists("features/" + name_whole_features):
                whole_features = np.load("features/" + name_whole_features, allow_pickle=True)
            else:
                'features file not found!!!'
            events = articles.keys()
            indplus = 0

            for event in events:
                docs = articles[event]

                for doc in docs:
                    text = doc['body'][:1500]
                    id = doc['uri']
                    ##
                    for f in whole_features:
                        if f['article_id'] == id:
                            features_doc = f
                    sentences = tokenize.sent_tokenize(text)
                    len_sent = len(sentences)
                    vec_embs = []

                    for ind in range(len_sent):

                        if len(sentences[ind]) > 0:
                            if ind == len_sent - 1:
                                text = sentences[ind]
                            elif ind == len_sent - 2:
                                text = sentences[ind] + " " + sentences[ind + 1]
                            else:
                                text = sentences[ind] + " " + sentences[ind + 1] + " " + sentences[ind + 2]

                            sentence = Sentence(text)

                            self.bert_embedding.embed(sentence)
                            avg_vector = np.zeros(768)
                            counter = 0
                            # now check out the embedded tokens.
                            for token in sentence:
                                vector = token.embedding.cpu().detach().numpy()
                                avg_vector = np.add(avg_vector, vector[0:768])
                                counter += 1
                            avg_vector /= counter
                            vec_embs.append(avg_vector)
                    features_doc['features']['t_bert'] = vec_embs
                    indplus += 1
                    print("******************   " + str(indplus))
            np.save("features/" +name_whole_features, whole_features)
            print("bert saved successfully!!!")

    def save_object_scene(self,name_whole_features, articles, image_path, image_names, img_format):
            if img_format == "png":
                till = -4
            else:
                till = -5
            if os.path.exists("features/" + name_whole_features):
                whole_features = np.load("features/" + name_whole_features, allow_pickle=True)
            else:
                'file features not found!!!'
            events = articles.keys()

            for event in events:
                print("event " + event + " ...")
                docs = articles[event]
                for i in range(len(docs)):  # img_name == article_id
                    img = ""
                    for image_n, image in zip(image_names, glob.glob(image_path + "*." + img_format)):
                        # o = image_n[0:till]
                        if image_n[0:till] == docs[i]['uri']:
                            img = image
                            break
                    if img != "":
                        print("article " + str(i) + " ...")
                        v_scene = self.scence_obj.get_img_embedding(img)
                        v_object = self.object_obj.get_img_embedding(img)
                        article_id = docs[i]['uri']
                        for f in whole_features:
                            if f['article_id'] == article_id:
                                features_id = f
                        features_id['features']['v_scene'] = v_scene
                        features_id['features']['v_object'] = v_object
            np.save("features/" + name_whole_features, whole_features)
            print("features obj and scene saved successfully!!!")

    def save_entity_overlap(self, name_whole_features, articles, lang):

                all_entities_vector = self.entities_main_vector(articles, 'wikifier', 'spacy')
                np.save("features/entities_vector_" + lang, all_entities_vector)

                if os.path.exists("features/" + name_whole_features):
                    whole_features = np.load("features/" + name_whole_features, allow_pickle=True)
                else:
                    'file features not found!!!'
                events = articles.keys()

                indplus = 0
                for event in events:
                    docs = articles[event]

                    for doc in docs:
                        id = doc['uri']
                        features = ""
                        for f in whole_features:
                            if f['article_id'] == id:
                                features = f

                        t_entity_overlap = self.map_entities_to_main_vector(doc['nel']['wikifier']['spacy'], all_entities_vector)
                        if features != "":
                            features['features']['t_entity_overlap'] = t_entity_overlap
                        indplus += 1
                        print("******************   " + str(indplus))

                np.save("features/" + name_whole_features, whole_features)
                print("features entity overlap saved successfully!!!")

    def save_location(self, pre_path, name_whole_features, image_format):

           if os.path.exists("features/" + name_whole_features):
               whole_features = np.load("features/" + name_whole_features, allow_pickle=True)
           else:
               'file features not found!!!'

           for wf in whole_features:
               img = pre_path+wf['article_id']+"."+image_format

               location_features = self.location_obj.get_img_embedding(img)

               if len(location_features)<1:
                   print(wf['article_id'])
               # wf['features']['v_location'] = location_features
               # print(location_features.shape)
               wf['features']['v_location'] = location_features
               # whole_features.append({'article_id': id, 'features': {'v_location': location_features}})

           np.save("features/"+name_whole_features, whole_features)

           print("features location saved successfully!!!")

def get_features(features_obj, feature, articles_with_nel, images_path, image_names, lang, image_format):
            features_obj.save_void_file_features(articles_with_nel, lang)
            name_saved_features = 'features_'+lang
            if not os.path.exists("features"):
                os.mkdir("features")
            if feature == 'scene_obj':
                if not os.path.exists("features/"+lang+"features_obj_scene.npy"):
                    features_obj.save_object_scene(articles_with_nel, name_saved_features, images_path, image_names, image_format)
            if feature=='bert':
                if not os.path.exists("features/"+lang+"features_bert.npy"):
                    features_obj.save_bert( name_saved_features, articles_with_nel)
            if feature=='entity':
                if not os.path.exists("features/"+lang+"features_entity_overlap.npy"):
                     features_obj.save_entity_overlap(name_saved_features, articles_with_nel, lang)
            if feature=='location':
                if not os.path.exists("features/"+lang+"features_location.npy"):
                    features_obj.save_location(images_path, name_saved_features, image_format)

lang = args.language
articles_with_nel = open_json(args.docs_path + "/nel_" + lang + ".json")
images_path = args.images_path
image_names = [ get_file_name(img) for img in glob.glob(images_path + "/*.jpeg") ]
feature_type = args.feature
features_obj = features()
get_features(features_obj, feature_type, articles_with_nel, images_path, image_names, lang, "jpeg")
