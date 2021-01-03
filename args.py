import argparse

def get_parser():

    parser = argparse.ArgumentParser(description='Multimodal news retrieval')
    parser.add_argument('--feature', default='bert', type=str, help='bert or location or object_scene or entity_overlap')
    parser.add_argument('--model_path_scene', default="scene_classification/resnet50_places365.pth.tar", type=str)
    parser.add_argument('--scene_labels', default="scene_classification/categories_places365.txt", type=str)
    parser.add_argument('--scene_hierarchy', default="scene_classification/scene_hierarchy_places365.csv", type=str)
    parser.add_argument('--model_path_object', default="object_classification/resnet50_weights_tf_dim_ordering_tf_kernels.h5", type=str)
    parser.add_argument('--model_path_location', default="location_verification/base_M/", type=str)
    parser.add_argument('--language', default='eng', type=str)
    parser.add_argument('--docs_path', default='data', type=str)
    parser.add_argument('--images_path', default='data/collected_images', type=str)

    return parser

