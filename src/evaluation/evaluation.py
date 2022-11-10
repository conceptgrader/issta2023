import pandas as pd
from src.concept_graph.concept import Concept
from src import helper as H
import networkx as nx



def print_rq3(questions:list):

    from src.evaluation import utils
    import config
    for question_id in questions:
        tutor_score = utils.read_results(config.GROUND_TRUTH_PATH.format(question_id))
        test_based_score = utils.read_results(config.TEST_BASED_PATH.format(question_id))
        concept_based_score = utils.read_results(config.CONCEPT_BASED_PATH.format(question_id))
        # cfg_based_score = utils.read_results(config.CFG_BASED_PATH.format(question_id))
        common = set(tutor_score) & set(test_based_score) & set(concept_based_score)# & set(cfg_based_score)

        tutor_score = {k: v for k, v in tutor_score.items() if k in common}
        test_based_score = {k: v for k, v in test_based_score.items() if k in common}
        concept_based_score = {k: v for k, v in concept_based_score.items() if k in common}
        import json

        with open('./question_{}.json'.format(question_id), 'w+') as output:
            output.write(json.dumps(find_abnormal(test_based_score, concept_based_score, tutor_score), indent=2))
            output.close()


def find_abnormal(test_mark, predict_mark, tutor_mark):


    assert len(predict_mark) == len(tutor_mark) == len(test_mark)
    assert predict_mark.keys() == tutor_mark.keys() == test_mark.keys()
    abnormal = {}
    for key in predict_mark.keys():
        if (abs(predict_mark[key] - tutor_mark[key]) - abs(test_mark[key] - tutor_mark[key])) > 5:
            abnormal[key] = {'test grade': test_mark[key], 'concept grade': predict_mark[key], 'tutor grade': tutor_mark[key]}
    return abnormal

print_rq3([1,2,3,4,5])

# def find_normal(test_mark, predict_mark, tutor_mark, threshold):
#     assert len(predict_mark) == len(tutor_mark) == len(test_mark)
#     assert predict_mark.keys() == tutor_mark.keys() == test_mark.keys()
#     normal = {}
#     for key in predict_mark.keys():
#         if abs(predict_mark[key] - tutor_mark[key]) <= threshold:
#             normal[key] = {'test grade': test_mark[key], 'concept grade': predict_mark[key], 'tutor grade': tutor_mark[key]}
#     return normal
