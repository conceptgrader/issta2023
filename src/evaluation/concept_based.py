import glob
import json
import config
import networkx as nx
import time

from joblib import Parallel, delayed

from src import helper as H
from src import data_extractor, evaluation
from src.concept_graph import concept_graph
from src.evaluation import utils

class Result():

    def __init__(self, name, grade, ref_name):
        self.name = name
        self.grade = grade
        self.ref_name = ref_name

    def __str__(self):
        return 'student: {} get {} by {}'.format(self.name, self.grade, self.ref_name)

def grading_by_common_nodes(G_refs, G_stus, max_mark, stu_name, question_id=None):

    start = time.time()
    func_list = list(G_refs.keys())
    result = {'name': stu_name, 'score':0}
    for func in func_list:
        result[func] = 0
    try:
        var_mapping = H.get_variable_mapping(G_refs[func_list[0]].name, G_stus[func_list[0]].name, question_id)
    except:
        print(stu_name + 'variable mapping error')
        return {'name': stu_name, 'score': None}

    for func_name in func_list:
        G_ref = G_refs[func_name]
        G_stu = G_stus[func_name]
        x = list(G_ref._node.keys())
        y = list(G_stu._node.keys())
        matched_nodes, unmatch_nodes = [],[]

        matched_edges = []
        for n in G_stu.nodes():
            new_n = H.replace_var(n, var_mapping[func_name])
            if G_ref.has_node(new_n):
                matched_nodes.append(new_n)
        print(matched_nodes)
        for e in G_stu.edges():
            if G_ref.has_edge(H.replace_var(e[0], var_mapping[func_name]), H.replace_var(e[1], var_mapping[func_name])):
                matched_edges.append(e)
        print(matched_nodes)
        node_score = len(matched_nodes) / len(G_ref.nodes())

        edge_score = len(matched_edges) / len(G_ref.edges())
        result[func_name] = 0.5 * (node_score + edge_score) * max_mark
    stop = time.time()
    t = (stop - start)
    result['time-taken'] = t
    return result


def concept_based_result(stu_graphs, ref_graphs, total_mark, question_id, save_path):
    num_of_jobs=8
    grade_dicts = {}
    res=Parallel(n_jobs=num_of_jobs)(delayed(grading_by_common_nodes)(
            ref_graph, stu_graph, total_mark, stu_name, question_id=question_id)
            for stu_name, stu_graph in stu_graphs.items() for ref_name, ref_graph in ref_graphs.items())
    for result in res:
        name = result['name']
        score = result['score']
        if score is None:
            continue
        if name not in grade_dicts:
            grade_dicts[name] = result
        else:
            print(result['time-taken'])
            grade_dicts[name]['time-taken'] += result['time-taken']
            if score > grade_dicts[name]['score']:
                grade_dicts[name]['score'] = score

    with open(save_path, 'w+') as output:
        for k, v in grade_dicts.items():
            output.write(json.dumps(v) + '\n')
        output.close()

def run_concept_based(tasks):
    for question_id, func_name in tasks.items():
        print('Question ', question_id)
        wrong_data_path = glob.glob(config.wrong_path.format(question_id))
        wrong_data_path = sorted(wrong_data_path, key=str.lower)
        reference_data_path = glob.glob(config.ref_path.format(question_id))
        reference_data_path = sorted(reference_data_path, key=str.lower)
        _, ref_concept_graphs, _, _ = concept_graph.create_ref_concept_graph_path(reference_data_path)
        fail_names, stu_concept_graphs, syntax_errors, exceptions = concept_graph.create_stu_concept_graph_path(
            wrong_data_path,
        )
        print('{} submissions in total'.format(len(wrong_data_path)))
        print('{} submissions have syntax error'.format(syntax_errors))
        print('{} submissions throws exception when creating concept graph'.format(exceptions))
        if question_id != 2:
            concept_based_save_path = config.CONCEPT_BASED_PATH.format(question_id)
        else:
            concept_based_save_path = config.CONCEPT_BASED_PATH.format(str(question_id)+'_')
        concept_based_result(stu_concept_graphs, ref_concept_graphs, 100, question_id, save_path = concept_based_save_path)

# run_concept_based()
