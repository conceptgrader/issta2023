import glob
import config
import pandas as pd


def get_test_path_single_assignment_question(question_id):

    test_path = glob.glob(config.test_path.format(question_id))
    return test_path[0]


def get_data_path_single_question(question_id):
    wrong_data_paths = glob.glob(config.wrong_path.format(question_id))
    reference_data_path = glob.glob(config.ref_path.format(question_id))
    return wrong_data_paths, reference_data_path

def read_code(path):
    code = ''
    with open(path, 'r') as f:
        for line in f.readlines():
            if 'print' not in line:
                code += line
            elif '#print' in line:
                pass
            else:
                code += line.split('print')[0]+'pass\n'
        f.close()
    return code
