import gzip
import json
import numpy as np
import config

from typing import Iterable, Dict
import pandas as pd


def read_results(evalset_file: str, fail_rate=0, range=(0,0), enableRange=False) -> Dict[str, Dict]:
    if not enableRange:
        if fail_rate == 0:
            return {task["name"]: task['score'] for task in stream_jsonl(evalset_file)}
        else:
            return {task["name"]: task['score'] for task in stream_jsonl(evalset_file) if task['fail_rate'] >= fail_rate}
    else:
        return {task["name"]: task['score'] for task in stream_jsonl(evalset_file) if
                task['fail_rate'] > range[0] * 100 and task['fail_rate'] <= range[1] * 100 + 0.01}

def read_time_results(evalset_file: str, fail_rate=0, range=(0,0), enableRange=False) -> Dict[str, Dict]:
    if not enableRange:
        if fail_rate == 0:
            return {task["name"]: 60 * task['time-taken'] for task in stream_jsonl(evalset_file)}
        else:
            return {task["name"]: 60 * task['time-taken'] for task in stream_jsonl(evalset_file) if task['fail_rate'] >= fail_rate}
    else:
        return {task["name"]: 60 * task['time-taken'] for task in stream_jsonl(evalset_file) if
                task['fail_rate'] > range[0] * 100 and task['fail_rate'] <= range[1] * 100 + 0.01}

def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, 'rt') as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)

def compute_result(predict_score, tutor_score, metric=''):
    assert len(predict_score) == len(tutor_score)
    assert predict_score.keys() == tutor_score.keys()
    list1 = list(predict_score.values())
    list2 = list(tutor_score.values())
    if metric == 'similarity':
        return round(np.dot(list1, list2)/(np.linalg.norm(list1)*np.linalg.norm(list2)), 3)
    elif metric == 'rmse':
        return round(np.sqrt(np.mean(((np.array(list1) - np.array(list2)) ** 2))), 3)
    elif metric == 'mae':
        return round(np.mean(np.absolute((np.array(list1) - np.array(list2)))), 3)

def read_tutor_marks(question_id):
    tutor_grades = pd.read_csv(config.DATA_PATH + 'task{}_tutors.csv'.format(question_id))
    sorted_grades = tutor_grades.sort_values(by=['name'], ascending=True).to_dict('records')
    with open(config.DATA_PATH + 'task{}_tutors.json'.format(question_id), 'w+') as output:
        for result in sorted_grades:
            output.write(json.dumps(result) + '\n')
        output.close()
    return

def read_results_q2(evalset_file: str, isTutor=True):
    if isTutor:
        return {task["name"]: {"name": task["name"], "unique_day": task['score_1'], "unique_month": task['score_2'],
                               "contains_unique_day": task['score_3']} for task in stream_jsonl(evalset_file)}
    else:
        return {task["name"]: {"name": task["name"], "unique_day": task['unique_day'], "unique_month": task['unique_month'],
                               "contains_unique_day": task['contains_unique_day'],
                               "time-taken": task['time-taken']} for task in stream_jsonl(evalset_file)}

def read_tutor_q2_marks():
    tutor_grades = pd.read_csv(config.DATA_PATH + 'task2_tutors.csv')
    sorted_grades = tutor_grades.sort_values(by=['name'], ascending=True).to_dict('records')
    with open(config.DATA_PATH + 'task2_tutors_.json', 'w+') as output:
        for result in sorted_grades:
            output.write(json.dumps(result) + '\n')
        output.close()

    test_result = read_results_q2(config.TEST_BASED_PATH.format('2_'))
    tutor_result = read_results_q2(config.DATA_PATH + 'task2_tutors_.json')
    process_q2_raw(tutor_result, test_result, save_path=config.DATA_PATH + 'task2_tutors.json')



def process_q2_raw(pending_result_raw, test_result, save_path):
    pending_result = {}
    for name, res in pending_result_raw.items():
        score = 0
        if test_result[name]['unique_day'] != 100 and test_result[name]['unique_day'] != -1:
            score += res['unique_day']
        if test_result[name]['unique_month'] != 100 and test_result[name]['unique_month'] != -1:
            score += res['unique_month']
        if test_result[name]['contains_unique_day'] != 100 and test_result[name]['contains_unique_day'] != -1:
            score += res['contains_unique_day']
        pending_result[name] = {'name': name, 'score': score/3}
        if 'time-taken' in res:
            pending_result[name]['time-taken'] = res['time-taken']
    with open(save_path, 'w+') as output:
        for name, result in pending_result.items():
            output.write(json.dumps(result) + '\n')
        output.close()
    return pending_result

def sum_time(times: dict, common: set):
    times = {k: v for k, v in times.items() if k in common}
    avg_time = sum(times.values()) / len(common)
    return avg_time

def compare_with_ground_truth(tutor_score: dict, predict_score: dict, common: set):
    tutor_score = {k: v for k, v in tutor_score.items() if k in common}
    predict_score = {k: v for k, v in predict_score.items() if k in common}

    similarity = compute_result(predict_score, tutor_score, metric='similarity')
    rmse = compute_result(predict_score, tutor_score, metric='rmse')
    mae = compute_result(predict_score, tutor_score, metric='mae')
    return {'similarity': similarity, 'rmse': rmse, 'mae': mae}
# read_tutor_marks(1)
# process tutor q2
read_tutor_q2_marks()

# process concept q2
test_raw = read_results_q2(config.TEST_BASED_PATH.format('2_'))
concept_based_raw = read_results_q2(config.CONCEPT_BASED_PATH.format('2_'), isTutor=False)
concept_based_score = process_q2_raw(concept_based_raw, test_raw, save_path=config.CONCEPT_BASED_PATH.format(2))

# read_tutor_marks(3)
# read_tutor_marks(4)
# read_tutor_marks(5)
