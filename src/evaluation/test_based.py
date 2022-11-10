import json
import glob
import time
import pandas as pd
import os
from joblib import Parallel, delayed
import csv

import config
from src import helper as H
from src.evaluation import utils


def execute_student_solution_q2(prog_path, tests_path, debug=False):
    n_1, n_2, n_3 = 7, 5, 5
    start = time.time()
    t_pass, t_all, t_1, t_2, t_3, e1,e2,e3 = H.exec_program_q2(prog_path, tests_path, debug=debug)
    score = round(t_pass / t_all * 100, 2)
    s1 = round(t_1/n_1*100, 2)
    s2 = round(t_2/n_2*100, 2)
    s3 = round(t_3/n_3*100, 2)
    if e1 == 1:
        s1 = -1
    if e2 == 1:
        s2 = -1
    if e3 == 1:
        s3 = -1
    stop = time.time()
    t = (stop - start)
    score = 0
    fr = 0
    ts = [t_1, t_2, t_3]
    ns = [n_1, n_2, n_3]
    tss = 0
    nss = 0
    for i, s in enumerate([s1, s2, s3]):
        if s==100 or s==-1:
            continue
        score += s
        tss += ts[i]
        nss += ns[i]
    score /= 3
    fr = tss/nss*100
    return {'name': prog_path.split('/')[-1],
            'pass': t_pass, 'total': t_all, 'score': score,
            'score_1': s1, 'score_2': s2, 'score_3': s3,
            'fail_rate':int((100 - fr)), 'time-taken': t}

def execute_student_solution(prog_path, tests_path, debug=False):

    start = time.time()
    t_pass, t_all = H.exec_program(prog_path, tests_path, debug=debug)
    score = round(t_pass / t_all * 100, 2)
    stop = time.time()
    t = (stop - start)
    return {'name': prog_path.split('/')[-1], 'pass': t_pass, 'total': t_all, 'score': score, 'fail_rate':int((100 - score)), 'time-taken': t}


def execute_student_solutions(question_id, progs_path, save_path):

    test_path = config.test_path.format(question_id)
    res = Parallel(n_jobs=40)(delayed(execute_student_solution_q2)(prog_path, test_path, True)
                              for prog_path in progs_path)
    with open(save_path, 'w+') as output:
        for result in res:
            output.write(json.dumps(result) + '\n')
        output.close()
    with open('tutor2_remove.csv', 'w') as f:  # You will need 'wb' mode in Python 2.x
        w = csv.writer(f)
        for result in res:
            w.writerow(result.values())

def run_test_based(tasks):
    for question_id, func_name in tasks.items():
        print('Question ', question_id)
        wrong_score_path = glob.glob(config.wrong_path.format(question_id))
        wrong_score_path = sorted(wrong_score_path, key=str.lower)
        test_based_save_path = config.RESULTS_PATH + 'test_based_result_question_{}_.json'.format(question_id)
        execute_student_solutions(question_id, wrong_score_path, save_path=test_based_save_path)







