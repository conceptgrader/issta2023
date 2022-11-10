import glob
import tempfile
import subprocess
import os
import sys
import traceback
from multiprocessing import Process, Queue
from src import data_extractor

from src.basic_framework.repair import BlockRepair
from src.basic_framework.cfs import get_cfs_map
from src.basic_framework.statement import *
from src.concept_graph.concept import Concept
import ast

def replace_var(concepts, var_mapping=None):
    new_concepts=[]
    for concept in concepts:
        new_concept = Concept(category=concept.category)
        members = concept.members
        for mem in members:
            new_mem = var_mapping[mem] if mem in var_mapping else mem
            new_concept.add_members(new_mem)
        new_concepts.append(new_concept)
    return tuple(new_concepts)

def get_variable_mapping(ref_path, stu_path, question_id):
    test_path = data_extractor.get_test_path_single_assignment_question(question_id)
    var_mapping = {}
    ref_code = data_extractor.read_code(ref_path)
    submitted_code = data_extractor.read_code(stu_path)
    corr_cfs_map = get_cfs_map(ref_code)
    for func_name in corr_cfs_map.keys():
        r = BlockRepair(ques_dir_path=test_path)
        var_mapping[func_name] = r.get_vn_map(submitted_code, ref_code, func_name)
    return var_mapping

def subprocess_run(cmd_list, prog_input=None, blame_str='subprocess', timeout=10, debug=False, raiseExp=True):
    # Run cmd_list
    proc = subprocess.Popen(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
    try:
        if prog_input is None:
            outs, errs = proc.communicate(timeout=timeout)
        else:
            outs, errs = proc.communicate(input=str(prog_input).encode(), timeout=timeout)

    except subprocess.TimeoutExpired:
        # Timeout?
        proc.kill()
        # if raiseExp:
        #     raise Exception('{}: Timeout'.format(blame_str))
        return False, ''

    # Failure?
    if proc.returncode != 0:
        if not debug:  # If not running in debug
            errs = 'Failure'  # fail with a simple "failure" msg

        if raiseExp:
            return proc.returncode != 0, '{}: {}'.format(blame_str, errs)
            # raise Exception('{}: {}'.format(blame_str, errs))
    return proc.returncode == 0, outs.decode(encoding='ISO-8859-1')


def exec_program(prog_path, tests_path, debug=False):
    fnc = data_extractor.read_code(prog_path)
    # fnc = open(prog_path, 'r').read()
    test_paths = sorted(glob.glob(tests_path + 'input_*.txt'))
    valid = 0

    print(prog_path.split('/')[-1])
    for test_path in test_paths:
        test = open(test_path, 'r').read().strip()
        expected_out = open(test_path.replace('input', 'output'), 'r').read()

        splitted_input = test.strip().split('\n')
        complete_prog = fnc + '\n' + '\n'.join(splitted_input[:-1]) + "\nprint(" + splitted_input[-1] + ")\n"

        tmp = tempfile.NamedTemporaryFile(delete=False)
        try:
            tmp.write(str.encode(complete_prog))
            tmp.close()
            cmd_list = ['python3', tmp.name]
            success, outs = subprocess_run(cmd_list)
            if success:
                if outs == expected_out:
                    valid += 1
                if outs != expected_out and debug:
                    print('Fail on {}, expected: {}, actual: {}'.format(splitted_input[-1], expected_out.strip(), outs.strip()))
            else:
                if debug:
                    print('{} Fail on {}, Error is {}'.format(prog_path, test, outs))
        finally:
            del_file(tmp.name)
    print(valid)
    # print('{} {}/{}'.format(prog_path, valid, len(test_paths)))
    return valid, len(test_paths)

def exec_program_q2(prog_path, tests_path, debug=False):
    fnc = data_extractor.read_code(prog_path)
    # fnc = open(prog_path, 'r').read()
    test_paths = sorted(glob.glob(tests_path + 'input_*.txt'))
    valid = 0

    empty1, empty2, empty3 = 0, 0, 0
    tree = ast.parse(fnc)
    funcdefs = tree.body
    for func in funcdefs:
        if func.end_lineno - func.lineno <= 3:
            if func.name == 'unique_day':
                empty1 = 1
            if func.name == 'unique_month':
                empty2 = 1
            if func.name == 'contains_unique_day':
                empty3 = 1

    print(prog_path.split('/')[-1])
    t1, t2, t3 = 0, 0, 0
    for test_path in test_paths:
        test = open(test_path, 'r').read().strip()
        expected_out = open(test_path.replace('input', 'output'), 'r').read()

        splitted_input = test.strip().split('\n')
        complete_prog = fnc + '\n' + '\n'.join(splitted_input[:-1]) + "\nprint(" + splitted_input[-1] + ")\n"

        tmp = tempfile.NamedTemporaryFile(delete=False)
        try:
            tmp.write(str.encode(complete_prog))
            tmp.close()
            cmd_list = ['python3', tmp.name]
            success, outs = subprocess_run(cmd_list)
            if success:
                if outs == expected_out:
                    valid += 1
                    name = test.split('(')[0]
                    if name == 'unique_day':
                        t1 += 1
                    elif name == 'unique_month':
                        t2 += 1
                    elif name == 'contains_unique_day':
                        t3 += 1
                if outs != expected_out and debug:
                    print('Fail on {}, expected: {}, actual: {}'.format(splitted_input[-1], expected_out.strip(), outs.strip()))
            else:
                if debug:
                    print('{} Fail on {}, Error is {}'.format(prog_path, test, outs))
        finally:
            del_file(tmp.name)
    # print('{} {}/{}'.format(prog_path, valid, len(test_paths)))
    return valid, len(test_paths), t1, t2, t3, empty1, empty2, empty3

def del_file(fname):
    if os.path.exists(fname):
        os.remove(fname)


def unwrapper(expr):
    wrap_left_str = "var_dict[\'"
    wrap_right_str = "\']"
    while True:
        l1 = expr.find(wrap_left_str)
        if l1 == -1:
            break
        else:
            left_part = expr[:l1]
            cond_right = expr[l1 + len(wrap_left_str):]
            l2 = cond_right.find(wrap_right_str)
            if l2 == -1:
                print("unwrapper: something wrong")
                break
            else:
                mid_part = cond_right[:l2]
                right_part = cond_right[l2 + len(wrap_right_str):]
                expr = left_part + mid_part + right_part
    return expr


def safe_eval_list(expr_list, score_list, var_dict, mpq):
    for i in range(len(expr_list)):
        expr = expr_list[i]
        score = score_list[i]
        try:
            expr_res = eval(expr)
            mpq.put((expr, score, expr_res))
        except:
            mpq.put((expr, score, None))


class FastEvaluator:

    def parallel_eval(self, expr_list, score_list, var_dict, n_jobs=8):

        relation_dict = {}
        score_dict = {}
        result_dict = {}

        seg_len = len(expr_list) // n_jobs + 1

        sys.setrecursionlimit(1000000)

        p_list = []
        mpq_list = []
        try:
            for i in range(n_jobs):
                part_expr_list = expr_list[seg_len * i: seg_len * (i + 1)]
                part_score_list = score_list[seg_len * i: seg_len * (i + 1)]
                mpq = Queue()
                p = Process(target=safe_eval_list, args=(part_expr_list, part_score_list, var_dict, mpq))
                p_list.append(p)
                mpq_list.append(mpq)

            for p in p_list:
                p.start()

            while True:
                all_dead = not any(p.is_alive() for p in p_list)
                exists_dead = any(not p.is_alive() for p in p_list)
                all_empty = all(mpq.empty() for mpq in mpq_list)
                if all_dead and all_empty:
                    break
                elif exists_dead:
                    c = 0
                    for i in range(len(p_list)):
                        if p_list[i].is_alive():
                            continue
                        mpq = mpq_list[i]
                        if not mpq.empty():
                            expr, score, expr_res = mpq.get()
                            result_dict[expr] = expr_res
                            if expr_res is not None:

                                try:
                                    relation_dict[expr_res] = relation_dict[expr_res]
                                except:
                                    is_add = True
                                    for res_key, expr_list in relation_dict.items():
                                        if len(expr_list) > 0:
                                            res = result_dict[expr_list[0]]
                                            if res == expr_res:
                                                expr_res = res_key
                                                is_add = False
                                                break
                                    if is_add:
                                        expr_res = expr

                                if expr_res not in relation_dict.keys():
                                    relation_dict[expr_res] = []
                                if expr_res not in score_dict.keys():
                                    score_dict[expr_res] = []
                                relation_dict[expr_res].append(expr)
                                score_dict[expr_res].append(score)
                            c = c + 1
        except:
            traceback.print_exc(file=sys.stderr)
        for mpq in mpq_list:
            mpq.close()

        return relation_dict, score_dict


def rm_bb_indent(bb_code):
    new_bb_code = ""
    curr_ind_list = []
    for line in bb_code.split("\n"):
        if len(line) == 0:
            continue

        curr_ind_list.append(get_indent(line))
        new_line = rm_indent(line)
        new_bb_code += new_line + "\n"

    assert(len(set(curr_ind_list)) <= 1)

    ind = 0
    if len(curr_ind_list) > 0:
        ind = curr_ind_list[0]

    return new_bb_code, ind


def resume_bb_indent(bb_code, ind):
    new_bb_code = ""
    ind_str = "".join([" " for tmp in range(ind)])
    for line in bb_code.split("\n"):
        if len(line) == 0:
            continue

        new_line = ind_str + line
        new_bb_code += new_line + "\n"
    return new_bb_code


def regularize(code):
    '''change code style (tab to space)'''
    # remove comment
    code = astunparse.unparse(ast.parse(code))

    # put logical lines into one physical line
    token_list = get_token_list(code)

    new_code = ""
    tmp_list = []
    indent_str = ""

    new_line_flag = False
    for token in token_list:
        if tok_name[token.exact_type] in ["NEWLINE", "ENDMARKER"]:
            new_code += indent_str + " ".join([tmp_token.string for tmp_token in tmp_list]) + "\n"
            tmp_list = []
            new_line_flag = True
        elif tok_name[token.exact_type] == "NL":
            pass
        elif tok_name[token.exact_type] == "COMMENT":
            pass
        elif tok_name[token.exact_type] == "INDENT":
            indent_str += "    "
        elif tok_name[token.exact_type] == "DEDENT":
            if new_line_flag:
                indent_str = indent_str[:-4]
        else:
            new_line_flag = False
            tmp_list.append(token)

    final_code = ""
    for line in new_code.split("\n"):

        token_list = get_token_list(line)
        if any([token.string in ["from", "import"] for token in token_list]):
            pass
        else:
            if get_indent(line) == 0 and \
                len(token_list) > 1 and \
                    all([token.string != "def" for token in token_list]):
                pass
            else:
                final_code += line + "\n"

    return final_code


def get_vari_names(code):
    vari_name_list = []
    root = ast.parse(code)
    for node in ast.walk(root):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            vari_name_list.append(str(node.id))
        elif isinstance(node, ast.arg):
            vari_name_list.append(str(node.arg))
    vari_name_list = list(set(vari_name_list))
    return vari_name_list


def swt_func_vn(func_code, vn_map):
    class VMTransformer(ast.NodeTransformer):
        def __init__(self, n_map):
            self.__n_map = n_map
            super()

        def visit_Name(self, node):
            if node.id in self.__n_map.keys():
                node.id = self.__n_map[node.id]
            return node

        def visit_arg(self, node):
            if node.arg in self.__n_map.keys():
                node.arg = self.__n_map[node.arg]
            return node

    tree = ast.parse(func_code)

    vmt = VMTransformer(vn_map)
    swt_tree = vmt.visit(tree)

    swt_func_code = astunparse.unparse(swt_tree)
    return regularize(swt_func_code)


def syntax_check(code):
    try:
        compile(code, "<string>", "exec")
        return True
    except:
        return False

import io, tokenize, re
def remove_comments_and_docstrings(source):
    io_obj = io.StringIO(source)
    out = ""
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0
    for tok in tokenize.generate_tokens(io_obj.readline):
        token_type = tok[0]
        token_string = tok[1]
        start_line, start_col = tok[2]
        end_line, end_col = tok[3]
        ltext = tok[4]
        if start_line > last_lineno:
            last_col = 0
        if start_col > last_col:
            out += (" " * (start_col - last_col))
        if token_type == tokenize.COMMENT:
            pass
        elif token_type == tokenize.STRING:
            if prev_toktype != tokenize.INDENT:
                if prev_toktype != tokenize.NEWLINE:
                    if start_col > 0:
                        out += token_string
        else:
            out += token_string
        prev_toktype = token_type
        last_col = end_col
        last_lineno = end_line
    out = '\n'.join(l for l in out.splitlines() if l.strip())
    return out