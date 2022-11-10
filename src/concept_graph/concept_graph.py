import sys
import networkx as nx
from src.concept_graph.builder import CFGBuilder

def create_ref_concept_graph_path(ref_path):
    concept_graphs = {}
    names = []
    exceptions = []
    syntax_error = []
    for ref in ref_path:
        try:
            ref_name = ref.split('/')[-1]
            concept_graph = create_concept_graph(ref)
            concept_graphs[ref_name] = create_net_graph_by_concept_node(concept_graph, concept_graph.name)
            names.append(ref_name)
        except SyntaxError:
            syntax_error.append(ref_name)
        except:
            exceptions.append(sys.exc_info())
    return names, concept_graphs, len(syntax_error), len(exceptions)


def create_stu_concept_graph_path(submitted_path):
    graphs = {}
    names = []
    exceptions = []
    syntax_error = []
    for assign in submitted_path:
        try:
            student_name = assign.split('/')[-1]
            cfg = create_concept_graph(assign)
            graph = create_net_graph_by_concept_node(cfg, cfg.name)
            if student_name not in names:
                names.append(student_name)
            graphs[student_name] = graph
        except SyntaxError:
            syntax_error.append(student_name)
        # except:
        print(assign.split('/')[-1])
        exceptions.append(sys.exc_info())
    return names, graphs, len(syntax_error), len(exceptions)


def create_concept_graph(assign):
    cfg = CFGBuilder().build_from_file(assign, assign)
    return cfg

def create_net_graph_by_concept_node(cfg, name):

    concept_graph_fnc = {}
    for fnc, fnc_cfg in cfg.functioncfgs.items():
        CG = nx.DiGraph()
        CG.name = name
        traverse_by_concept_node(fnc_cfg.entryblock, CG, fnc)
        concept_graph_fnc[fnc] = CG
    return concept_graph_fnc

def traverse_by_concept_node(block, G, fnc):
    #Todo: what if only one statement, no exit
    for exit_link in block.exits:
        target_blk = exit_link.target
        source_blk = exit_link.source

        source_concepts = []
        for concept in source_blk.concepts:
            source_concepts.append(concept)

        target_concepts = []
        for concept in target_blk.concepts:
            target_concepts.append(concept)

        if not G.has_edge(tuple(source_concepts), tuple(target_concepts)):
            G.add_node(tuple(source_concepts))
            G.add_node(tuple(target_concepts))
            G.add_edge(tuple(source_concepts), tuple(target_concepts))
            traverse_by_concept_node(target_blk, G, fnc)

def replace_vari_in_concept(key, val, stmt):
    concepts = stmt.split('\n')
    new_concepts = []
    for concept in concepts:
        ss = concept.split(' ')
        ss = [val if key == s else s for s in ss]
        new_concept = ' '.join(ss)
        new_concepts.append(new_concept)
    new_stmt = '\n'.join(new_concepts)
    return new_stmt

def traverse(block, G, fnc, var_mapping=None):
    for exit_link in block.exits:
        target_blk = exit_link.target
        source_blk = exit_link.source

        source_mapped_statements = []
        for stmt in source_blk.statements:
            if var_mapping is not None:
                for key, val in var_mapping[fnc].items():
                    if key in stmt and 'buggy_' not in val and 'ref_' not in key:
                        stmt = replace_vari_in_concept(key, val, stmt)
                        # stmt = stmt.replace(' '+key, ' '+val)
            source_mapped_statements.append(stmt)

        target_mapped_statements = []
        for stmt in target_blk.statements:
            if var_mapping is not None:
                for key, val in var_mapping[fnc].items():
                    if key in stmt and 'buggy_' not in val and 'ref_' not in key:
                        stmt = replace_vari_in_concept(key, val, stmt)
                        # stmt = stmt.replace(' '+key, ' '+val)
            target_mapped_statements.append(stmt)

        if G.has_edge(''.join(source_mapped_statements), ''.join(target_mapped_statements)):
            return
        G.add_node(''.join(target_mapped_statements))
        G.add_edge(''.join(source_mapped_statements), ''.join(target_mapped_statements))
        traverse(target_blk, G, fnc, var_mapping=var_mapping)
