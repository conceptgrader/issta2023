"""
Control flow graph builder.
"""
# Aurelien Coet, 2018.
# Modified by Andrei Nacu, 2020

import ast
from .model import Block, Link, CFG
import sys
import astor

from src import helper
from src.concept_graph.concept import Concept


def is_py38_or_higher():
    if sys.version_info.major == 3 and sys.version_info.minor >= 8:
        return True
    return False


NAMECONSTANT_TYPE = ast.Constant if is_py38_or_higher() else ast.NameConstant


def invert(node):
    """
    Invert the operation in an ast node object (get its negation).

    Args:
        node: An ast node object.

    Returns:
        An ast node object containing the inverse (negation) of the input node.
    """
    inverse = {ast.Eq: ast.NotEq,
               ast.NotEq: ast.Eq,
               ast.Lt: ast.GtE,
               ast.LtE: ast.Gt,
               ast.Gt: ast.LtE,
               ast.GtE: ast.Lt,
               ast.Is: ast.IsNot,
               ast.IsNot: ast.Is,
               ast.In: ast.NotIn,
               ast.NotIn: ast.In}

    if type(node) == ast.Compare:
        op = type(node.ops[0])
        inverse_node = ast.Compare(left=node.left, ops=[inverse[op]()],
                                   comparators=node.comparators)
    elif isinstance(node, ast.BinOp) and type(node.op) in inverse:
        op = type(node.op)
        inverse_node = ast.BinOp(node.left, inverse[op](), node.right)
    elif type(node) == NAMECONSTANT_TYPE and node.value in [True, False]:
        inverse_node = NAMECONSTANT_TYPE(value=not node.value)
    else:
        inverse_node = ast.UnaryOp(op=ast.Not(), operand=node)

    return inverse_node


def merge_exitcases(exit1, exit2):
    """
    Merge the exitcases of two Links.

    Args:
        exit1: The exitcase of a Link object.
        exit2: Another exitcase to merge with exit1.

    Returns:
        The merged exitcases.
    """
    if exit1:
        if exit2:
            return ast.BoolOp(ast.And(), values=[exit1, exit2])
        return exit1
    return exit2


class CFGBuilder(ast.NodeVisitor):
    """
    Control flow graph builder.

    A control flow graph builder is an ast.NodeVisitor that can walk through
    a program's AST and iteratively build the corresponding CFG.
    """

    def __init__(self, separate=False):
        super().__init__()
        self.after_loop_block_stack = []
        self.curr_loop_guard_stack = []
        self.current_block = None
        self.separate_node_blocks = separate
        self.var_list = []

    # ---------- CFG building methods ---------- #
    def build(self, name, tree, asynchr=False, entry_id=0):
        """
        Build a CFG from an AST.

        Args:
            name: The name of the CFG being built.
            tree: The root of the AST from which the CFG must be built.
            async: Boolean indicating whether the CFG being built represents an
                   asynchronous function or not. When the CFG of a Python
                   program is being built, it is considered like a synchronous
                   'main' function.
            entry_id: Value for the id of the entry block of the CFG.

        Returns:
            The CFG produced from the AST.
        """
        self.cfg = CFG(name, asynchr=asynchr)
        # Tracking of the current block while building the CFG.
        self.current_id = entry_id
        self.current_block = self.new_block()
        self.cfg.entryblock = self.current_block
        # Actual building of the CFG is done here.
        self.visit(tree)
        self.clean_cfg(self.cfg.entryblock)
        return self.cfg

    def build_from_src(self, name, src):
        """
        Build a CFG from some Python source code.

        Args:
            name: The name of the CFG being built.
            src: A string containing the source code to build the CFG from.

        Returns:
            The CFG produced from the source code.
        """
        tree = ast.parse(src, mode='exec')
        return self.build(name, tree)

    def build_from_file(self, name, filepath):
        """
        Build a CFG from some Python source file.

        Args:
            name: The name of the CFG being built.
            filepath: The path to the file containing the Python source code
                      to build the CFG from.

        Returns:
            The CFG produced from the source file.
        """
        with open(filepath, 'r') as src_file:
            src = src_file.read()
            src = helper.remove_comments_and_docstrings(src)
            return self.build_from_src(name, src)

    # ---------- Graph management methods ---------- #
    def new_block(self):
        """
        Create a new block with a new id.

        Returns:
            A Block object with a new unique id.
        """
        self.current_id += 1
        return Block(self.current_id)

    def add_statement(self, block, statement):
        """
        Add a statement to a block.

        Args:
            block: A Block object to which a statement must be added.
            statement: An AST node representing the statement that must be
                       added to the current block.
        """
        block.statements.append(statement)

    def add_concept(self, block, concept_node):

        block.concepts.append(concept_node)

    def add_exit(self, block, nextblock, exitcase=None):
        """
        Add a new exit to a block.

        Args:
            block: A block to which an exit must be added.
            nextblock: The block to which control jumps from the new exit.
            exitcase: An AST node representing the 'case' (or condition)
                      leading to the exit from the block in the program.
        """
        newlink = Link(block, nextblock, exitcase)
        block.exits.append(newlink)
        nextblock.predecessors.append(newlink)

    def new_loopguard(self):
        """
        Create a new block for a loop's guard if the current block is not
        empty. Links the current block to the new loop guard.

        Returns:
            The block to be used as new loop guard.
        """
        if (self.current_block.is_empty() and
                len(self.current_block.exits) == 0):
            # If the current block is empty and has no exits, it is used as
            # entry block (condition test) for the loop.
            loopguard = self.current_block
        else:
            # Jump to a new block for the loop's guard if the current block
            # isn't empty or has exits.
            loopguard = self.new_block()
            self.add_exit(self.current_block, loopguard)
        return loopguard

    def new_functionCFG(self, node, asynchr=False):
        """
        Create a new sub-CFG for a function definition and add it to the
        function CFGs of the CFG being built.

        Args:
            node: The AST node containing the function definition.
            async: Boolean indicating whether the function for which the CFG is
                   being built is asynchronous or not.
        """
        self.current_id += 1
        # A new sub-CFG is created for the body of the function definition and
        # added to the function CFGs of the current CFG.
        func_body = ast.Module(body=node.body)
        func_builder = CFGBuilder()
        self.cfg.functioncfgs[node.name] = func_builder.build(node.name,
                                                              func_body,
                                                              asynchr,
                                                              self.current_id)
        self.current_id = func_builder.current_id + 1

    def clean_cfg(self, block, visited=[]):
        """
        Remove the useless (empty) blocks from a CFG.

        Args:
            block: The block from which to start traversing the CFG to clean
                   it.
            visited: A list of blocks that already have been visited by
                     clean_cfg (recursive function).
        """
        # Don't visit blocks twice.
        if block in visited:
            return
        visited.append(block)

        # Empty blocks are removed from the CFG.
        if block.is_empty():
            for pred in block.predecessors:
                for exit in block.exits:
                    self.add_exit(pred.source, exit.target,
                                  merge_exitcases(pred.exitcase,
                                                  exit.exitcase))
                    # Check if the exit hasn't yet been removed from
                    # the predecessors of the target block.
                    if exit in exit.target.predecessors:
                        exit.target.predecessors.remove(exit)
                # Check if the predecessor hasn't yet been removed from
                # the exits of the source block.
                if pred in pred.source.exits:
                    pred.source.exits.remove(pred)

            block.predecessors = []
            # as the exits may be modified during the recursive call, it is unsafe to iterate on block.exits
            # Created a copy of block.exits before calling clean cfg , and iterate over it instead.
            for exit in block.exits[:]:
                self.clean_cfg(exit.target, visited)
            block.exits = []
        else:
            for exit in block.exits[:]:
                self.clean_cfg(exit.target, visited)

    # ---------- AST Node visitor methods ---------- #
    def goto_new_block(self, node):
        if self.separate_node_blocks:
            newblock = self.new_block()
            self.add_exit(self.current_block, newblock)
            self.current_block = newblock
        self.generic_visit(node)

    def visit_Constant(self, node):
        return str(node.value)

    def visit_Name(self, node):
        return node.id

    def visit_Expr(self, node):
        # Todo: take care of it
        action, target, val = '', '', ''
        concept = ''
        if isinstance(node.value, ast.Call):
            if hasattr(node.value.func, 'attr'):
                if node.value.func.attr == 'append':
                    action = 'insert'
                    target = node.value.func.value.id
                    val =  "".join([astor.to_source(arg) for arg in node.value.args])
                    concept = 'insert {} to {}'.format("".join([astor.to_source(arg) for arg in node.value.args]), node.value.func.attr)
                else:
                    action = 'call'
                    target = node.value.func.attr
                    val = "".join([astor.to_source(arg) for arg in node.value.args])
                    concept = 'calling {} with {}'.format(node.value.func.attr, "".join([astor.to_source(arg) for arg in node.value.args]))
            else:
                action = 'call'
                target = node.value.func.id
                val = "".join([astor.to_source(arg) for arg in node.value.args])
                concept = 'calling {} with {}'.format(node.value.func.id, "".join([ astor.to_source(arg) for arg in node.value.args]))
        concept += '\n'

        self.add_statement(self.current_block, node)
        concept_node = Concept(category=action, target=target, val=val)
        self.add_concept(self.current_block, concept_node)
        # self.add_statement(self.current_block, node)
        self.goto_new_block(node)

    def visit_Call(self, node):

        #Todo: nested function call
        def visit_func(node):
            if type(node) == ast.Name:
                return node.id
            elif type(node) == ast.Attribute:
                # Recursion on series of calls to attributes.
                func_name = visit_func(node.value)
                func_name += "." + node.attr
                return func_name
            elif type(node) == ast.Str:
                return node.s
            elif type(node) == ast.Subscript:
                return node.value.id
            elif type(node) == ast.Constant:
                return node.value
        concept = Concept()
        concept.set_category('call')
        func = node.func
        func_name = visit_func(func)
        built_fnc = ['range', 'enumerate']
        if func_name in built_fnc:
            member = func_name
        else:
            member = func_name
        concept.add_members(member)
        self.current_block.func_calls.append(func_name)
        return concept

    def visit_Assign(self, node):
        assert len(node.targets) == 1
        action, target = '', ''
        # ToDo: Fix target is subscript
        if isinstance(node.targets[0], ast.Name):
            # x = a
            target = node.targets[0].id
            if target not in self.var_list and (not isinstance(node.value, ast.Constant)):# and not isinstance(node.value, ast.List)):
                action = 'declare'
            elif target in self.var_list:
                action = 'update'
            self.var_list.append(target)

        if 'update' in action and (isinstance(node.value, ast.Constant) or isinstance(node.value, ast.List)):
            action = 'clear'

        if isinstance(node.value, ast.Call):
            action = 'by calling {} with {}'.format(self.visit_Call(node.value),
                                                    " and ".join([astor.to_source(arg) for arg in node.value.args]))
        concept = action + target

        if action != '':
            self.add_statement(self.current_block, concept+'\n')
            concept_node = Concept(category=action, target=target)
            self.add_concept(self.current_block, concept_node)

        self.goto_new_block(node)

    def visit_AnnAssign(self, node):
        self.add_statement(self.current_block, node)
        self.goto_new_block(node)

    def visit_AugAssign(self, node):

        # ToDo: Fix node.subscript
        action = ''
        # target = node.target.id
        target = self.visit(node.target)
        value = ''

        if isinstance(node.op, ast.Add) or isinstance(node.op, ast.Sub):
            action = 'update'

        if isinstance(node.value, ast.Call):
            a= self.visit_Call(node.value)
            # value = ' by ' + self.visit_Call(node.value)
            value = str(a)
        concept = action + ' ' + target + ' ' + value

        self.add_statement(self.current_block, concept+'\n')
        concept_node = Concept(category=action, target=target)
        self.add_concept(self.current_block, concept_node)
        self.goto_new_block(node)

    def visit_Raise(self, node):
        # TODO
        pass

    def visit_Assert(self, node):
        self.add_statement(self.current_block, node)
        # New block for the case in which the assertion 'fails'.
        failblock = self.new_block()
        self.add_exit(self.current_block, failblock, invert(node.test))
        # If the assertion fails, the current flow ends, so the fail block is a
        # final block of the CFG.
        self.cfg.finalblocks.append(failblock)
        # If the assertion is True, continue the flow of the program.
        successblock = self.new_block()
        self.add_exit(self.current_block, successblock, node.test)
        self.current_block = successblock
        self.goto_new_block(node)

    def visit_Compare(self, node):

        concept = self.visit(node.ops[0])
        concept_node = Concept()
        concept_node.set_category(concept)
        if isinstance(node.left, ast.Name) or isinstance(node.left, ast.Constant):
            left_val = self.visit(node.left)
            concept_node.add_members(left_val)
        else:
            left_val = type(node.left).__name__
            concept_node.add_members(left_val)
            sub_concept = self.visit(node.left)
            concept_node.add_subconcept(sub_concept)

        if isinstance(node.comparators[0], ast.Name) or isinstance(node.comparators[0], ast.Constant):
            right_val = self.visit(node.comparators[0])
            concept_node.add_members(right_val)
        else:
            right_val = type(node.comparators[0]).__name__
            concept_node.add_members(right_val)
            sub_concept = self.visit(node.comparators[0])
            concept_node.add_subconcept(sub_concept)
        return concept_node

    def visit_LtE(self, node):
        concept = 'numeric relation'
        return concept

    def visit_Lt(self, node):
        concept = 'numeric relation'
        return concept

    def visit_GtE(self, node):
        concept = 'numeric relation'
        return concept

    def visit_Gt(self, node):
        concept = 'numeric relation'
        return concept

    def visit_Is(self, node):
        return 'equivalence relation'

    def visit_IsNot(self, node):
        return 'equivalence relation'

    def visit_In(self, node):
        concept = 'containment relation'
        return concept

    def visit_NotIn(self, node):
        concept = 'containment relation'
        return concept

    def visit_Eq(self, node):
        concept = 'equivalence relation'
        return concept

    def visit_NotEq(self, node):
        concept = 'equivalence relation'
        return concept

    def visit_Add(self, node):
        concept = 'sum of'
        return concept

    def visit_Sub(self, node):
        concept = 'difference between'
        return concept

    def visit_BinOp(self, node):
        left_concept = self.visit(node.left)
        right_concept = self.visit(node.right)
        op_concept = self.visit(node.op)
        concept_node = Concept(category=op_concept, target=left_concept, val=right_concept)
        return concept_node

    def visit_BoolOp(self, node):

        op = type(node.op).__name__
        left_concept = self.visit(node.values[0])
        right_concept = self.visit(node.values[1])

        if left_concept is None:
            left_concept = ''
        if right_concept is None:
            right_concept = ''

        concept_node = Concept(category=op, target=left_concept, val=right_concept)
        return concept_node

    def visit_UnaryOp(self, node):
        concept = 'existing relation'
        return Concept(category=concept)

    def visit_Subscript(self, node):
        concept_node = Concept()

        if isinstance(node.slice, ast.Index):
            concept = 'element of'
            concept_node.set_category(concept)
            if isinstance(node.slice.value, ast.Name) or isinstance(node.slice.value, ast.Constant):
                value = self.visit(node.value)
                element = self.visit(node.slice.value)
                concept_node.add_members(value)
                concept_node.add_members(element)
            else:
                member = type(node.slice.value).__name__
                concept_node.add_members(member)
                sub_concept = self.visit(node.slice.value)
                concept_node.add_subconcept(sub_concept)
        elif isinstance(node.slice, ast.Slice):
                concept = 'subrange'
                concept_node.set_category(concept)
                self.visit(node.slice)
                print('subrangeggg')
                pass
        else:
            print("Abnormal Slice Type")
        return concept_node

    def visit_If(self, node):
        # Add the If statement at the end of the current block.
        # ToDO: move check for every if
        concept = 'check '
        concept_node = Concept()
        if isinstance(node.test, ast.Name) or isinstance(node.test, ast.Constant): # if a: if true:
            condition = self.visit(node.test)
            concept_node.add_members(condition)
        else:
            condition = self.visit(node.test)
            concept += condition.category
            concept_node.add_subconcept(condition)

        concept_node.set_category(concept)

        self.add_statement(self.current_block, node)
        self.add_concept(self.current_block, concept_node)

        # Create a new block for the body of the if.
        if_block = self.new_block()
        self.add_exit(self.current_block, if_block, node.test)

        # Create a block for the code after the if-else.
        afterif_block = self.new_block()

        # New block for the body of the else if there is an else clause.
        if len(node.orelse) != 0:
            else_block = self.new_block()
            self.add_exit(self.current_block, else_block, invert(node.test))
            self.current_block = else_block
            # Visit the children in the body of the else to populate the block.
            for child in node.orelse:
                self.visit(child)
            # If encountered a break, exit will have already been added
            if not self.current_block.exits:
                self.add_exit(self.current_block, afterif_block)
        else:
            self.add_exit(self.current_block, afterif_block, invert(node.test))

        # Visit children to populate the if block.
        self.current_block = if_block
        for child in node.body:
            self.visit(child)
        if not self.current_block.exits:
            self.add_exit(self.current_block, afterif_block)

        # Continue building the CFG in the after-if block.
        self.current_block = afterif_block

    def visit_While(self, node):
        loop_guard = self.new_loopguard()
        self.current_block = loop_guard

        # self.add_statement(self.current_block, node)

        self.add_statement(self.current_block, self.visit(node.test))
        self.curr_loop_guard_stack.append(loop_guard)


        # New block for the case where the test in the while is True.
        while_block = self.new_block()
        self.add_exit(self.current_block, while_block, node.test)

        # New block for the case where the test in the while is False.
        afterwhile_block = self.new_block()
        self.after_loop_block_stack.append(afterwhile_block)
        inverted_test = invert(node.test)
        # Skip shortcut loop edge if while True:
        if not (isinstance(inverted_test, NAMECONSTANT_TYPE) and
                inverted_test.value is False):
            self.add_exit(self.current_block, afterwhile_block, inverted_test)

        # Populate the while block.
        self.current_block = while_block
        for child in node.body:
            self.visit(child)
        if not self.current_block.exits:
            # Did not encounter a break statement, loop back
            self.add_exit(self.current_block, loop_guard)

        # Continue building the CFG in the after-while block.
        self.current_block = afterwhile_block
        self.after_loop_block_stack.pop()
        self.curr_loop_guard_stack.pop()

    def visit_For(self, node):
        loop_guard = self.new_loopguard()
        self.current_block = loop_guard

        concept_node = Concept()
        concept_node.set_category('iterate')
        target = self.visit(node.target)
        iter = self.visit(node.iter)
        concept_node.add_members(target)
        concept_node.add_members(iter)
        # concept = 'iterate {} through {}\n'.format(target, iter)
        self.add_concept(self.current_block, concept_node)
        self.add_statement(self.current_block, node)
        self.curr_loop_guard_stack.append(loop_guard)
        # New block for the body of the for-loop.
        for_block = self.new_block()
        self.add_exit(self.current_block, for_block, node.iter)

        # Block of code after the for loop.
        afterfor_block = self.new_block()
        self.add_exit(self.current_block, afterfor_block)
        self.after_loop_block_stack.append(afterfor_block)
        self.current_block = for_block

        # Populate the body of the for loop.
        for child in node.body:
            self.visit(child)
        if not self.current_block.exits:
            # Did not encounter a break
            self.add_exit(self.current_block, loop_guard)

        # Continue building the CFG in the after-for block.
        self.current_block = afterfor_block
        # Popping the current after loop stack,taking care of errors in case of nested for loops
        self.after_loop_block_stack.pop()
        self.curr_loop_guard_stack.pop()

    def visit_Break(self, node):
        assert len(self.after_loop_block_stack), "Found break not inside loop"
        self.add_exit(self.current_block, self.after_loop_block_stack[-1])

    def visit_Continue(self, node):
        assert len(self.curr_loop_guard_stack), "Found continue outside loop"
        self.add_exit(self.current_block, self.curr_loop_guard_stack[-1])

    def visit_Import(self, node):
        self.add_statement(self.current_block, node)

    def visit_ImportFrom(self, node):
        self.add_statement(self.current_block, node)

    def visit_FunctionDef(self, node):
        self.add_statement(self.current_block, node)
        self.new_functionCFG(node, asynchr=False)

    def visit_AsyncFunctionDef(self, node):
        self.add_statement(self.current_block, node)
        self.new_functionCFG(node, asynchr=True)

    def visit_Await(self, node):
        afterawait_block = self.new_block()
        self.add_exit(self.current_block, afterawait_block)
        self.goto_new_block(node)
        self.current_block = afterawait_block

    def visit_Return(self, node):

        concept_node = Concept()
        concept_node.set_category('return')
        if node.value is None: #return
            pass
        elif isinstance(node.value, ast.Name) or isinstance(node.value, ast.Constant):
            member = self.visit(node.value)
            concept_node.add_members(member)
        else:
            member = type(node.value).__name__
            subconcept = self.visit(node.value)
            concept_node.add_members(member)
            concept_node.add_subconcept(subconcept)
        # concept = 'return ' + str(self.visit(node.value))
        # concept = 'return ' + str(target)
        # self.add_statement(self.current_block, concept)
        self.add_statement(self.current_block, node)
        self.add_concept(self.current_block, concept_node)
        self.cfg.finalblocks.append(self.current_block)
        # Continue in a new block but without any jump to it -> all code after
        # the return statement will not be included in the CFG.
        self.current_block = self.new_block()

    def visit_Yield(self, node):
        self.cfg.asynchr = True
        afteryield_block = self.new_block()
        self.add_exit(self.current_block, afteryield_block)
        self.current_block = afteryield_block
