class Concept(object):

    def __init__(self, category='', target='', val=''):
        # action(op) used in a concept.
        self.category = category
        # target(lhs) of a concept
        self.target = target
        # val (rhs) in a concept
        self.val = val
        self.members = []
        self.sub_concepts = []

    def set_category(self, category):
        self.category = category

    def add_members(self, new_member):
        self.members.append(new_member)

    def add_subconcept(self, subconcept):
        self.sub_concepts.append(subconcept)

    def compare(self, ref_concept):
        if self == ref_concept:
            return 1
        else:
            return 0

    def __str__(self):
        return 'category: {}, members: {}'.format(self.category, self.members)
        # return 'category: {}, target: {}, val: {}'.format(self.category, self.target, self.val)

    def __eq__(self, other):
        return self.category == other.category

    def __hash__(self):
        return hash(str(self))