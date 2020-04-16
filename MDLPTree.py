import copy
import numbers
import pprint

import math
from pandas import Series

SINGLE_OBSERVATION_PENALTY = 1


class MDLPTree:
    """
    Class to model an decision tree nodes
    """

    def __init__(self, attribute=None, split_value=None, value=None, data=None, remaining=None, classes=None,
                 class_attr=None) -> None:
        """
        Initializes a node (or leaf)

        :param attribute: The attribute that this node splits on (node only)
        :param split_value: the real-value or category to split on
        :param value:
            Leaf: class
            Node: (sub-node if true or < split_value, sub-node if false or >= split_value)
        """

        self.attribute = attribute
        self.split_value = split_value
        self.value = value
        self.parent = None
        self.data = data
        self.remaining = remaining
        self.classes = classes
        self.class_attr = class_attr

        if not self.is_leaf():
            for x in self.value:
                x.parent = self

        if self.value is None and data is not None and class_attr is not None:
            self.value = data[class_attr].mode()[0]
            if self.classes is None:
                self.classes = list(set(data[class_attr]))
            if self.remaining is None:
                self.remaining = dict([(a, list(set(data[a]))) for a in data.columns if a != class_attr])

                # Check if values should be minimized
                for k, v in self.remaining.items():
                    if len(v) < 16:
                        continue

                    splits = math.floor(math.sqrt(len(v)))
                    block_len = len(v) / (splits + 1)
                    indices = [round((i + 1) * block_len) for i in range(splits)]
                    self.remaining[k] = [v[i] for i in indices]

    def is_leaf(self):
        """
        Returns whether the node is a leaf
        """
        return self.attribute is None

    def has_discrete_attribute(self):
        """
        Returns whether the node has a discrete attribute
        """
        return not self.is_leaf() and isinstance(self.split_value, str)

    def has_continuous_attribute(self):
        """
        Returns whether the node has a continuous attribute
        """
        return not self.is_leaf() and isinstance(self.split_value, numbers.Number)

    def is_decision_node(self):
        """
        Returns whether the node is a decision node (has only leafs as sub-nodes)
        """
        if self.is_leaf():
            return False
        return all([x.is_leaf() for x in self.sub_nodes()])

    def sub_nodes(self):
        """
        Returns all sub_nodes of the node (or an empty list if leafs)
        """
        if self.is_leaf():
            return []
        else:
            return self.value

    def decision_nodes(self):
        """
        Returns all decision-nodes
        """
        if self.is_decision_node():
            return [self]
        elif self.is_leaf():
            return []
        else:
            return [n for v in self.sub_nodes() for n in v.decision_nodes() if not v.is_leaf()]

    def leafs(self):
        """
        Returns all leafs
        """
        if self.is_leaf():
            return [self]
        else:
            return [n for v in self.sub_nodes() for n in v.leafs()]

    def node_count(self):
        """
        Returns how many non-leafs there are in the (sub) nodes
        """
        if self.is_leaf():
            return 0
        else:
            return 1 + sum([x.node_count() for x in self.sub_nodes()])

    def leaf_count(self):
        """
        Returns how many leafs there are in the (sub) nodes
        """
        if self.is_leaf():
            return 1
        else:
            return sum([x.leaf_count() for x in self.sub_nodes()])

    @staticmethod
    def comb(n, k):
        """
        Calculates the combinations n over k
        """
        return math.factorial(n) // (math.factorial(n - k) * math.factorial(k))

    def cost(self):
        """
        Calculates the cost of a node

        PLEASE NOTE: This requires, data, classes, class_attr and remaining_classes
        """
        n = len(self.data)
        if self.is_leaf():
            # Calculate the tree cost
            tree_cost = 1 + math.log2(len(self.classes))

            # Calculate the exception cost
            if len(self.classes) == 2:
                b = math.floor(n / 2)
                k = len(self.data[self.data[self.class_attr] == self.value])
                return tree_cost + math.log2(b + 1) + math.log2(self.comb(n, k))
            else:
                # Not possible to have exceptions if there is only 1 data frame
                if n == 1:
                    return tree_cost + SINGLE_OBSERVATION_PENALTY

                other_classes = self.classes.copy()
                other_classes.remove(self.value)
                cost = 1
                k = 0
                # Find all n over k(i) combinations
                for c_i in other_classes:
                    k_i = len(self.data[self.data[self.class_attr] == c_i])
                    cost *= self.comb(n, k_i)
                    k += k_i

                # Multiply the cost by the (n + k - 1) over (k - 1) combinations
                if k > 0:
                    cost *= self.comb(n + k - 1, k - 1)
                else:
                    # If no exceptions, use this method
                    cost *= n
                return tree_cost + math.log2(cost)
        else:
            # Calculate tree cost
            cost = 1 + math.log2(len(self.remaining))
            remaining_values = len(self.remaining[self.attribute])
            if remaining_values > 2:
                cost += math.log2(remaining_values)

            # Return tree cost plus the cost of the subtree
            return cost + self.value[0].cost() + self.value[1].cost()

    def expand(self):
        # Do not expand if there are no remaining classes or if there is only one outcome class
        if len(self.remaining) == 0 or len(set(self.data[self.class_attr])) <= 1:
            # Make leaf with most prevailing class
            self.value = self.data[self.class_attr].mode()[0]
            self.split_value = None
            self.attribute = None
            return self
        else:
            score = {}
            tree = {}

            split_values = [(a, v) for a, l in self.remaining.items() for v in l]
            for (a, v) in split_values:
                # Remove from the remaining classes
                rc = copy.deepcopy(self.remaining)
                rc[a].remove(v)
                if (len(rc[a])) == 1:
                    del rc[a]

                if isinstance(v, numbers.Number):
                    d0 = self.data[self.data[a] < v]
                    d1 = self.data[self.data[a] >= v]
                else:
                    d0 = self.data[self.data[a] == v]
                    d1 = self.data[self.data[a] != v]

                if len(d0) == 0 or len(d1) == 0:
                    continue

                # Construct subtree
                n = MDLPTree(a, v, (MDLPTree(data=d0, remaining=rc, classes=self.classes, class_attr=self.class_attr),
                                    MDLPTree(data=d1, remaining=rc, classes=self.classes, class_attr=self.class_attr)),
                             data=self.data, remaining=self.remaining, classes=self.classes, class_attr=self.class_attr)

                tree[(a, v)] = n.value
                score[(a, v)] = n.cost()

            if len(score) == 0:
                self.value = self.data[self.class_attr].mode()[0]
                self.split_value = None
                self.attribute = None
                return self

            a, v = sorted(score.items(), key=lambda item: item[1])[0][0]
            self.attribute = a
            self.split_value = v
            self.value = tree[(a, v)]

            for x in self.value:
                x.expand()

            return self

    def __force_prune(self):
        """
        Forces pruning of this nodes, regardless of costs or it being a decision node or not
        """
        self.attribute = None
        self.split_value = None
        self.value = self.data[self.class_attr].mode()[0]
        return self

    def __prune(self) -> bool:
        """
        Prunes a decision node if pruning reduces the cost
        :return: True if the node was pruned, false otherwise
        """
        if not self.is_decision_node():
            raise AssertionError("__prune() can only be invoked on decision nodes")

        if self.__copy__().__force_prune().cost() < self.cost():
            # Prune
            self.__force_prune()
            return True

        return False

    def prune(self):
        """
        Recursively prunes all decision nodes where pruning reduces the cost
        """
        c = True
        while c:
            r = [False]
            nodes = self.decision_nodes()
            for n in nodes:
                r.append(n.__prune())

            c = any(r)
        return self

    @staticmethod
    def train(data, class_attr='Class'):
        return MDLPTree(data=data, class_attr=class_attr).expand().prune()

    def predict(self, series):
        if self.is_leaf():
            return self.value

        if self.has_discrete_attribute():
            if series[self.attribute] == self.split_value:
                return self.value[0].predict(series)
            else:
                return self.value[1].predict(series)
        else:
            if series[self.attribute] < self.split_value:
                return self.value[0].predict(series)
            else:
                return self.value[1].predict(series)

    def predict_all(self, df):
        correct = 0
        res = []
        for i, s in df.iterrows():
            x = self.predict(s)
            if x == s[self.class_attr]:
                correct += 1
            res.append(x)

        return Series(dict([(i, self.predict(s)) for i, s in df.iterrows()]))

    def __copy__(self):
        return MDLPTree(
            copy.deepcopy(self.attribute),
            copy.deepcopy(self.split_value),
            copy.deepcopy(self.value),
            copy.deepcopy(self.data),
            copy.deepcopy(self.remaining),
            copy.deepcopy(self.classes),
            copy.deepcopy(self.class_attr),

        )

    def __getitem__(self, item):
        """
        Get's the sub node by the attribute value (does not work for leafs)
        :param item: attribute value
        :return: Corresponding sub-nodes
        """
        assert not self.is_leaf(), 'Cannot get item from leaf'

        if self.has_discrete_attribute():
            if item == self.split_value:
                return self.value[0]
            else:
                return self.value[1]
        else:
            if item < self.split_value:
                return self.value[0]
            else:
                return self.value[1]

    def __repr__(self):
        """
        Returns a readable representation of the node
        """
        if self.is_leaf():
            return self.value

        if self.has_discrete_attribute():
            return {
                self.attribute + '[' + self.split_value + ']': self.value[0].__repr__(),
                self.attribute + '[not ' + self.split_value + ']': self.value[1].__repr__(),
            }
        else:
            return {
                self.attribute + '[<' + str(self.split_value) + ']': self.value[0].__repr__(),
                self.attribute + '[>=' + str(self.split_value) + ']': self.value[1].__repr__(),
            }

    def __str__(self):
        """
        Returns a readable string representation of the node
        """
        return str(self.__repr__())

    def pprint(self):
        """
        Pretty-prints the decision tree
        """
        return pprint.pprint(self.__repr__())
