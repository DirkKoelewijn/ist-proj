import pandas as pd
from math import log2, factorial, floor


class MDLPTree:
    """
    Class to model and construct MDLP Trees
    """
    def __init__(self,
                 data: pd.DataFrame,
                 attr: str = None,
                 values: dict = None,
                 attributes: dict = None,
                 classes: list = None,
                 class_attr: str = 'Class') -> None:
        self._data = data
        self._attr = attr
        self._values = values
        self._class_attr = class_attr
        self._attributes = attributes
        self._classes = classes

        if self._attr is None:
            self._attr = data[class_attr].mode()[0]

        if self._attributes is None:
            self._attributes = dict([(a, list(set(data[a]))) for a in list(data.columns) if a != class_attr])

        if self._classes is None:
            self._classes = list(set(data[class_attr]))

    def __repr__(self):
        return str(self)

    def __str__(self):
        if self._values is None:
            return self._attr
        else:
            _res = '{'
            for _k, _v in self._values.items():
                _res += self._attr + '[' + _k + '] : '
                _lines = _v.__repr__()
                for _line in _lines.split('\n'):
                    if len(_line) > 0:
                        _res += _line + ', '
            return _res[:-2] + '}'

    def __copy__(self):
        return MDLPTree(
            self._data.copy(),
            self._attr,
            self._values.copy(),
            self._attributes.copy(),
            self._classes.copy(),
            self._class_attr
        )

    def __getitem__(self, item):
        return self._values[item]

    def is_leaf(self):
        """
        Returns whether this node is a leaf
        :return:
        """
        return self._values is None

    def is_decision_node(self):
        """
        Returns whether this node is a decision node
        """
        return not self.is_leaf() and all([dt.is_leaf() for dt in self._values.values()])

    def decision_nodes(self):
        """
        Returns the decision nodes of the tree
        """
        if self.is_decision_node():
            return [self]
        elif self.is_leaf():
            return []
        else:
            return [n for v in self._values.values() for n in v.decision_nodes() if not v.is_leaf()]

    def node_count(self):
        """
        Returns the amount of nodes in a tree
        """
        if self.is_leaf():
            return 0
        else:
            return 1 + sum([n.node_count() for n in self._values.values()])

    def leaf_count(self):
        """
        Returns the amount of leafs in the tree
        """
        if self.is_leaf():
            return 1
        else:
            return sum([n.node_count() for n in self._values.values()])

    @staticmethod
    def __comb(n, k):
        """
        Calculates the combinations n over k
        """
        return factorial(n) / (factorial(n - k) * factorial(k))

    def __tree_cost(self):
        """
        Returns the pure tree cost (without exceptions)
        """
        if self._values is None:
            return 1 + log2(len(self._classes))
        else:
            return 1 + log2(len(self._attributes)) + sum([dt.__tree_cost() for dt in self._values.values()])

    def __exception_cost(self):
        """
        Returns the cost of purely the exceptions
        """
        if self._values is None:
            # Get data length n and list of classes
            n = len(self._data)
            c = self._classes

            # If the class is binary, calculate and return the L(n,k,b) from the paper
            if len(c) <= 2:
                b = floor(n / 2)
                k = len(self._data[self._data[self._class_attr] == c[0]])
                return log2(b + 1) + log2(self.__comb(n, k))

            # Else, calculate and return the L(n; k(1), k(2), ..., k(t))
            else:
                # Find the most frequent class and remove from c
                c_max = self._data[self._class_attr].mode()[0]
                c.remove(c_max)

                # Initialize cost and k with identity values
                cost = 1
                k = 0

                # Find all n over k(i) combinations
                for c_i in c:
                    k_i = len(self._data[self._data[self._class_attr] == c_i])
                    cost *= self.__comb(n, k_i)
                    k += k_i

                # Multiply the cost by the (n + k - 1) over (k - 1) combinations
                cost *= self.__comb(n + k - 1, k - 1)
                return log2(cost)
        else:
            return sum([dt.__exception_cost() for dt in self._values.values()])

    def cost(self):
        """
        Returns the total cost of the tree
        """
        return self.__tree_cost() + self.__exception_cost()

    def build(self):
        """
        Constructs and prunes the three
        """
        return self.construct().prune()

    def construct(self):
        """
        Only constructs the tree, without pruning it
        """
        if len(self._attributes) == 0 or len(set(self._data[self._class_attr])) <= 1:
            return self
        else:
            score = {}
            tree = {}
            for attr, values in self._attributes.items():
                tree[attr] = {}
                # Set new attributes
                new_attributes = self._attributes.copy()
                del new_attributes[attr]

                for v in values:
                    new_data = self._data[self._data[attr] == v]
                    if len(new_data) == 0:
                        new_data = self._data.copy()
                    # Create new leafs
                    tree[attr][v] = MDLPTree(new_data, attributes=new_attributes, classes=self._classes,
                                             class_attr=self._class_attr)

                score[attr] = sum([dt.cost() for dt in tree[attr].values()])

            split_attr = sorted(score.items(), key=lambda item: item[1])[0][0]
            self._attr = split_attr
            self._values = tree[split_attr]
            for k, v in self._values.items():
                self._values[k] = v.construct()
            return self

    def __force_prune(self):
        """
        Forces pruning of this nodes, regardless of costs or it being a decision node or not
        """
        self._attr = self._data[self._class_attr].mode()[0]
        self._values = None
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
            self._attr = self._data[self._class_attr].mode()[0]
            self._values = None
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

    def printable(self, include_cost=False):
        """
        Returns a printable version of the tree
        """
        if self._values is None:
            if include_cost:
                return self._attr + ' (' + str(round(self.cost(), 3)) + ')\n'
            return self._attr + '\n'
        else:
            _res = ''
            if include_cost:
                _res = '(' + str(round(self.cost(), 3)) + ')\n'

            for _k, _v in self._values.items():
                _res += self._attr + ' - ' + _k + '\n'
                _lines = _v.printable(include_cost)
                for _line in _lines.split('\n'):
                    if len(_line) > 0:
                        _res += '\t' + _line + '\n'
            return _res


if __name__ == '__main__':
    print('')
