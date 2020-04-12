import pandas as pd
from math import log2, factorial, floor

df = pd.DataFrame({
    'Outlook': ['sunny', 'sunny', 'overcast', 'rain', 'rain', 'rain', 'overcast', 'sunny', 'sunny', 'rain', 'sunny',
                'overcast', 'overcast', 'rain'],
    'Temperature': ['hot', 'hot', 'hot', 'mild', 'cool', 'cool', 'cool', 'mild', 'cool', 'mild', 'mild', 'mild', 'hot',
                    'mild'],
    'Humidity': ['high', 'high', 'high', 'high', 'normal', 'normal', 'normal', 'high', 'normal', 'normal', 'normal',
                 'high', 'normal', 'high'],
    'Windy': ['false', 'true', 'false', 'false', 'false', 'true', 'true', 'false', 'false', 'false', 'true', 'true',
              'false', 'true'],
    'Class': ['N', 'N', 'P', 'P', 'P', 'N', 'P', 'N', 'P', 'P', 'P', 'P', 'P', 'N']
})

df2 = pd.DataFrame({
    'Windy': ['false', 'true', 'false', 'false', 'false', 'true', 'true', 'false', 'false', 'false', 'true', 'true',
              'false', 'true'],
    'Class': ['N', 'P', 'N', 'N', 'N', 'P', 'P', 'N', 'N', 'N', 'P', 'P', 'N', 'P']
})


def comb(n, k):
    return factorial(n) / (factorial(n - k) * factorial(k))


class DecisionTree:
    def __init__(self,
                 data: pd.DataFrame,
                 attr: str = None,
                 values: dict = None,
                 attributes: dict = None,
                 classes: list = None,
                 class_attr: str = 'Class',
                 parent=None) -> None:
        self._data = data
        self._attr = attr
        self._values = values
        self._class_attr = class_attr
        self._attributes = attributes
        self._classes = classes
        self._parent = parent

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
        return DecisionTree(
            self._data.copy(),
            self._attr,
            self._values.copy(),
            self._attributes.copy(),
            self._classes.copy(),
            self._class_attr
        )

    def print(self, include_cost = False):
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
                _lines = _v.print(include_cost)
                for _line in _lines.split('\n'):
                    if len(_line) > 0:
                        _res += '\t' + _line + '\n'
            return _res

    def __getitem__(self, item):
        return self._values[item]

    def tree_cost(self):
        if self._values is None:
            return 1 + log2(len(self._classes))
        else:
            return 1 + log2(len(self._attributes)) + sum([dt.tree_cost() for dt in self._values.values()])

    def exception_cost(self):
        if self._values is None:
            # Get data length n and list of classes
            n = len(self._data)
            c = self._classes

            # If the class is binary, calculate and return the L(n,k,b) from the paper
            if len(c) <= 2:
                b = floor(n / 2)
                k = len(self._data[self._data[self._class_attr] == c[0]])
                return log2(b + 1) + log2(comb(n, k))

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
                    cost *= comb(n, k_i)
                    k += k_i

                # Multiply the cost by the (n + k - 1) over (k - 1) combinations
                cost *= comb(n + k - 1, k - 1)
                return log2(cost)
        else:
            return sum([dt.exception_cost() for dt in self._values.values()])

    def cost(self):
        return self.tree_cost() + self.exception_cost()

    def construct(self):
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
                    # Construct subtree
                    tree[attr][v] = DecisionTree(new_data,
                                                 attributes=new_attributes,
                                                 classes=self._classes,
                                                 class_attr=self._class_attr,
                                                 parent=self)

                score[attr] = sum([dt.cost() for dt in tree[attr].values()])

            split_attr = sorted(score.items(), key=lambda item: item[1])[0][0]
            self._attr = split_attr
            self._values = tree[split_attr]
            for k, v in self._values.items():
                self._values[k] = v.construct()
            return self

    def is_leaf(self):
        return self._values is None

    def is_decision_node(self):
        if self.is_leaf():
            return False
        else:
            return all([dt.is_leaf() for dt in self._values.values()])

    def get_decision_nodes(self):
        if self.is_decision_node():
            return [self]
        elif self.is_leaf():
            return []
        else:
            return [n for v in self._values.values() for n in v.get_decision_nodes() if not v.is_leaf()]

    @staticmethod
    def mdl(data):
        return DecisionTree(data).construct().prune_tree()

    def prune(self):
        if not self.is_decision_node():
            raise AssertionError("try_prune() can only be invoked on decision nodes")

        self._attr = self._data[self._class_attr].mode()[0]
        self._values = None
        return self

    def try_prune(self):
        if not self.is_decision_node():
            raise AssertionError("try_prune() can only be invoked on decision nodes")

        pruned = self.__copy__().prune()

        if pruned.cost() < self.cost():
            self.prune()
            return True

        return False

    def prune_tree(self):
        c = True
        while c:
            r = [False]
            nodes = self.get_decision_nodes()
            for n in nodes:
                r.append(n.try_prune())

            c = any(r)
        return self


if __name__ == '__main__':
    x = DecisionTree(df).construct()
    print(x.print(True))
