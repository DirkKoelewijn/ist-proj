import pprint


class Node:
    """
    Class to model an decision tree nodes
    """

    def __init__(self, attribute=None, cut_value=None, value=None) -> None:
        """
        Initializes a node (or leaf)

        :param attribute: The attribute that this node splits on (node only)
        :param cut_value: The cut value for the attribute (nodes with real-value attributes only)
        :param value: The class (leafs), a dict mapping from attribute value to nodes (nodes with discrete attributes)
        or a tuple of (node < cut_value, node >= cut_value)
        """
        self.attribute = attribute
        self.cut_value = cut_value
        self.value = value

    def is_leaf(self):
        """
        Returns whether the node is a leaf
        """
        return self.attribute is None

    def has_discrete_attribute(self):
        """
        Returns whether the node has a discrete attribute
        """
        return not self.is_leaf() and self.cut_value is None

    def has_continuous_attribute(self):
        """
        Returns whether the node has a continuous attribute
        """
        return not self.is_leaf() and self.cut_value is not None

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
        if self.has_discrete_attribute():
            return self.value.values()
        else:
            return self.value

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

    def __getitem__(self, item):
        """
        Get's the sub node by the attribute value (does not work for leafs)
        :param item: attribute value
        :return: Corresponding sub-nodes
        """
        assert not self.is_leaf(), 'Cannot get item from leaf'

        if self.has_discrete_attribute():
            return self.value[item]
        else:
            if item < self.cut_value:
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
            d = self.value
        else:
            d = {'<' + str(self.cut_value): self.value[0], '>=' + str(self.cut_value): self.value[1]}

        res = {}
        for k, v in d.items():
            res[self.attribute + ' ' + k] = v.__repr__()
        return res

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
