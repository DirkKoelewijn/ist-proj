import numbers
import pprint


class Node:
    """
    Class to model an decision tree nodes
    """

    def __init__(self, attribute=None, split_value=None, value=None) -> None:
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
                self.attribute + '[not ' + self.split_value + ']': self.value[0].__repr__(),
            }
        else:
            return {
                self.attribute + '[<' + str(self.split_value) + ']': self.value[0].__repr__(),
                self.attribute + '[>=' + str(self.split_value) + ']': self.value[0].__repr__(),
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
