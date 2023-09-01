import math

from DecisonTree import Leaf, Question, DecisionNode, class_counts
from utils import *

"""
Make the imports of python packages needed
"""


class ID3:
    def __init__(self, label_names: list, min_for_pruning=0, target_attribute='diagnosis'):
        self.label_names = label_names
        self.target_attribute = target_attribute
        self.tree_root = None
        self.used_features = set()
        self.min_for_pruning = min_for_pruning

    @staticmethod
    def entropy(rows: np.ndarray, labels: np.ndarray):
        """
        Calculate the entropy of a distribution for the classes probability values.
        :param rows: array of samples
        :param labels: rows data labels.
        :return: entropy value.
        """

        counts = class_counts(rows, labels)
        impurity = 0.0

        # ====== YOUR CODE: ======
        num_of_examples = len(rows)
        for label in counts.keys():
            prob_l = counts[label] / num_of_examples
            impurity -= prob_l * math.log2(prob_l)
        # ========================

        return impurity

    def info_gain(self, left, left_labels, right, right_labels, current_uncertainty):
        """
        Calculate the information gain, as the uncertainty of the starting node, minus the weighted impurity of
        two child nodes.
        :param left: the left child rows.
        :param left_labels: the left child labels.
        :param right: the right child rows.
        :param right_labels: the right child labels.
        :param current_uncertainty: the current uncertainty of the current node
        :return: the info gain for splitting the current node into the two children left and right.
        """
        assert (len(left) == len(left_labels)) and (len(right) == len(right_labels)), \
            'The split of current node is not right, rows size should be equal to labels size.'

        info_gain_value = 0.0
        # ====== YOUR CODE: ======
        info_gain_value = current_uncertainty
        left_fraction = len(left) / (len(left) + len(right))
        right_fraction = len(right) / (len(left) + len(right))
        info_gain_value -= (left_fraction * self.entropy(left, left_labels))
        info_gain_value -= (right_fraction * self.entropy(right, right_labels))
        # ========================

        return info_gain_value

    def partition(self, rows, labels, question: Question, current_uncertainty):
        """
        Partitions the rows by the question.
        :param rows: array of samples
        :param labels: rows data labels.
        :param question: an instance of the Question which we will use to partition the data.
        :param current_uncertainty: the current uncertainty of the current node
        :return: Tuple of (gain, true_rows, true_labels, false_rows, false_labels)
        """

        gain, true_rows, true_labels, false_rows, false_labels = None, None, None, None, None
        assert len(rows) == len(labels), 'Rows size should be equal to labels size.'

        # ====== YOUR CODE: ======
        true_rows, true_labels, false_rows, false_labels = [], [], [], []

        for idx, row in enumerate(rows):
            if question.match(row):
                true_rows.append(row)
                true_labels.append(labels[idx])
            else:
                false_rows.append(row)
                false_labels.append(labels[idx])
        gain = self.info_gain(true_rows, true_labels, false_rows, false_labels, current_uncertainty)

        # ========================

        return gain, true_rows, true_labels, false_rows, false_labels

    def find_best_split(self, rows, labels):
        """
        Find the best question to ask by iterating over every feature / value and calculating the information gain.
        :param rows: array of samples
        :param labels: rows data labels.
        :return: Tuple of (best_gain, best_question, best_true_rows, best_true_labels, best_false_rows, best_false_labels)
        """
        best_gain = - math.inf  # keep track of the best information gain
        best_question = None  # keep train of the feature / value that produced it
        best_false_rows, best_false_labels = None, None
        best_true_rows, best_true_labels = None, None
        current_uncertainty = self.entropy(rows, labels)
        # ====== YOUR CODE: ======
        for idx, feature in enumerate(self.label_names[1:]):
            feature_values = sorted(list(set([row[idx] for row in rows])))
            for value_idx in range(len(feature_values) - 1):
                question = Question(feature, idx, (feature_values[value_idx] + feature_values[value_idx + 1])/2)
                curr_values = self.partition(rows, labels, question, current_uncertainty)
                if best_gain <= curr_values[0]:
                    best_gain, best_true_rows, best_true_labels, best_false_rows, best_false_labels = curr_values
                    best_question = question
        # ========================

        return best_gain, best_question, best_true_rows, best_true_labels, best_false_rows, best_false_labels

    def build_tree(self, rows, labels):
        """
        Build the decision Tree in recursion.
        :param rows: array of samples
        :param labels: rows data labels.
        :return: a Question node, This records the best feature / value to ask at this point, depending on the answer.
                or leaf if we have to prune this branch (in which cases ?)

        """
        best_question = None
        true_branch, false_branch = None, None

        # ====== YOUR CODE: ======
        if not self.entropy(rows, labels):
            return Leaf(rows, labels)
        if len(labels) <= self.min_for_pruning:
            return Leaf(rows, labels)
        values = self.find_best_split(rows, labels)
        best_question = values[1]
        true_branch = self.build_tree(values[2], values[3])
        false_branch = self.build_tree(values[4], values[5])
        # ========================

        return DecisionNode(best_question, true_branch, false_branch)

    def fit(self, x_train, y_train):
        """
        Trains the ID3 model. By building the tree.
        :param x_train: A labeled training data.
        :param y_train: training data labels.
        """

        # ====== YOUR CODE: ======
        self.tree_root = self.build_tree(x_train, y_train)
        # ========================

    def predict_sample(self, row, node: DecisionNode or Leaf = None):
        """
        Predict the most likely class for single sample in subtree of the given node.
        :param row: vector of shape (1,D).
        :return: The row prediction.
        """

        if node is None:
            node = self.tree_root
        prediction = None

        # ====== YOUR CODE: ======
        if type(node) is Leaf:
            max_value, max_key = -math.inf, None
            for key, value in node.predictions.items():
                if max_value < value:
                    max_value = value
                    max_key = key
            return max_key

        if node.question.match(row):
            prediction = self.predict_sample(row, node.true_branch)
        else:
            prediction = self.predict_sample(row, node.false_branch)
        # ========================

        return prediction

    def predict(self, rows):
        """
        Predict the most likely class for each sample in a given vector.
        :param rows: vector of shape (N,D) where N is the number of samples.
        :return: A vector of shape (N,) containing the predicted classes.
        """

        y_pred = None

        # ====== YOUR CODE: ======
        predictions = [self.predict_sample(row) for row in rows]
        y_pred = np.array(predictions)
        # ========================

        return y_pred
