"""
Random Forest Lab

Daniel Perkins
MATH 403 (001)
10/6/24
"""
from platform import uname
import graphviz
import os
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from uuid import uuid4
import random
from time import time

# Problem 1
class Question:
    """Questions to use in construction and display of Decision Trees.
    Attributes:
        column (int): which column of the data this question asks
        value (int/float): value the question asks about
        features (str): name of the feature asked about
    Methods:
        match: returns boolean of if a given sample answered T/F"""

    def __init__(self, column, value, feature_names):
        self.column = column
        self.value = value
        self.features = feature_names[self.column]

    def match(self, sample):
        """Returns T/F depending on how the sample answers the question
        Parameters:
            sample ((n,), ndarray): New sample to classify
        Returns:
            (bool): How the sample compares to the question"""
        return sample[self.column] >= self.value

    def __repr__(self):
        return "Is %s >= %s?" % (self.features, str(float(self.value)))

def partition(data, question):
    """Splits the data into left (true) and right (false)
    Parameters:
        data ((m,n), ndarray): data to partition
        question (Question): question to split on
    Returns:
        left ((j,n), ndarray): Portion of the data matching the question
        right ((m-j, n), ndarray): Portion of the data NOT matching the question
    """
    left, right = [], []
    for datum in data:  # Split up data
        if question.match(datum): left.append(datum)
        else: right.append(datum)
    # Convert to correct shape and return it
    return np.array(left).reshape(-1, len(datum)), np.array(right).reshape(-1, len(datum))


# Helper function
def num_rows(array):
    """ Returns the number of rows in a given array """
    if array is None:
        return 0
    elif len(array.shape) == 1:
        return 1
    else:
        return array.shape[0]

# Helper function
def class_counts(data):
    """ Returns a dictionary with the number of samples under each class label
        formatted {label : number_of_samples} """
    if len(data.shape) == 1: # If there's only one row
        return {data[-1]: 1}
    
    # Create a dictionary  with unique values as keys and the counts as values
    unique, counts = np.unique(data[:, -1], return_counts=True)
    return dict(zip(unique, counts))


# Helper function
def info_gain(data, left, right):
    """Return the info gain of a partition of data.
    Parameters:
        data (ndarray): the unsplit data
        left (ndarray): left split of data
        right (ndarray): right split of data
    Returns:
        (float): info gain of the data"""
        
    def gini(data):
        """Return the Gini impurity of given array of data.
        Parameters:
            data (ndarray): data to examine
        Returns:
            (float): Gini impurity of the data"""
        counts = class_counts(data)
        N = num_rows(data)
        impurity = 1
        for lbl in counts:
            prob_of_lbl = counts[lbl] / N
            impurity -= prob_of_lbl**2
        return impurity
        
    p = num_rows(right)/(num_rows(left)+num_rows(right))
    return gini(data) - p*gini(right)-(1-p)*gini(left)

# Problem 2, Problem 6
def find_best_split(data, feature_names, min_samples_leaf=5, random_subset=False):
    """Find the optimal split
    Parameters:
        data (ndarray): Data in question
        feature_names (list of strings): Labels for each column of data
        min_samples_leaf (int): minimum number of samples per leaf
        random_subset (bool): for Problem 6
    Returns:
        (float): Best info gain
        (Question): Best question"""
    best_gain = 0
    best_question = None

    # Get the indices of columns where values differ
    # cols_with_diff = np.any(data != data[0, :], axis=0)
    # indices = np.where(cols_with_diff)[0].tolist()[:-1]
    indices = np.arange(len(feature_names) - 1)
    
    for i in indices:  # For each column
        unique_vals = set(data[:, i])  # Set of unique values
        for val in unique_vals:    # For each row value
            question = Question(column=i, value=val, feature_names=features)
            left, right = partition(animals, question)
            if(len(left) < min_samples_leaf or len(right) < min_samples_leaf): 
                continue   # Go to next leaf if the leaf is too small
            gain = info_gain(data, left, right)
            if gain > best_gain:  # If the gain is higher than previous best
                best_gain = gain
                best_question = question
    if best_question is None: return None
    else: return best_gain, best_question

# Problem 3
class Leaf:
    """Tree leaf node
    Attribute:
        prediction (dict): Dictionary of labels at the leaf"""
    def __init__(self, data):
        self.prediction = class_counts(data)

class Decision_Node:
    """Tree node with a question
    Attributes:
        question (Question): Question associated with node
        left (Decision_Node or Leaf): child branch
        right (Decision_Node or Leaf): child branch"""
    def __init__(self, question, left_branch, right_branch):
        self.question = question
        self.left = left_branch
        self.right = right_branch


# Prolem 4
def build_tree(data, feature_names, min_samples_leaf=5, max_depth=4, current_depth=0, random_subset=False):
    """Build a classification tree using the classes Decision_Node and Leaf
    Parameters:
        data (ndarray)
        feature_names(list or array)
        min_samples_leaf (int): minimum allowed number of samples per leaf
        max_depth (int): maximum allowed depth
        current_depth (int): depth counter
        random_subset (bool): whether or not to train on a random subset of features
    Returns:
        Decision_Node (or Leaf)"""
    raise NotImplementedError('Problem 4 Incomplete')

# Problem 5
def predict_tree(sample, my_tree):
    """Predict the label for a sample given a pre-made decision tree
    Parameters:
        sample (ndarray): a single sample
        my_tree (Decision_Node or Leaf): a decision tree
    Returns:
        Label to be assigned to new sample"""
    raise NotImplementedError('Problem 5 Incomplete')

def analyze_tree(dataset, my_tree):
    """Test how accurately a tree classifies a dataset
    Parameters:
        dataset (ndarray): Labeled data with the labels in the last column
        tree (Decision_Node or Leaf): a decision tree
    Returns:
        (float): Proportion of dataset classified correctly"""
    raise NotImplementedError('Problem 6 Incomplete')

# Problem 6
def predict_forest(sample, forest):
    """Predict the label for a new sample, given a random forest
    Parameters:
        sample (ndarray): a single sample
        forest (list): a list of decision trees
    Returns:
        Label to be assigned to new sample"""
    raise NotImplementedError('Problem 6 Incomplete')

def analyze_forest(dataset, forest):
    """Test how accurately a forest classifies a dataset
    Parameters:
        dataset (ndarray): Labeled data with the labels in the last column
        forest (list): list of decision trees
    Returns:
        (float): Proportion of dataset classified correctly"""
    raise NotImplementedError('Problem 6 Incomplete')

# Problem 7
def prob7():
    """ Using the file parkinsons.csv, return three tuples of floats. For tuples 1 and 2,
        randomly select 130 samples; use 100 for training and 30 for testing.
        For tuple 3, use the entire dataset with an 80-20 train-test split.
        Tuple 1:
            a) Your accuracy in a 5-tree forest with min_samples_leaf=15
                and max_depth=4
            b) The time it took to run your 5-tree forest
        Tuple 2:
            a) Scikit-Learn's accuracy in a 5-tree forest with
                min_samples_leaf=15 and max_depth=4
            b) The time it took to run that 5-tree forest
        Tuple 3:
            a) Scikit-Learn's accuracy in a forest with default parameters
            b) The time it took to run that forest with default parameters
    """
    raise NotImplementedError('Problem 7 Incomplete')


## Code to draw a tree
def draw_node(graph, my_tree):
    """Helper function for drawTree"""
    node_id = uuid4().hex
    #If it's a leaf, draw an oval and label with the prediction
    if hasattr(my_tree, "prediction"):
        graph.node(node_id, shape="oval", label="%s" % my_tree.prediction)
        return node_id
    else: # If it's not a leaf, make a question box
        graph.node(node_id, shape="box", label="%s" % my_tree.question)
        left_id = draw_node(graph, my_tree.left)
        graph.edge(node_id, left_id, label="T")
        right_id = draw_node(graph, my_tree.right)
        graph.edge(node_id, right_id, label="F")
        return node_id

def draw_tree(my_tree, filename='Digraph'):
    """Draws a tree"""
    # Remove the files if they already exist
    for file in [f'{filename}.gv', f'{filename}.gv.pdf']:
        if os.path.exists(file):
            os.remove(file)
    graph = graphviz.Digraph(comment="Decision Tree")
    draw_node(graph, my_tree)
    # graph.render(view=True) #This saves Digraph.gv and Digraph.gv.pdf
    in_wsl = False
    in_wsl = 'microsoft-standard' in uname().release
    if in_wsl:
        graph.render(f'{filename}.gv', view=False)
        os.system(f'cmd.exe /C start {filename}.gv.pdf')
    else:
        graph.render(view=True)


if __name__=="__main__":
    # Load in the data
    animals = np.loadtxt("animals.csv", delimiter=",")
    # Load in feature names
    features = np.loadtxt("animal_features.csv", delimiter=",", dtype=str, comments=None)
    # Load in sample names
    names = np.loadtxt('animal_names.csv', delimiter=",", dtype=str)
    
    # Prob 1
    # question = Question(column=1, value=3, feature_names=features)
    # left, right = partition(animals, question)
    # print(len(left), (len(right)))
    
    # question = Question(column=1, value=75, feature_names=features)
    # left, right = partition(animals, question)
    # print(len(left), (len(right)))
    
    # Prob 2
    # print(find_best_split(animals, features))
    
    # Prob 3
    leaf = Leaf(animals)