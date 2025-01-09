"""Volume 3: Naive Bayes Classifiers."""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

class NaiveBayesFilter(ClassifierMixin):
    '''
    A Naive Bayes Classifier that sorts messages into spam or ham.
    '''
    # Problem 1
    def fit(self, X, y):
        '''
        Compute the values P(C=Ham), P(C=Spam), and P(x_i|C) to fit the model.

        Parameters:
            X (pd.Series): training data
            y (pd.Series): training labels
        '''
        
        # Count the number of samples and solve equations 14.3 and 14.4
        N_ham = y.value_counts()["ham"]
        N_spam = y.value_counts()["spam"]
        N = len(y)
        self.p_ham = N_ham / N
        self.p_spam = N_spam / N
        
        # Initialize dictionaries
        ham_counts = dict()
        spam_counts = dict()
        num_total_words_ham = 0
        num_total_words_spam = 0
        
        for sentence, label in zip(X, y):
            words = sentence.split()  # Split sentence into words
            for word in words:  # Add words to ham and spam counts
                if word not in ham_counts: ham_counts[word] = 0
                if word not in spam_counts: spam_counts[word] = 0
                if label=="ham":
                    num_total_words_ham += 1
                    ham_counts[word] += 1
                elif label=="spam":
                    num_total_words_spam += 1
                    spam_counts[word] += 1
                    
        # Convert the counts to probabilities (equation 14.5)
        self.ham_probs = {word: (count + 1)/(num_total_words_ham + 2) for word, count in ham_counts.items()}
        self.spam_probs = {word: (count + 1)/(num_total_words_spam + 2) for word, count in spam_counts.items()}
        
        return self

    # Problem 2
    def predict_proba(self, X):
        '''
        Find ln(P(C=k,x)) for each x in X and for each class.

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,2): Log probability each message is ham or spam.
                Column 0 is ham, column 1 is spam.
        '''
        # Initialize matrix with zeros
        N = len(X)
        probs = np.zeros((N, 2))
        
        for i, sentence in enumerate(X):  # Iterate through each sentence
            words = sentence.split()
            ln_p_ham = np.log(self.p_ham)  # log probability of being ham
            ln_p_spam = np.log(self.p_spam)  # log probability of being spam
            
            for word in words:   # Equation 14.6
                if word in self.ham_probs:
                    ln_p_ham += np.log(self.ham_probs[word])
                if word in self.spam_probs: 
                    ln_p_spam += np.log(self.spam_probs[word])
                if word not in self.ham_probs and word not in self.spam_probs:
                    ln_p_ham += np.log(1/2)
                    ln_p_spam += np.log(1/2)
                    
            probs[i] = np.array([ln_p_ham, ln_p_spam])
                
        return probs
        

    # Problem 3
    def predict(self, X):
        '''
        Predict the labels of each row in X, using self.predict_proba().
        The label will be a string that is either 'spam' or 'ham'.

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,): label for each message
        '''
        # Get probabilities and make predictions
        probs = self.predict_proba(X)
        argmaxes = np.argmax(probs, axis=1)
        
        # Initialize predictions as all ham and then adjust using a mask
        prediction = np.array(["ham"]*len(argmaxes)).astype(object)
        mask = argmaxes==1
        prediction[mask] = "spam"
        
        return prediction

def prob4():
    """
    Create a train-test split and use it to train a NaiveBayesFilter.
    Predict the labels of the test set.
    
    Compute and return the following two values as a tuple:
    - What proportion of the spam messages in the test set were correctly identified by the classifier?
    - What proportion of the ham messages were incorrectly identified?
    """
    # Read in the data
    df = pd.read_csv('sms_spam_collection.csv')
    X = df.Message
    y = df.Label
    
    # Randomly split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    # Train the classifier
    nb = NaiveBayesFilter()
    nb.fit(X_train, y_train)
    predictions = nb.predict(X_test)
    
    # Calculate the desired accuracies/errors
    correct_spam = 0
    total_spam = 0
    incorrect_ham = 0
    total_ham = 0
    for i, label in enumerate(y_test):
        if label=="spam":
            total_spam += 1
            if label==predictions[i]: correct_spam += 1
        if label=="ham":
            total_ham += 1
            if label!=predictions[i]: incorrect_ham += 1
        
    # Return results
    spam_acc = correct_spam / total_spam
    ham_error = incorrect_ham / total_ham
    return spam_acc, ham_error



# Problem 5
class PoissonBayesFilter(ClassifierMixin):
    '''
    A Naive Bayes Classifier that sorts messages in to spam or ham.
    This classifier assumes that words are distributed like
    Poisson random variables.
    '''
    def fit(self, X, y):
        '''
        Compute the values P(C=Ham), P(C=Spam), and r_{i,k} to fit the model.

        Parameters:
            X (pd.Series): training data
            y (pd.Series): training labels
        '''
        # Count the number of samples and solve equations 14.3 and 14.4
        N_ham = y.value_counts()["ham"]
        N_spam = y.value_counts()["spam"]
        N = len(y)
        self.p_ham = N_ham / N
        self.p_spam = N_spam / N
        self.N_ham = N_ham
        self.N_spam = N_spam
        
        # Initialize things
        ham_counts = dict()
        spam_counts = dict()
        num_total_words_ham = 0
        num_total_words_spam = 0
        
        # Count the number of occurences for each word in each class
        for sentence, label in zip(X, y):
            words = sentence.split()  # Split sentence into words
            for word in words:  # Add words to ham and spam counts
                if word not in ham_counts: ham_counts[word] = 0
                if word not in spam_counts: spam_counts[word] = 0
                if label=="ham":
                    num_total_words_ham += 1
                    ham_counts[word] += 1
                elif label=="spam":
                    num_total_words_spam += 1
                    spam_counts[word] += 1    
                    
        # Equation (14.9)
        self.ham_rates = {word: (count + 1)/(num_total_words_ham + 2) for word, count in ham_counts.items()}
        self.spam_rates = {word: (count + 1)/(num_total_words_spam + 2) for word, count in spam_counts.items()}
        
        self.n_ham_words = num_total_words_ham
        self.n_spam_words = num_total_words_spam
        
        return self

    def predict_proba(self, X):
        '''
        Find ln(P(C=k,x)) for each x in X and for each class.

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,2): Log probability each message is ham or spam.
                Column 0 is ham, column 1 is spam.
        '''
        # Initialize matrix with zeros
        N = len(X)
        probs = np.zeros((N, 2))

        
        for i, sentence in enumerate(X):  # Iterate through each sentence
            words = sentence.split()
            ln_p_ham = np.log(self.p_ham)  # log probability of being ham
            ln_p_spam = np.log(self.p_spam)  # log probability of being spam
            n = len(words)
            
            # Equation 14.10
            for word, ni in zip(*np.unique(words, return_counts=True)):

                # Calculate r values for each class
                if word in self.ham_rates: r_ham = self.ham_rates[word]
                else: r_ham = 1/(self.n_ham_words + 2)
                if word in self.spam_rates: r_spam = self.spam_rates[word]
                else: r_spam = 1/(self.n_spam_words + 2)

                # Add in the corresponding log terms
                ln_p_ham += stats.poisson.logpmf(ni, r_ham*n)
                ln_p_spam += stats.poisson.logpmf(ni, r_spam*n)

            probs[i] = np.array([ln_p_ham, ln_p_spam])

        return probs

    def predict(self, X):
        '''
        Predict the labels of each row in X, using self.predict_proba().
        The label will be a string that is either 'spam' or 'ham'.

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,): label for each message
        '''
        # Get probabilities and make predictions
        probs = self.predict_proba(X)
        argmaxes = np.argmax(probs, axis=1)
        
        # Initialize predictions as all ham and then adjust using a mask
        prediction = np.array(["ham"]*len(argmaxes)).astype(object)
        mask = argmaxes==1
        prediction[mask] = "spam"
        
        return prediction

def prob6():
    """
    Create a train-test split and use it to train a PoissonBayesFilter.
    Predict the labels of the test set.
    
    Compute and return the following two values as a tuple:
    - What proportion of the spam messages in the test set were correctly identified by the classifier?
    - What proportion of the ham messages were incorrectly identified?
    """
    # Read in the data
    df = pd.read_csv('sms_spam_collection.csv')
    X = df.Message
    y = df.Label
    
    # Randomly split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    # Train the classifier
    pb = PoissonBayesFilter()
    pb.fit(X_train, y_train)
    predictions = pb.predict(X_test)
    
    # Calculate the desired accuracies/errors
    correct_spam = 0
    total_spam = 0
    incorrect_ham = 0
    total_ham = 0
    for i, label in enumerate(y_test):
        if label=="spam":
            total_spam += 1
            if label==predictions[i]: correct_spam += 1
        if label=="ham":
            total_ham += 1
            if label!=predictions[i]: incorrect_ham += 1
        
    # Return results
    spam_acc = correct_spam / total_spam
    ham_error = incorrect_ham / total_ham
    return spam_acc, ham_error
    
# Problem 7
def sklearn_naive_bayes(X_train, y_train, X_test):
    '''
    Use sklearn's methods to transform X_train and X_test, create a
    naïve Bayes filter, and classify the provided test set.

    Parameters:
        X_train (pandas.Series): messages to train on
        y_train (pandas.Series): labels for X_train
        X_test  (pandas.Series): messages to classify

    Returns:
        (ndarray): classification of X_test
    '''
    # Transform the training data
    vectorizer = CountVectorizer()
    train_counts = vectorizer.fit_transform(X_train)
    
    # Fit multinomial Naive Bayes
    clf = MultinomialNB()
    clf = clf.fit(train_counts, y_train)
    
    # Make the predictions
    test_counts = vectorizer.transform(X_test)
    labels = clf.predict(test_counts)
    
    return labels



if __name__=="__main__":
    df = pd.read_csv('sms_spam_collection.csv')
    # separate the data into the messages and labels
    X = df.Message
    y = df.Label
    
    # nb = NaiveBayesFilter()
    # nb.fit(X[:300], y[:300])

    # Prob 1
    # print(nb.ham_probs["out"])
    # print(nb.spam_probs["out"])
    
    # Prob 2
    # print(nb.predict_proba(X[800:805]))
    
    # Prob 3
    # print(nb.predict(X[800:805]))
    
    # Prob 4
    # print(prob4())
    
    # Prob 5
    # pb = PoissonBayesFilter()
    # pb.fit(X[:300], y[:300])
    # print(pb.ham_rates["in"])
    # print(pb.spam_rates["in"])
    # print(pb.predict_proba(X[800:805]))
    # print(pb.predict(X[800:805]))
    
    # Prob 6
    print(prob6())
    
    # Prob 7
    # df = pd.read_csv('sms_spam_collection.csv')
    # X = df.Message
    # y = df.Label
    # X_train, X_test, y_train, y_test = train_test_split(X, y)
    # prediction = sklearn_naive_bayes(X_train, y_train, X_test)
    # # Calculate the accuracy
    # correct = 0
    # for i, label in enumerate(y_test):
    #     if label==prediction[i]: correct += 1
    # print(correct / len(y_test))