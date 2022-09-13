'''naive_bayes_multinomial.py
Naive Bayes classifier with Multinomial likelihood for discrete features
Kelly Putnam
CS 251 Data Analysis Visualization, Spring 2021
'''
import numpy as np


class NaiveBayes:
    '''Naive Bayes classifier using Multinomial likeilihoods (discrete data belonging to any
     number of classes)'''
    def __init__(self, num_classes):
        '''Naive Bayes constructor

        TODO:
        - Add instance variable for `num_classes`
        '''
        # class_priors: ndarray. shape=(num_classes,).
        #   Probability that a training example belongs to each of the classes
        #   For spam filter: prob training example is spam or ham
        self.class_priors = None
        # class_likelihoods: ndarray. shape=(num_classes, num_features).
        #   Probability that each word appears within class c
        self.class_likelihoods = None
        self.num_classes = num_classes




    def train(self, data, y):
        '''Train the Naive Bayes classifier so that it records the "statistics" of the training set:
        class priors (i.e. how likely an email is in the training set to be spam or ham?) and the
        class likelihoods (the probability of a word appearing in each class — spam or ham)

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.
        y: ndarray. shape=(num_samps,). Corresponding class of each data sample.

        TODO:
        - Compute the instance variables self.class_priors and self.class_likelihoods needed for
        Bayes Rule. See equations in notebook.
        '''
        num_samps, num_features = data.shape

        priors = []
        likelihoods = []

        for i in range(self.num_classes):

            nonzeros = np.count_nonzero(y==i, axis = 0)
            count = nonzeros/num_samps
            priors.append(count)

            individual_likelihood = []

            for feat in range(num_features):
                inds = np.argwhere(y==i)
                ncw = np.sum(data[inds,:])
                nc = np.sum(data[inds, feat])
                lcw = (nc + 1)/(ncw + num_features)

                individual_likelihood.append(lcw)

            likelihoods.append(individual_likelihood)


        self.class_priors = np.array(priors)
        self.class_likelihoods = np.array(likelihoods)










    def predict(self, data):
        '''Combine the class likelihoods and priors to compute the posterior distribution. The
        predicted class for a test sample from `data` is the class that yields the highest posterior
        probability.

        Parameters:
        -----------
        data: ndarray. shape=(num_test_samps, num_features). Data to predict the class of
            Need not be the data used to train the network

        Returns:
        -----------
        ndarray of nonnegative ints. shape=(num_samps,). Predicted class of each test data sample.

        TODO:
        - For the test samples, we want to compute the log of the posterior by evaluating
        the the log of the right-hand side of Bayes Rule without the denominator (see notebook for
        equation). This can be done without loops using matrix multiplication or with a loop and
        a series of dot products.
        - Predict the class of each test sample according to the class that produces the largest
        log(posterior) probability (hint: can be done without loops).

        NOTE: Remember that you are computing the LOG of the posterior (see notebook for equation).
        NOTE: The argmax function could be useful here.
        '''
        num_test_samps, num_features = data.shape

        logpc = np.log(self.class_priors)
        summation = np.dot(data, np.log(self.class_likelihoods.T))

        logpost = logpc + summation
        predictions = logpost.argmax(axis = 1)

        return predictions











    def accuracy(self, y, y_pred):
        '''Computes accuracy based on percent correct: Proportion of predicted class labels `y_pred`
        that match the true values `y`.

        Parameters:
        -----------
        y: ndarray. shape=(num_data_sams,)
            Ground-truth, known class labels for each data sample
        y_pred: ndarray. shape=(num_data_sams,)
            Predicted class labels by the model for each data sample

        Returns:
        -----------
        float. Between 0 and 1. Proportion correct classification.

        NOTE: Can be done without any loops
        '''
        difference = y_pred - y
        non0s = np.count_nonzero(difference)
        resids = (y.shape[0]-non0s)
        accuracy = resids/(y.shape[0])
        return accuracy









    def confusion_matrix(self, y, y_pred):
        '''Create a confusion matrix based on the ground truth class labels (`y`) and those predicted
        by the classifier (`y_pred`).

        Recall: the rows represent the "actual" ground truth labels, the columns represent the
        predicted labels.

        Parameters:
        -----------
        y: ndarray. shape=(num_data_samps,)
            Ground-truth, known class labels for each data sample
        y_pred: ndarray. shape=(num_data_samps,)
            Predicted class labels by the model for each data sample

        Returns:
        -----------
        ndarray. shape=(num_classes, num_classes).
            Confusion matrix
        '''
        # To get the number of classes, you can use the np.unique
        # function to identify the number of unique categories in the
        # y matrix.


        confusionmatrix = np.zeros(shape = (self.num_classes, self.num_classes))

        for i in range(len(y)):
            r = np.int(y[i])
            c = np.int(y_pred[i])
            confusionmatrix[r,c] += 1
        
        return confusionmatrix

