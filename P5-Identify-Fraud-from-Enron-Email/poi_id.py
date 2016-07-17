
# coding: utf-8

# # P5: Identify Fraud from Enron Email
# 
# [Enron](https://en.wikipedia.org/wiki/Enron) was an American energy, commodities, and services company based in Houston, Texas. At the end of 2001, it was revealed that it's reported financial condition was sustained by an institutionalized, systematic, and creatively planned accounting fraud, known since as the Enron scandal. Enron has since become a well-known example of willful corporate fraud and corruption.
# 
# This project will attempt to predict whether a person was culpable in the fraud by sorting everyone into two classes (POI and Not POI). A modified version of the [Enron Email Dataset](https://www.cs.cmu.edu/~./enron/) is used for training multiple classifiers.

# In[20]:

get_ipython().magic(u'matplotlib inline')
### Import Libraries

### import file libraries
import sys
import pickle
sys.path.append("../tools/")

### import utility libraries
from time import time
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint

### import provided libraries
from feature_format import featureFormat, targetFeatureSplit
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
from tester import dump_classifier_and_data, test_classifier

### import pipeline libraries
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV

### import classification libraries
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

### import evaluation libraries
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer
from sklearn.metrics import confusion_matrix


# ### Feature Exploration

# In[21]:

### Feature Exploration

### initalize features_list
features_list = ['poi']
financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                      'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                      'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']
email_features = ['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
                  'shared_receipt_with_poi']

### features_list with all features, PCA will be used later to reduce dimensionality
features_list = features_list + financial_features + email_features

### load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### format features
data = featureFormat(data_dict, features_list, sort_keys = True)

### exploring the dataset
def explore_data(data_dict):
    """
    function for exploring the big picture of the enron data
    """
    print "*"*100
    print "Exploring the Dataset"
    print "Number of people in dataset:", len(data_dict)
    print "Number of features per person:", len(data_dict['SKILLING JEFFREY K'])
    print "Number of POIs in dataset:", sum(1 for k in data_dict.iterkeys() if data_dict[k]['poi'] == 1)
    for i in financial_features:
        print "Percent of POIs that have NaN for {0}:".format(i), round(float(sum(
        1 for k in data_dict.iterkeys() if data_dict[k][i] == "NaN" and data_dict[k]['poi'] == 1)) /
        sum(1 for k in data_dict.iterkeys() if data_dict[k]['poi'] == 1), 2)
    
    for i in email_features:
        print "Percent of POIs that have NaN for {0}:".format(i), round(float(sum(
        1 for k in data_dict.iterkeys() if data_dict[k][i] == "NaN" and data_dict[k]['poi'] == 1)) /
        sum(1 for k in data_dict.iterkeys() if data_dict[k]['poi'] == 1), 2)
    print "*"*100
        
explore_data(data_dict)


# Right off the bat, it is clear that this isn't the most robust set of data for our goal. Of the 146 people in the dataset, only 18 are POI. We will have to keep this in mind when forming our classifiers.
# 
# loan_advances and director_fees stand out as having a high percentage of POI with NaN values. This could be useful information, but I will err on the side of conservatism and remove these features from the list.

# In[22]:

def feature_add(feature, remove=False):
    """
    function to add or remove a feature to features_list
    """
    if remove == False:
        features_list.append(feature)
    if remove == True:
        features_list.remove(feature)
    

feature_add('loan_advances', remove=True)
feature_add('director_fees', remove=True)
data = featureFormat(data_dict, features_list, sort_keys = True)

print "Revised features list:", features_list


# ### Outlier Exploration

# In[23]:

### Outlier Exploration
def explore_outlier(feature1, feature2, features_list, data):
    """
    visualize the relationship between features
    """

    for point in data:
        var1 = point[features_list.index(feature1)]
        var2 = point[features_list.index(feature2)]
        plt.scatter( var1, var2 )
    
    print "*"*100
    print "{} VS {}".format(feature1, feature2)
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.show()
    print "*"*100

explore_outlier('salary', 'bonus', features_list, data)


# An obvious outlier can be seen in this plot. Under further exploration, it appears 'TOTAL' is being treated as an individual person in this dataset. This is a quark of spreadsheets and can be removed.

# In[24]:

### Remove Outlier
def remove_outlier(outlier):
    """
    removes outliers from data
    """

    data_dict.pop(outlier)
    print "Removing outlier '{0}' from the data set".format(outlier)

remove_outlier('TOTAL')
data = featureFormat(data_dict, features_list, sort_keys = True)
explore_outlier('salary', 'bonus', features_list, data)


# This looks much better. The remaining outliers seem to be cases of accurate information and will be left in the dataset. Let's proceed with our outlier exploration.

# In[25]:

### Countine Outlier Exploration
from itertools import tee, izip
def pairwise(iterable):
    """
    s -> (s0,s1), (s1,s2), (s2, s3), ...
    """
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)

for f1, f2 in pairwise(features_list[1:]):
    explore_outlier(f1, f2, features_list, data)


# There still are some extreme data points in the plots. We will now try to isolate the largest value for every feature to analyze whether it truly is an outlier.

# In[26]:

### Isolate outlier values
def isolate_outliers(features_list):
    """
    isolate the largest values in the features
    """
    print "*"*100
    print "Isolating Outliers"
    for i in features_list:
        largest = 0
        person = ""
        for k in data_dict.iterkeys():
            if data_dict[k][i] > largest and data_dict[k][i] != 'NaN':
                largest = data_dict[k][i]
                person = str(k)
    
        print "{0} had the largest {1} of:".format(person, i), largest
    print "*"*100
        
isolate_outliers(features_list[1:])


# This returns a who's who of Enron’s higher ups. With a quick Google search, a face can be put to a lot of these names. For example, John Lavarto's eight million dollar bonus was mainly the result of mark to market profits in the energy trading unit of Enron. Timothy Belden, who had the most shared receipts with POI, was the head of trading at Enron Energy Services and the mastermind of a scheme to drive up California's energy prices.
# 
# " had the largest deferred_income of: 0" is a bit awkward and will need more exploration.

# In[27]:

### Explore deferred_income
def explore_def_inc():
    """
    function to explore deferred income
    """
    print "*"*100
    print "Explore Deferred Income"
    lowest = 0
    person = ""
    for i in data_dict:
        def_inc = data_dict[i]['deferred_income']
        if def_inc < lowest:
            lowest = def_inc
            person = str(i)
        print "{0!s} had a deferred_income of:".format(i), def_inc
    print "{} had the largest deferred income of:".format(person), lowest
    print "*"*100

explore_def_inc()


# It seems this was just a coding issue where deferred income was actually negative numbers. Taking absolute values in my original code would solve this.
# 
# Finally I want to check for cases where a person's features mainly consist of NaN values. These are probably less useful data points and can be removed.

# In[28]:

### Exploring NaN outliers
def explore_nan_outliers():
    largest = 0
    person = ""
    nan_dict = {}
    for k in data_dict:
        counter = 0
        for v in data_dict[k]:
            if data_dict[k][v] == "NaN":
                counter += 1
            if counter > largest:
                largest = counter
                person = str(k)
            nan_dict[str(k)] = counter
    print "*"*100
    print "Exploring NaN Outliers"
    print "{0} had the largest number of NaNs:".format(person), largest
    print "The five people with the most NaN values:", sorted(nan_dict, key=nan_dict.get, reverse=True)[:5]
    print "*"*100
    
explore_nan_outliers()
    


# Of the people with the most NaN values, 'LOCKHART EUGENE E' and 'THE TRAVEL AGENCY IN THE PARK' seem like outliers. Lockheart has a NaN value for everything except POI and The Travel Agency doesn't seem to be a person.
# 
# These will be removed for the final part of Outlier exploration.

# ### Remove Outliers

# In[29]:

### Remove all outliers

outlier_list = ['TOTAL', 'LOCKHART EUGENE E', 'THE TRAVEL AGENCY IN THE PARK']

print "*"*100
print "Remove Outliers"
for i in outlier_list:
    if i in data_dict:
        remove_outlier(i)
print "*"*100


# ### Create New Features
# 
# I will be adding three new features to the dataset that represent the percent of email interactions with POI's

# In[30]:

### Create new features
def add_poi_interaction_perc(data_dict, features_list):
    """
    add proportion of email interaction with pois
    """
    fields = ['to_messages', 'from_messages',
              'from_poi_to_this_person', 'from_this_person_to_poi']
    for k in data_dict:
        person = data_dict[k]
        is_valid = True
        for field in fields:
            if person[field] == 'NaN':
                is_valid = False
        if is_valid:
            total_messages = person['to_messages'] +                             person['from_messages']
            poi_messages = person['from_poi_to_this_person'] +                           person['from_this_person_to_poi']
            person['poi_interaction_perc'] = float(poi_messages) / total_messages
        else:
            person['poi_interaction_perc'] = 'NaN'
    feature_add('poi_interaction_perc')
    
def add_from_poi_perc(data_dict, features_list):
    """
    add proportion of emails from poi
    """
    fields = ['to_messages', 'from_messages',
              'from_poi_to_this_person', 'from_this_person_to_poi']
    for k in data_dict:
        person = data_dict[k]
        is_valid = True
        for field in fields:
            if person[field] == 'NaN':
                is_valid = False
        if is_valid:
            total_messages = person['to_messages'] +                             person['from_messages']
            from_poi_messages = person['from_poi_to_this_person']
            person['from_poi_perc'] = float(from_poi_messages) / total_messages
        else:
            person['from_poi_perc'] = 'NaN'
    feature_add('from_poi_perc')
    
def add_to_poi_perc(data_dict, features_list):
    """
    add proportion of emails to poi
    """
    fields = ['to_messages', 'from_messages',
              'from_poi_to_this_person', 'from_this_person_to_poi']
    for k in data_dict:
        person = data_dict[k]
        is_valid = True
        for field in fields:
            if person[field] == 'NaN':
                is_valid = False
        if is_valid:
            total_messages = person['to_messages'] +                             person['from_messages']
            to_poi_messages = person['from_this_person_to_poi']
            person['to_poi_perc'] = float(to_poi_messages) / total_messages
        else:
            person['to_poi_perc'] = 'NaN'
    feature_add('to_poi_perc')

add_poi_interaction_perc(data_dict, features_list)
add_from_poi_perc(data_dict, features_list)
add_to_poi_perc(data_dict, features_list)

print "Revised features list:", features_list


# Now that we have added new features, lets explore them a bit.

# In[31]:

### Explore new Features
data = featureFormat(data_dict, features_list, sort_keys = True)

explore_outlier('from_poi_perc', 'to_poi_perc', features_list, data)
explore_outlier('poi_interaction_perc', 'shared_receipt_with_poi', features_list, data)
isolate_outliers(features_list[18:])


# DONAHUE JR JEFFREY M had the highest ratio of interaction with POIs. This makes sense since he was a managing director. HUMPHREY GENE E was CEO of Enron Investment Partners and the largest percent of emails to POIs.

# ### Feature Scaling
# 
# Implementation for a feature scaling function in case of later need.

# In[32]:

### Feature scaling
def scale_features(features):
    """
    Scale features using the Min-Max algorithm
    """
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)

    return features

### Feature removal with Kbest (Not used, PCA will take it's place later)
#def get_k_best(data_dict, features_list, num_features):
#    """
#    runs scikit-learn's SelectKBest feature selection
#    returns dict where keys=features, values=scores
#    """
#    from sklearn.feature_selection import SelectKBest
#    
#    data = featureFormat(data_dict, features_list)
#    labels, features = targetFeatureSplit(data)
#
#    k_best = SelectKBest(k=num_features)
#    k_best.fit(features, labels)
#    scores = k_best.scores_
#    unsorted_pairs = zip(features_list[1:], scores)
#    sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
#    k_best_features = dict(sorted_pairs[:num_features])
#    print "{0} best features: {1}\n".format(num_features, k_best_features.keys())
#    return k_best_features
#    
#best_features = get_k_best(data_dict, features_list[1:], num_features)
#features_list = [features_list[0]] + best_features.keys()


# ### Unoptimized Classifier Exploration
# 
# Implementation of multiple classifiers that have not been optimized

# In[16]:

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Scale features if nessessary
features = scale_features(features)


# In[33]:

### Set up function for Unoptimized Classifier Exploration

### Set up list of classifiers function
def setup_classifer_list():
    """
    Initiate all classifiers and params to be used
    """
    classifier_list = []
    
    ### Naive Bayes methods are a set of supervised learning algorithms based on
    ### applying Bayes’ theorem with the “naive” assumption of independence between
    ### every pair of features.
    clf_nb = GaussianNB()
    params_nb = {}
    classifier_list.append((clf_nb, params_nb))
    
    ### Decision Trees (DTs) are a non-parametric supervised learning method
    ### used for classification and regression. The goal is to create a model
    ### that predicts the value of a target variable by learning simple decision
    ### rules inferred from the data features.
    clf_tree = DecisionTreeClassifier()
    params_tree = {"criterion": ['gini', 'entropy'],
                   "splitter": ['best', 'random'],
                   "max_features": [None, 'auto'],
                   "min_samples_split":[2, 5, 10, 20],
                   "class_weight": [None, "balanced"]}
    
    classifier_list.append((clf_tree, params_tree))

    ### Similar to SVC with parameter kernel=’linear’, but implemented in terms
    ### of liblinear rather than libsvm, so it has more flexibility in the choice
    ### of penalties and loss functions and should scale better to large numbers of samples.
    clf_svm = LinearSVC()
    params_svm = {"C": [0.5, 1, 5, 10, 100, 10**10],
                  "dual": [True, False],
                  "tol": [10**-1, 10**-10],
                  "class_weight": [None, 'balanced']}
    classifier_list.append((clf_svm, params_svm))

    ### The core principle of AdaBoost is to fit a sequence of weak learners
    ### (i.e., models that are only slightly better than random guessing, such
    ### as small decision trees) on repeatedly modified versions of the data. The
    ### predictions from all of them are then combined through a weighted majority
    ### vote (or sum) to produce the final prediction.
    clf_adaboost = AdaBoostClassifier()
    params_adaboost = {"n_estimators": [10, 20, 25, 30, 40, 50, 100]}
    classifier_list.append((clf_adaboost, params_adaboost))

    ### With RandomForest a diverse set of classifiers is created by introducing
    ### randomness in the classifier construction. The prediction of the ensemble
    ### is given as the averaged prediction of the individual classifiers.
    clf_random_tree = RandomForestClassifier()
    params_random_tree = {"n_estimators": [2, 3, 5],
                          "criterion": ['gini', 'entropy'], 
                          "max_features": [None, 'auto'],
                          "min_samples_split": [2, 5, 10, 20]}
    classifier_list.append((clf_random_tree, params_random_tree))

    ### The principle behind nearest neighbor methods is to find a predefined number
    ### of training samples closest in distance to the new point, and predict the label
    ### from these.
    clf_knn = KNeighborsClassifier()
    params_knn = {"n_neighbors": [2, 3, 5, 8],
                  "weights": ['uniform', 'distance'],
                  "p": [1, 2, 3]}
    classifier_list.append((clf_knn, params_knn))

    ### Logistic regression, despite its name, is a linear model for classification rather
    ### than regression. Logistic regression is also known in the literature as logit
    ### regression, maximum-entropy classification (MaxEnt) or the log-linear classifier.
    ### In this model, the probabilities describing the possible outcomes of a single trial
    ### are modeled using a logistic function.
    clf_log = LogisticRegression()
    params_log = {"dual": [True, False],
                  "C": [0.05, 0.5, 1, 10, 10**2,10**5,10**10, 10**20],
                  "class_weight": [None, 'balanced'],
                  "tol": [10**-1, 10**-5, 10**-10]}
    classifier_list.append((clf_log, params_log))

    return classifier_list

### Train Unoptimized Classifiers Function
def train_unoptimized(features_train, labels_train, pca_pipeline=False):
    """
    train classifiers but do not optimize them
    """
    classifier = setup_classifer_list()

    if pca_pipeline:
        classifier = transform_pca_pipeline(clf_supervised)
        
    train_list = []
    for clf, params in classifier:
        clf = clf.fit(features_train, labels_train)
        train_list.append(clf)
    
    return train_list

### Evaluation function
def evaluate(clf, features_test, labels_test):
    """
    function to evaluate classifier scores
    """
    pred = clf.predict(features_test)
            
    f1 = f1_score(labels_test, pred)
    recall = recall_score(labels_test, pred)
    precision = precision_score(labels_test, pred)
    
    return f1, recall, precision

def explore_unoptimized(features, labels, pca_pipeline=False, num_iters=10, test_size=0.3):
    """
    Run evaluation metrics multiple times to get a sense of how each classifer
    handles different data splits
    """

    eval_grid = [[] for n in range(7)] ### number of classifiers to get data on
    for i in range(num_iters):

        ### training and test sets
        features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=test_size)

        ### train classifiers
        classifier_list = train_unoptimized(features_train, labels_train, pca_pipeline)

        for i, clf in enumerate(classifier_list):
            scores = evaluate(clf, features_test, labels_test)
            eval_grid[i].append(scores)

    # Make a copy of the classifications list. Just want the structure.
    sum_list = {}
    for i, col in enumerate(eval_grid):
        sum_list[classifier_list[i]] = ((sum(np.asarray(col))) / num_iters)

    rank_list = sorted(sum_list.keys() , key = lambda k: sum_list[k][0], reverse=True)
    return rank_list, sum_list


# In[90]:

### Run unoptimized classifiers and return best fit
t0 = time()
unoptimized_rank_list, unoptimized_sum_list = explore_unoptimized(features, labels,
                                                                  pca_pipeline = False,
                                                                  num_iters=100, test_size=0.1)
print "*"*100
print "Multiple Training time:", round(time()-t0, 3), "s"
pprint(unoptimized_rank_list)
print "*"*100
pprint(unoptimized_sum_list)
print "*"*100

best_unoptimized = unoptimized_rank_list[0]
scores_unoptimized = unoptimized_sum_list[best_unoptimized]
print "Best unoptimized classifier is:", best_unoptimized
print "With scores of f1, recall, precision:", scores_unoptimized
print "*"*100


# The best unoptimized classifier with a recall of .28 and a precision of .29 turned out to be:
# 
# * AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0, n_estimators=50, random_state=None)
# 
# Let's try running this classifier manually.

# In[23]:

### Manually Unoptimized Classifier

clf = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
                         learning_rate=1.0, n_estimators=50, random_state=None)

print "*"*100
print "Test Classifier Manually"
test_classifier(clf, my_dataset, features_list)

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
print 'Feature Ranking:'
for i in range(len(features_list) - 1):
    print "{} feature no.{} ({})".format(i+1,indices[i],importances[indices[i]])
print "*"*100


# Using the provided tester, our results are similar. Randomness definitly plays a role in transfering classifier performance.
# 
# We find some interesting results. The best feature with this classifier is 'total_stock_value' with 16% of explained variance. This makes sense intuitively, as the people with the most stock had the most incentive to want stock prices to increase. This could lead to fraud.
# 
# Enough with this bare bones classifier, now let us try optimizing the classifiers with the use of PCA and GridsearchCV.

# ### Optimized Classifier Exploration
# 
# Implementation of multiple classifiers that have been optimized

# In[34]:

### Optimized Classifier exploration functions

### intilize pipelines for PCA function
def transform_pca_pipeline(classifier_list):
    """
    Function takes a classifier list and returns a list of piplines of the
    same classifiers and PCA.
    """

    pca = PCA()
    params_pca = {"pca__n_components":[2, 3, 4, 5, 10, 20], "pca__whiten":[False]}

    for i in range(len(classifier_list)):

        name = "clf_" + str(i)
        clf, params = classifier_list[i]

        # Modify params for gridsearch
        new_params = {}
        for key, value in params.iteritems():
            new_params[name + "__" + key] = value

        new_params.update(params_pca)
        classifier_list[i] = (Pipeline([("pca", pca), (name, clf)]), new_params)

    return classifier_list

### optimize classifiers function
def optimize_classifier_list(classifier_list, features_train, labels_train):
    """
    Takes a list of tuples for classifiers and parameters, and returns
    a list of the best estimator optimized to it's given parameters.
    """

    best_estimators = []
    
    for clf, params in classifier_list:
        
        scorer = make_scorer(f1_score)
        clf = GridSearchCV(clf, params, scoring=scorer)
        clf = clf.fit(features_train, labels_train)
        clf = clf.best_estimator_       
        clf_optimized = clf
        best_estimators.append(clf_optimized)

    return best_estimators

### train the optimized classifiers function
def train_optimized(features_train, labels_train, pca_pipeline=False):
    """
    train classifiers and optimize them
    """
    classifier = setup_classifer_list()

    if pca_pipeline:
        classifier = transform_pca_pipeline(classifier)

    classifier = optimize_classifier_list(classifier, features_train, labels_train)

    return classifier

### explore all the optimized classifiers function
def explore_optimized(my_dataset, features_list, folds = 1000, pca_pipeline=False):
    """
    Run evaluation metrics multiple times to get a sense of how each classifer
    handles different data splits (optimized)
    """
    
    ### initialize evaluation grid
    eval_grid = [[] for n in range(7)] ### number of classifiers to get data on
    
    ### get data for testing
    data = featureFormat(my_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    
    ### StratifiedShuffleSplit is a variation of ShuffleSplit, which returns stratified splits,
    ### i.e which creates splits by preserving the same percentage for each target class as in the complete set.
    cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
    
    ### initalize count to keep track of progress
    count = 1
    print "*"*100
    for train_idx, test_idx in cv:
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )

        ### train classifiers
        t0 = time()
        classifier_list = train_optimized(features_train, labels_train, pca_pipeline)
        print "Training and optimizing fold {} done in:".format(count), round(time()-t0, 3), "s"
        
        for i, clf in enumerate(classifier_list):
            scores = evaluate(clf, features_test, labels_test)
            eval_grid[i].append(scores)
        
        count += 1
    
    print "*"*100

    # Make a copy of the classifications list. Just want the structure.
    sum_list = {}
    for i, col in enumerate(eval_grid):
        sum_list[classifier_list[i]] = ((sum(np.asarray(col))) / folds)

    rank_list = sorted(sum_list.keys() , key = lambda k: sum_list[k][0], reverse=True)
    return rank_list, sum_list


# In[96]:

### Run optimized classifiers and return best fit (takes a long time with a lot of folds)
t0 = time()
optimized_rank_list, optimized_sum_list = explore_optimized(my_dataset, features_list,
                                                            folds = 2, pca_pipeline=True)
print "*"*100
print "Multiple Training time:", round(time()-t0, 3), "s"
pprint(optimized_rank_list)
print "*"*100
pprint(optimized_sum_list)
print "*"*100

best_optimized = optimized_rank_list[0]
scores_optimized = optimized_sum_list[best_optimized]
print "Best optimized classifier is:", best_optimized
print "With scores of f1, recall, precision:", scores_optimized
print "*"*100


# After running multiple classifiers through an optimizing program, we get that the KNeighborsClassifier worked best.

# ### Explore KNeighborsClassifier with plots

# In[246]:

### Explore our KNeighborsClassifier

### Create function to plot confusion matricies
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    function to plot confusion matricies
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['NonPOI', 'POI'], rotation=45)
    plt.yticks(tick_marks, ['NonPOI', 'POI'])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

### Create function to return plots for my classifier
def plot_classifier(my_dataset, features_list, folds=100):
    """
    function that returns plots for exploring my classifier
    """

    ### format data
    data = featureFormat(my_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    
    ### initalize PCA
    pca = PCA(copy=True)

    ### fit PCA for graphical purposes
    pca.fit(features)
    
    ### PLOT 1:
    ### PCA Spectrum
    plt.axes([.2, .2, .7, .7])
    plt.plot(pca.explained_variance_, linewidth=2)
    plt.xlabel('n_components')
    plt.ylabel('explained_variance_')
    print "*"*100
    print "PCA Spectrum"
    plt.show()
    print "*"*100
    
    ### stratify the data for multiple fold plots
    cv = StratifiedShuffleSplit(labels, folds, test_size=0.4)
    count = 1
    for train_idx, test_idx in cv:
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )

        ### PLOT 2:
        ### confusion matrix
        clf_final = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                                         metric_params=None, n_jobs=1, n_neighbors=3, p=3,
                                         weights='uniform')

        pca = PCA(copy=True, n_components=4, whiten=False)

        clf = Pipeline(steps=[("pca", pca), ("clf", clf_final)])
        clf.fit(features_train, labels_train)

        # predict
        pred = clf.predict(features_test)

        # Compute confusion matrix
        cm = confusion_matrix(labels_test, pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print "*"*100
        print "Normalized confusion matrix for fold:", count
        print(cm_normalized)
        plt.figure()
        plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
        plt.show()
        print "*"*100
        count += 1    
    
plot_classifier(my_dataset, features_list, folds=2)


# The PCA spectrum for our dataset drops off significantly after 5 components.
# 
# The confusion matrix shows that our classifier does a great job of finding true negatives and avoiding false positives. However, it also tends to find a lot of false negatives. It would seem the classifier guesses NonPOI more than would be optimal.
# 
# Now let's manually test this classifier using the provided tester.

# In[35]:

### Manually test Classifier
clf_final = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                                 metric_params=None, n_jobs=1, n_neighbors=3, p=3,
                                 weights='uniform')

pca = PCA(copy=True, n_components=4, whiten=False)

clf = Pipeline(steps=[("pca", pca), ("clf", clf_final)])

print "*"*100
print "Test Classifier Manually"
test_classifier(clf, my_dataset, features_list)
print "*"*100


# Using the provided tester, it seems that our classifier passes with:
# 
# * Accuracy: 0.88160
# * Precision: 0.61200
# * Recall: 0.30600
# 
# Now let's export our results for review.

# In[37]:

### dump data
print "*"*100
print "Dump data for testing"
print "Final feature list:", features_list

dump_classifier_and_data(clf, my_dataset, features_list)
print "*"*100


# ### Conclusion
# 
# Many different algorithms were tested with both an unoptimized version and an optimized version. The variances between algorithms were numerous, but I will focus on the difference between the best performing unoptimzed algorithm and the best performing optimized algorithm.
# 
# Unoptimized algorithm:
# 
# * AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0, n_estimators=50, random_state=None)
# * Accuracy: 0.83080 Precision: 0.33166 Recall: 0.26500	F1: 0.29461 F2: 0.27610
# 
# Optimized algorithm:
# 
# * Pipeline(steps=[('pca', PCA(copy=True, n_components=4, whiten=False)), ('clf_5', KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=1, n_neighbors=3, p=3, weights='uniform')
# * Accuracy: 0.88160 Precision: 0.61200 Recall: 0.30600	F1: 0.40800 F2: 0.34000
# 	
# The optimized algorithm performs better in every evaluation metric.  The final results being an accuracy of .88 and an F1 score of .41.
# 
