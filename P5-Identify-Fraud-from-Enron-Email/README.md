# Identify Fraud from Enron Email

### Project Overview
In this project, you will play detective, and put your machine learning skills to use by building an algorithm to identify Enron Employees who may have committed fraud based on the public Enron financial and email dataset.

### Why this Project?
This project will teach you the end-to-end process of investigating data through a machine learning lens.

It will teach you how to extract and identify useful features that best represent your data, a few of the most commonly used machine learning algorithms today, and how to evaluate the performance of your machine learning algorithms.

### What will I learn?
By the end of the project, you will be able to:
* Deal with an imperfect, real-world dataset
* Validate a machine learning result using test data
* Evaluate a machine learning result using quantitative metrics
* Create, select and transform features
* Compare the performance of machine learning algorithms
* Tune machine learning algorithms for maximum performance
* Communicate your machine learning algorithm results clearly

## Getting Started
Install required libraries and run poi_id.py:  
```
### import file libraries
import sys
import pickle

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
```

The tools folder contains two supplemental programs
* feature_format.py helps structure our data for analysis
* tester.py is used to test our analysis and dump the data for review
