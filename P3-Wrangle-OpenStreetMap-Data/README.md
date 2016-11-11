# Wrangle OpenStreetMaps Data

## Project Overview
You will choose any area of the world in https://www.openstreetmap.org and use data munging techniques, such as assessing the quality of the data for validity, accuracy, completeness, consistency and uniformity, to clean the OpenStreetMap data for a part of the world that you care about. Finally, you will choose either MongoDB or SQL as the data schema to complete your project.

## Why this Project?
What’s so hard about retrieving data from databases or various files formats? You grab some data from this file and that database, clean it up, merge it, and then feed it into your state of the art, deep learning algorithm … Right?

But the reality is this -- anyone who has worked with data extensively knows it is an absolute nightmare to get data from different data sources to play well with each other.

And this project will teach you all of the skills you need to deal with even the most nightmarish data wrangling scenarios.

## Getting Started
Install required libraries and run project3_code.py:  
```
import xml.etree.cElementTree as ET  
import pprint  
import re  
from collections import defaultdict  
import codecs  
import json  
from pymongo import MongoClient  
import operator  
import os
```
