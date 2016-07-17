
# coding: utf-8

# #OpenStreetMap Project - Code

# In[1]:


"""
Created on Thu Mar 17 15:45:00 2016
@author: David Shahrestani
"""

#Load Libraries for project
import xml.etree.cElementTree as ET
import pprint
import re
from collections import defaultdict
import codecs
import json
from pymongo import MongoClient
import operator
import os

#Set up path for OSM file
FILENAME = "c:\\users\\david shahrestani\\downloads\\data wrang temp\\project\\santa-barbara.osm"
FILENAMEJSON = "c:\\users\\david shahrestani\\downloads\\data wrang temp\\project\\santa-barbara.osm.json"

#Set up regular expressions for project
lower = re.compile(r'^([a-z]|_)*$')
lower_colon = re.compile(r'^([a-z]|_)*:([a-z]|_)*$')
street_type_re = re.compile(r'\b\S+\.?$', re.IGNORECASE)
postal_code_re = re.compile(r'(\d{5})-\d{4}')
postal_code_re2 = re.compile(r'(\d{5}):\d{5}')
problemchars = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')


# In[3]:

"""
Your task is to use the iterative parsing to process the map file and
find out not only what tags are there, but also how many, to get the
feeling on how much of which data you can expect to have in the map.
Fill out the count_tags function. It should return a dictionary with the 
tag name as the key and number of times this tag can be encountered in 
the map as value.

Note that your code will be tested with a different data file than the 'example.osm'    
"""

def add_tag(tag, tag_count):
    """Initialize or add a tag to tag_count"""
    if tag in tag_count:
        tag_count[tag] += 1
    else:
        tag_count[tag] = 1

        
def count_tags(filename):
    """Count tags in OSM file and print them out"""
    tag_count = {}
    tag_keys = {}
    counter = 0

    for _, element in ET.iterparse(filename, events=("start",)):
        add_tag(element.tag, tag_count)
        if element.tag == 'tag' and 'k' in element.attrib:
            add_tag(element.get('k'), tag_keys)

    tag_keys = sorted(tag_keys.items(), key=operator.itemgetter(1))[::-1]
    
    return tag_count, tag_keys


print(count_tags(FILENAME))


# In[13]:

"""
Your task is to explore the data a bit more.
Before you process the data and add it into MongoDB, you should
check the "k" value for each "<tag>" and see if they can be valid keys in MongoDB,
as well as see if there are any other potential problems.

We have provided you with 3 regular expressions to check for certain patterns
in the tags. As we saw in the quiz earlier, we would like to change the data model
and expand the "addr:street" type of keys to a dictionary like this:
{"address": {"street": "Some value"}}
So, we have to see if we have such tags, and if we have any tags with problematic characters.
Please complete the function 'key_type'.
"""

def key_type(element, keys):
    """Check "k" values against provided reg expressions"""
    if element.tag == "tag":
        k_value = element.attrib['k']
        if lower.search(k_value) is not None:
            keys['lower'] += 1
        elif lower_colon.search(k_value) is not None:
            keys['lower_colon'] += 1
        elif problemchars.search(k_value) is not None:
            keys["problemchars"] += 1
        else:
            keys['other'] += 1

    return keys

    
def process_map_keys(filename):
    """Process the OSM file to find key issues"""
    keys = {"lower": 0, "lower_colon": 0, "problemchars": 0, "other": 0}
    
    for _, element in ET.iterparse(filename):
        keys = key_type(element, keys)

    return keys


print(process_map_keys(FILENAME))


# In[14]:

"""
Your task is to explore the data a bit more.
The first task is a fun one - find out how many unique users
have contributed to the map in this particular area!

The function process_map should return a set of unique user IDs ("uid")
"""

def process_map_users(filename):
    """Process the OSM file to find out how many unique user ID's there are"""
    users = set()
    
    for _, element in ET.iterparse(filename):
        if element.tag == "node" or element.tag == "way" or element.tag == "relation":
            users.add(element.attrib['uid'])

    return users


print(process_map_users(FILENAME))


# In[2]:

"""
Your task in this exercise has two steps:

- audit the OSMFILE and change the variable 'mapping' to reflect the changes needed to fix 
    the unexpected street types to the appropriate ones in the expected list.
    You have to add mappings only for the actual problems you find in this OSMFILE,
    not a generalized solution, since that may and will depend on the particular area you are auditing.
- write the update_name function, to actually fix the street name.
    The function takes a string with street name as an argument and should return the fixed name
    We have provided a simple test so that you see what exactly is expected
"""

#expected street types
expected = ["Street", "Avenue", "Boulevard", "Drive", "Court", "Place", "Square", "Lane", "Road", 
            "Trail", "Parkway", "Commons"]

def audit_street_type(street_types, street_name):
    m = street_type_re.search(street_name)
    
    if m:
        street_type = m.group()
        if street_type not in expected:
            street_types[street_type].add(street_name)


def is_street_name(elem):
    return (elem.attrib['k'] == "addr:street")


def audit(osmfile):
    """Audit the OSM file for street types that are not expected"""
    osm_file = open(osmfile, "r", encoding="utf8")
    street_types = defaultdict(set)
    
    for event, elem in ET.iterparse(osm_file, events=("start",)):
        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_street_name(tag):
                    audit_street_type(street_types, tag.attrib['v'])
                                        
    return street_types


street_types = audit(FILENAME)
pprint.pprint(dict(street_types))


# In[3]:

#Mapping to correct street types
mapping = { "St.": "Street", "St": "Street",
            "Ave.": "Avenue", "Ave": "Avenue",
            "Rd.": "Road", "Rd": "Road",
            "W.": "West", "W": "West",
            "N.": "North", "N": "North",
            "S.": "South", "S": "South",
            "E.": "East", "E": "East",
            "Dr.": "Drive", "Dr": "Drive",
            "Blvd.": "Boulevard", "Blvd": "Boulevard",
            "Hwy": "Highway",
            "del": "Del"}

#Mapping to correct other errors noticed in manual scan of data
manual_mapping = {" Embarcadero del Norte": "Embarcadero Del Norte",
                  "Hwy 33 PM 30.12": "Highway 33",
                  "1116 Maricopa Hwy": "Maricopa Highway",
                  "339 W Gonzales Rd": "West Gonzales Road",
                  "\u200e3687 Sagunto St": "Sagunto Street",
                  "721 Jonata Park Road": "Jonata Park Road",
                  "400 Storke Road": "Storke Road",
                  "3999 State Street": "State Street",
                  "301 West Front Street": "West Front Street",
                  "Trigo": "Trigo Road",
                  "Loma Vista": "Loma Vista Avenue",
                  "Sabado Tarde": "Sabado Tarde Road",
                  "S Seaward": "South Seaward Drive",
                  "Del Playa": "Del Playa Drive",
                  "N. Fairview": "North Fairview Avenue",
                  "Abrego": "Abrego Street"}

def update_name(name, mapping, manual_mapping):  
    """Updates street names based on mapping above"""
    if name in manual_mapping.keys():
        name = manual_mapping[name]
        return name
    else:
        new_name = []
        for part in name.split(" "):
            if part in mapping.keys():
                part = mapping[part]
            new_name.append(part)
            
        return " ".join(new_name)


def test_update_name():
    for st_type, ways in street_types.items():
        for name in ways:
            better_name = update_name(name, mapping, manual_mapping)
            print(name, "=>", better_name)
            
            
test_update_name()


# In[4]:

"""
Further examination of the Postal Codes in the OSM file in order to audit and correct
any errors.
"""

#All tags representing Postal Codes discovered earlier in tag audit
ZIPCODE_TAGS = ['addr:postcode', 'tiger:zip_left', 'tiger:zip_left_1', 'tiger:zip_left_2', 
                'tiger:zip_left_3', 'tiger:zip_left_4', 'tiger:zip_right', 'tiger:zip_right_1',
                'tiger:zip_right_2', 'tiger:zip_right_3', 'tiger:zip_right_4']

def is_postal_code(elem):
    return (elem.attrib['k'] in ZIPCODE_TAGS)


def audit_postal_codes(osmfile):
    """Audit OSM file to find Postal Codes that do not being with 93 and are not length 5"""
    osm_file = open(osmfile, "r", encoding="utf8")
    postal_codes = {}
    
    for event, elem in ET.iterparse(osm_file, events=("start",)):
        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_postal_code(tag):
                    if postal_code_re.search(tag.attrib['v']) or postal_code_re2.search(tag.attrib['v']) or not tag.attrib['v'].startswith('93'):
                        if tag.attrib['v'] not in postal_codes:
                            postal_codes[tag.attrib['v']] = 1
                        else:
                            postal_codes[tag.attrib['v']] += 1
                        
    return postal_codes


postal_codes = audit_postal_codes(FILENAME)
print(postal_codes)


# In[5]:

#Mapping to correct error "3701" correct postal code researched manually.
code_mapping = {"3701": "93105"}

def update_postal_codes(postal, mapping):
    """Update postal codes programmatically and with mapping as necessary"""
    m = postal_code_re.search(postal)
    m2 = postal_code_re2.search(postal)
    
    if postal in mapping.keys():
        postal = mapping[postal]
    elif m:
        postal = m.group(1)
    elif m2:
        postal = m2.group(1)
    else:
        postal = postal
    
    return postal

        
def test_update_postal_codes():
    for code in postal_codes.keys():
        better_code = update_postal_codes(code, code_mapping)
        print(code, "=>", better_code)
                    
            
test_update_postal_codes()


# In[6]:


"""
Your task is to wrangle the data and transform the shape of the data
into the model we mentioned earlier. The output should be a list of dictionaries
that look like this:

{
"id": "2406124091",
"type: "node",
"visible":"true",
"created": {
          "version":"2",
          "changeset":"17206049",
          "timestamp":"2013-08-03T16:43:42Z",
          "user":"linuxUser16",
          "uid":"1219059"
        },
"pos": [41.9757030, -87.6921867],
"address": {
          "housenumber": "5157",
          "postcode": "60625",
          "street": "North Lincoln Ave"
        },
"amenity": "restaurant",
"cuisine": "mexican",
"name": "La Cabana De Don Luis",
"phone": "1 (773)-271-5176"
}

You have to complete the function 'shape_element'.
We have provided a function that will parse the map file, and call the function with the element
as an argument. You should return a dictionary, containing the shaped data for that element.
We have also provided a way to save the data in a file, so that you could use
mongoimport later on to import the shaped data into MongoDB. 

Note that in this exercise we do not use the 'update street name' procedures
you worked on in the previous exercise. If you are using this code in your final
project, you are strongly encouraged to use the code from previous exercise to 
update the street names before you save them to JSON. 

In particular the following things should be done:
- you should process only 2 types of top level tags: "node" and "way"
- all attributes of "node" and "way" should be turned into regular key/value pairs, except:
    - attributes in the CREATED array should be added under a key "created"
    - attributes for latitude and longitude should be added to a "pos" array,
      for use in geospacial indexing. Make sure the values inside "pos" array are floats
      and not strings. 
- if second level tag "k" value contains problematic characters, it should be ignored
- if second level tag "k" value starts with "addr:", it should be added to a dictionary "address"
- if second level tag "k" value does not start with "addr:", but contains ":", you can process it
  same as any other tag.
- if there is a second ":" that separates the type/direction of a street,
  the tag should be ignored, for example:

<tag k="addr:housenumber" v="5158"/>
<tag k="addr:street" v="North Lincoln Avenue"/>
<tag k="addr:street:name" v="Lincoln"/>
<tag k="addr:street:prefix" v="North"/>
<tag k="addr:street:type" v="Avenue"/>
<tag k="amenity" v="pharmacy"/>

  should be turned into:

{...
"address": {
    "housenumber": 5158,
    "street": "North Lincoln Avenue"
}
"amenity": "pharmacy",
...
}

- for "way" specifically:

  <nd ref="305896090"/>
  <nd ref="1719825889"/>

should be turned into
"node_refs": ["305896090", "1719825889"]
"""

CREATED = [ "version", "changeset", "timestamp", "user", "uid"]

def is_address(elem):
    """Return true if tag start with 'addr:'"""
    if elem.attrib['k'][:5] == "addr:":
        return True
    

def shape_element(element):
    """Process and shape data"""
    node = {}
    node["created"]={}
    node["address"]={}
    node["pos"]=[]
    refs=[]
    
    #Only process the node and way tags
    if element.tag == "node" or element.tag == "way":
        if "id" in element.attrib:
            node["id"] = element.attrib["id"]           
        node["type"] = element.tag
        
        #Add visible were available
        if "visible" in element.attrib.keys():
            node["visible"] = element.attrib["visible"]
      
        #Add values for the CREATED feild
        for elem in CREATED:
            if elem in element.attrib:
                node["created"][elem]=element.attrib[elem]
                
        #Add values for latitude and longitute      
        if "lat" in element.attrib:
            node["pos"].append(float(element.attrib["lat"]))
    
        if "lon" in element.attrib:
            node["pos"].append(float(element.attrib["lon"]))

        #Iterate through subtags
        for tag in element.iter("tag"):
            #Ignore problem characters
            if problemchars.search(tag.attrib['k']):
                continue
            
            #Add housenumber values
            if tag.attrib['k'] == "addr:housenumber":
                node["address"]["housenumber"]= tag.attrib['v']
            
            #Add and update postal code values
            if is_postal_code(tag):
                node["address"]["postcode"]= tag.attrib['v']
                node["address"]["postcode"]= update_postal_codes(node["address"]["postcode"], code_mapping)
                
            #Add and update street name values  
            if is_street_name(tag):
                node["address"]["street"]= tag.attrib['v']
                node["address"]["street"]= update_name(node["address"]["street"], mapping, manual_mapping)

            #Add values for non-address, non-postal tags that don't include ":"
            if not is_address(tag) and not is_postal_code(tag) and (":" not in tag.attrib['k']):
                node[tag.attrib['k']]=tag.attrib['v']

        #Extract node reference values
        for nd in element.iter("nd"):
             refs.append(nd.attrib["ref"])
        
        #Remove empty addresses
        if node["address"] == {}:
            node.pop("address", None)

        #Add node reference values
        if refs != []:
           node["node_refs"]=refs
            
        return node
    else:
        return None


def process_map(file_in, pretty = False):
    file_out = "{0}.json".format(file_in)
    data = []
    with codecs.open(file_out, "w") as fo:
        for _, element in ET.iterparse(file_in):
            el = shape_element(element)
            if el:
                data.append(el)
                if pretty:
                    fo.write(json.dumps(el, indent=2)+"\n")
                else:
                    fo.write(json.dumps(el) + "\n")
    
    print("Complete")
    return data


JSONdata = process_map(FILENAME, False)


# In[7]:

"""
Transfer the recenty created JSON file into MongoDB
using PyMongo. Create mongodbproject database and SB collection.
"""

def get_db(db_name):
    client = MongoClient("mongodb://localhost:27017")
    db = client[db_name]
    return db
    
    
def get_collection(db, collection):
    collections_db = db[collection]
    return collections_db
    

def insert_data(json_data, db_collection):
    """Insert the JSON data file into MongoDB"""
    with open(json_data, 'r') as f:
        for each_line in f.readlines():
            db_collection.insert(json.loads(each_line))
    print("Complete")

    
#Set up database and collection
db = get_db('mongodbproject')
db_SB = get_collection(db, 'SB')


insert_data(FILENAMEJSON, db_SB)


# In[8]:

"""
Data Overview for MongoDB collection
"""

#File sizes
print("santabarbara.osm size:", os.path.getsize(FILENAME)/1024/1024)
print("santabarbara.osm.json size:", os.path.getsize(FILENAMEJSON)/1024/1024)

#Documents
print("# of documents:", db_SB.find().count())

#Nodes
print("# of nodes:", db_SB.find({"type":"node"}).count())

#Ways
print("# of ways:", db_SB.find({"type":"way"}).count())
      
#Unique users
print("# of unique users:", len(db_SB.distinct("created.user")))

#Universites
print("# of universities:", db_SB.find({"amenity":"university"}).count())


# In[9]:

"""
Further analysis of the MongoDB collection using PyMongo
"""

#Aggregation operations return a cursor object
#Iterate over in order to print out one by one
def agg_print(result):
    for item in result:
        print(item)
        
#User Contributions        
pipeline = [{"$group":{"_id": "$created.user",
                       "count": {"$sum": 1}}},
            {"$project": {"proportion": {"$divide" :["$count",db_SB.find().count()]}}},
            {"$sort": {"proportion": -1}},
            {"$limit": 3}]
result = db_SB.aggregate(pipeline)
print("Proportions of top users' contributions:")
agg_print(result)


#Amenities
pipeline = [{"$match":{"amenity":{"$exists":1}}},
            {"$group":{"_id":"$amenity", "count":{"$sum":1}}},
            {"$sort":{"count":-1}},
            {"$limit":10}]
result = db_SB.aggregate(pipeline)
print("Top 10 appearing amenities:")
agg_print(result)


#Cuisines
pipeline = [{"$match":{"amenity":{"$exists":1}, "amenity":"restaurant", "cuisine":{"$exists":1}}}, 
            {"$group":{"_id":"$cuisine", "count":{"$sum":1}}},        
            {"$sort":{"count":-1}}, 
            {"$limit":10}]
result = db_SB.aggregate(pipeline)
print("Most popular cuisines:")
agg_print(result)


#Universities
pipeline = [{"$match":{"amenity":{"$exists":1}, "amenity": "university", "name":{"$exists":1}}},
            {"$group":{"_id":"$name", "count":{"$sum":1}}},
            {"$sort":{"count":-1}}]
result = db_SB.aggregate(pipeline)
print("Universities:")
agg_print(result)


# In[10]:

"""
Review of Postal Code and Street corrections
"""

#Postal codes
pipeline = [{"$group":{"_id": "$address.postcode",
                       "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}]
result = db_SB.aggregate(pipeline)
print("Postal codes:")
agg_print(result)


#Streets
pipeline = [{"$group":{"_id": "$address.street",
                       "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}]
result = db_SB.aggregate(pipeline)
print("Streets:")
agg_print(result)


# In[ ]:



