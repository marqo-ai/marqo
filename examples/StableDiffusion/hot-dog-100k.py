#####################################################
### STEP 0. Import and define any helper functions
#####################################################

from marqo import Client
from marqo.errors import MarqoApiError
import torch
import json
import pprint
import glob
import os
import pandas as pd
import subprocess

#####################################################
### STEP 1. Setup some things and create the documents for indexing
#####################################################

# this should be where the images are unzipped
# get the dataset from here https://drive.google.com/file/d/16_1MlX9GH-6v060jYA23eTJwH74fSU4L/view?usp=sharing
images_directory = 'hot-dog-100k/'

# the images are accessed via docker from here - you will be able
# to access them at something like http://[::]:8000/ or http://localhost:8000/
docker_path = 'http://host.docker.internal:8222/'

# we start an image server for easier access from within docker
pid = subprocess.Popen(['python3', '-m', 'http.server', '8222', '--directory', images_directory], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

# now find all the files
files = glob.glob(images_directory + "/*.jpg")

# we want to map the filename only with its docker path
files_map = {os.path.basename(f):f for f in files}

# update them to use the correct path
files_docker = [f.replace(images_directory, docker_path) for f in files]

# now we create our documents for indexing - a list of python dicts with keys as field names
documents = [{"image_docker":file_docker, '_id':os.path.basename(file_docker)} for file_docker,file_local in zip(files_docker, files)]

#####################################################
### STEP 2. Initial indexing
#####################################################

# we create the index and can set the model we want to use


# get the marqo client
client = Client()

# index name - should be lowercase
index_name = 'hot-dogs-100k'

settings = {
        "model":'ViT-L/14',
        "treat_urls_and_pointers_as_images": True,
        }
client.create_index(index_name, **settings)

# here we use parallel indexing to speed up the task - a gpu is recomended (device='cuda')
responses = client.index(index_name).add_documents(documents, device='cpu'
                                                    , processes=4, batch_size=50)


#####################################################
### Step 3. Add some labels via zero-shot learning
#####################################################


index_name = 'one_dog_two'

# the documents here are actually serving as labels. the document (i.e. label)
# that most closely matches is returned first and can be used as a label
# each one gets scored and those scores can also be kept
labels = [{"label":"one hot dog"}, {"label":"two hot dogs"},
                {"label":"a hamburger"}, {"label": "a face"}]

# get a copy of the labels only
label_strings = [list(a.values())[0] for a in labels]

# we create a new index
settings = {
        "model":'ViT-L/14',
        "treat_urls_and_pointers_as_images": True,
        }
client.create_index(index_name, **settings)

# add our labels to the index
responses = client.index(index_name).add_documents(labels)

# loop through the documents and search against the labels to get scores
for doc in documents:

    # the url for the image is what is used as the search - an image
    # note: you will want a gpu to index the whole dataset device="cuda"
    responses = client.index(index_name).search(doc['image_docker'], device='cpu')

    # now retrieve the score for each label and add it to our document
    for lab in label_strings:
        doc[lab.replace(' ','_')] = [r['_score'] for r in responses['hits'] if r['label'] == lab][0]

documents_image_docker = [doc.pop('image_docker') for doc in documents]
responses = client.index("hot-dogs-100k").add_documents(documents, device='cpu',
                                                            processes=3, batch_size=50)

#####################################################
### Step 4. Remove the black images
#####################################################

query = 'a black image'

results = client.index(index_name).search(query)

# remove the blank images
results = client.index(index_name).search(results['hits'][0]['image_docker'], limit=100)

# we check the results - scores of very close to 1 are duplicated (this value can change depending on the task)
documents_delete = [r['_id'] for r in results['hits'] if r['_score'] > 0.99999]

client.index(index_name).delete_documents(documents_delete)


#####################################################
### Step 5. order the images based on their similarity with each other
#####################################################

# pick one to start
results = client.index("hot-dogs-100k").search('a photo of a smiling face', 
                        searchable_attributes=['image_docker'], 
                        filter_string="a_face:[0.58 TO 0.99] AND a_hamburger:[0.60 TO 0.99]", device='cuda')

# find the document that matches closest with the query
index = [ind for ind,doc in enumerate(documents) if doc['_id'] == results['hits'][0]['_id'] ][0]
current_document = documents[index]
# create a list to store the "sorted" documents
ordered_documents = [current_document['_id']]

for i in range(len(documents)):

    # remove current document
    client.index(index_name).delete_documents([current_document['_id']])

    # now search with it to get next best
    results = client.index(index_name).search(current_document['image_docker'],
                            searchabel_attributes=['image_docker'], 
                            filter_string="a_face:[0.58 TO 0.99] AND a_hamburger:[0.60 TO 0.99]",
                            device='cuda')

    next_document = results['hits'][0]

    # now add it
    ordered_documents.append(next_document['_id'])

    current_document = next_document

ordered_images = [files_map[f] for f in ordered_documents]
deleted_documents = [d for d in documents if d['_id'] in ordered_documents]

#####################################################
### Step 6. Animate them
#####################################################
import sys
import subprocess
from pathlib import Path

def prepend_number(filename, number, new_dir):
    _dir, _name = os.path.split(filename)

    return _dir + f'/{new_dir}/' + number + '_' + _name

def copyWithSubprocess(cmd):        
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

save_base_dir = 'outputs/'
Path(save_base_dir).mkdir(parents=True, exist_ok=True)

new_images = [save_base_dir + f'{str(i).zfill(5)}' + os.path.basename(f) for i,f in enumerate(ordered_images)]

for image, new_image in zip(ordered_images, new_images):

    cmd=['cp', image, new_image]

    copyWithSubprocess(cmd)
