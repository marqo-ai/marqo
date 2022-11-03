#####################################################
### STEP 0. Import and define any helper functions
#####################################################

from marqo import Client
from marqo.errors import MarqoApiError
import torch
import json
import pprint

def read_json(filename: str) -> dict:
    # reads a json file
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def replace_title(data: dict) -> dict:
    # removes the wikipedia from the title for better matching
    data['title'] = data['title'].replace('- Wikipedia', '')
    return data

def split_big_docs(data, field='content', char_len=5e4):
    # there are some large documents which can cause issues for some users
    new_data = []
    for dat in data:
        
        content = dat[field]
        N = len(content)
        
        if N >= char_len:
            n_chunks = math.ceil(N / char_len)
            new_content = np.array_split(list(content), n_chunks)
            
            for _content in new_content:
                new_dat = copy.deepcopy(dat)
                new_dat[field] = ''.join(_content)
                new_data.append(new_dat)
        else:
            new_data.append(dat)
    return new_data

#####################################################
### STEP 1. load the data
#####################################################

# download the json formatted simplewiki from here - 
# https://www.kaggle.com/datasets/louisgeisler/simple-wiki?resource=download
# or from 
# https://drive.google.com/file/d/1OEqXeIdqaZb6BwzKIgw8G_sDi91fBawt/view?usp=sharing
dataset_file = "simplewiki.json"

# get the data
data = read_json(dataset_file)
# clean up the title
data = [replace_title(d) for d in data]
# split big ones to make it easier for users on all hardware
data = split_big_docs(data)
print(f"loaded data with {len(data)} entries")

#####################################################
### STEP 2. start Marqo
#####################################################

# Follow the instructions here https://github.com/marqo-ai/marqo

#####################################################
### STEP 3. index some data with marqo
#####################################################

# we use an index name. the index name needs to be lower case.
index_name = 'marqo-simplewiki-demo-all'

# setup the client
client = Client()

# we create the index. Note if it already exists an error will occur 
# as you cannot overwrite an existing index
# try:
#     client.delete_index(index_name)
# except:
#     pass

# we create the index and can set the model we want to use
# the onnx models are typically faster on both CPU and GPU
# to use non-onnx just use the name 'all_datasets_v4_MiniLM-L6'
client.create_index(index_name, model='onnx/all_datasets_v4_MiniLM-L6')

device = 'cpu'

# here we use parallel indexing to speed up the task
# Note: to use multiprocessing you will want at least 8GB of RAM and the maximum number 
# of processes that can be supported will be system dependent. 
responses = client.index(index_name).add_documents(data, device=device, processes=2, batch_size=50)

# optionally take a look at the responses
#pprint.print(responses)

#######################################
### STEP 4. Searching with marqo ######
#######################################


# after indexing we can search using both keyword (lexical) and neural search
# this will perform neural search across all indexed fields

# lets create a query
query = 'what is air made of?'

results = client.index(index_name).search(query)

# we can check the results - lets look at the top hit
pprint.pprint(results['hits'][0])

# we also get highlighting which tells us why this article was returned
pprint.pprint(results['hits'][0]['_highlights'])

# we can restrict the search to specific fields as well 
results = client.index(index_name).search(query, searchable_attributes=['content'])

# we can check the results - lets look at the top hit
pprint.pprint(results['hits'][0])

# we can check the results - lets look at the top hit
pprint.pprint(results['hits'][0])

# we use lexical search instead of tensor search
results = client.index(index_name).search(query, searchable_attributes=['content'], search_method='LEXICAL')

# we can check the results - lets look at the top hit
pprint.pprint(results['hits'][0])

# we can check the results - lets look at the top hit
pprint.pprint(results['hits'][0])


# lets create another query
query = 'what is a cube?'

results = client.index(index_name).search(query, searchable_attributes=['content'])

# we can check the results - lets look at the top hit
pprint.pprint(results['hits'][0])

# we also get highlighting which tells us why this article was returned
pprint.pprint(results['hits'][0]['_highlights'])

