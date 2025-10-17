import marqo
import pprint
import pandas as pd


####################################################
### STEP 1: Load Data
####################################################

def load_data(file: str, number_data: int) -> dict:
    podcast_data = pd.read_csv(file).head(number_data)[['name', 'description']].to_dict('records')


# dataset came from this link: https://www.vox.com/today-explained
# the .csv file has the following headers:
# name, description
# (name of podcast, short description)

# create a 'transcript' key and add the transcript text as values to each record
id_counter = 1
for data in podcast_data:
    path = "data/transcripts/" + data['name'] + ".txt"
    with open(path, 'r') as f:
        content = f.read()
        data['transcript'] = content
        data['_id'] = str(id_counter)  # _id is a special key which is unique to every document
    id_counter += 1

return podcast_data

dataset_file = "data/podcast_data.csv"
podcast_data = load_data(dataset_file, 2)
'''
format of podcast_data -
[{'name': '....', 'description': '....', 'transcript': '....'},
{'name': '....', 'description': '....', 'transcript': '....'}]
'''

#####################################################
### STEP 2. Start Marqo
#####################################################

# Follow the instructions here https://github.com/marqo-ai/marqo/tree/2.0.0


####################################################
### STEP 3: Index Data
####################################################

index_name = "marqo-podcast-search-demo"
mq = marqo.Client(url='http://localhost:8882')  # Connection to Marqo Docker Container
mq.create_index(index_name)
mq.index(index_name).add_documents(podcast_data,
                                   tensor_fields=['name', 'description', 'transcript'], client_batch_size=64)
stats = mq.index(index_name).get_stats()  # get the stats for the index
print(f"{stats['numberOfDocuments']} documents added to index: {index_name}")

####################################################
### STEP 4: Search using Marqo
####################################################

# let's create a query and perform tensor search
query = 'what is long covid?'
results = mq.index(index_name).search(query)

# ['_highlights'] will return only the relevant portion rather than the whole transcript
print("Result 1 -", end=" ")
pprint.pprint(results['hits'][0]['_highlights'])  # [0] returns the top hit
print("Result 2 -", end=" ")
pprint.pprint(results['hits'][1]['_highlights'])  # [1] returns the second hit

# let's create another query and perform tensor search on a particular field
query = 'water issues in US'
results = mq.index(index_name).search(query)

print("Result 3 -", end=" ")
pprint.pprint(results['hits'][0]['_highlights'])

# let's create another query and perform lexical search on a particular field
query = 'water crisis'
results = mq.index(index_name).search(query, search_method='LEXICAL')

print("Result 4 -", end=" ")
pprint.pprint(results['hits'][0]['name'])
print("Result 5 -", end=" ")
pprint.pprint(results['hits'][0]['_highlights'])  # [_highlights] will return an empty list if using lexical search