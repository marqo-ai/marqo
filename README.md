<p align="center">
  <img src="assets/logo.svg" alt="Marqo" width="150" height="150" />
</p>

<h1 align="center">Marqo</h1>

<p align="center">
  <b>Neural search for humans.</b>
</p>

<p align="center">
  <a align="center" href="https://join.slack.com/t/marqo-community/shared_invite/zt-1d737l76e-u~b3Rvey2IN2nGM4wyr44w"><img src="https://img.shields.io/badge/Slack-blueviolet?logo=slack&amp;logoColor=white&style=flat-square"></a>
</p>

A deep-learning powered, open-source search engine which seamlessly integrates with your applications, websites, and workflow. 

<!-- end marqo-description -->

## Getting started

1. Marqo requires docker. To install docker go to https://docs.docker.com/get-docker/
2. Use docker to run [Opensearch](https://opensearch.org/):
```bash
docker run -p 9200:9200 -p 9600:9600 -e "discovery.type=single-node" opensearchproject/opensearch:2.1.0
```
3. Install the Marqo client:
```bash
pip install marqo
```
4. Start indexing and searching! Let's look at a simple example below:

```python
import marqo

mq = marqo.Client(url='https://localhost:9200', main_user="admin", main_password="admin")

mq.index("my-first-index").add_documents([
    {
        "Title": "The Travels of Marco Polo",
        "Description": "A 13th-century travelogue describing Polo's travels"
    }, 
    {
        "Title": "Extravehicular Mobility Unit (EMU)",
        "Description": "The EMU is a spacesuit that provides environmental protection, "
                       "mobility, life support, and communications for astronauts",
        "_id": "article_591"
    }]
)

results = mq.index("my-first-index").search(
    q="What is the best outfit to wear on the moon?"
)

```

- `mq` is the client that wraps the`marqo` API
- `add_documents()` takes a list of documents, represented as python dicts, for indexing
- `add_documents()` creates an index with default settings, if one does not already exist
- You can optionally set a document's ID with the special `_id` field. Otherwise, marqo will generate one.
- If the index doesn't exist, Marqo will create it. If it exists then Marqo will add the documents to the index.

Let's have a look at the results:

```python
# let's print out the results:
import pprint
pprint.pprint(results)

{
    'hits': [
        {   
            'Title': 'Extravehicular Mobility Unit (EMU)',
            'Description': 'The EMU is a spacesuit that provides environmental protection, mobility, life support, and' 
                           'communications for astronauts',
            '_highlights': {
                'Description': 'The EMU is a spacesuit that provides environmental protection, '
                               'mobility, life support, and communications for astronauts'
            },
            '_id': 'article_591',
            '_score': 1.2387788
        }, 
        {   
            'Title': 'The Travels of Marco Polo',
            'Description': "A 13th-century travelogue describing Polo's travels",
            '_highlights': {'Title': 'The Travels of Marco Polo'},
            '_id': 'e00d1a8d-894c-41a1-8e3b-d8b2a8fce12a',
            '_score': 1.2047464
        }
    ],
    'limit': 10,
    'processingTimeMs': 49,
    'query': 'What is the best outfit to wear on the moon?'
}
```

- Each hit corresponds to a document that matched the search query
- They are ordered from most to least matching
- `limit` is the maximum number of hits to be returned. This can be set as a parameter during search
- Each hit has a `_highlights` field. This was the part of the document that matched the query the best


## Other basic operations

### Get document
Retrieve a document by ID.

```python
result = mq.index("my-first-index").get_document(document_id="article_591")
```

Note that by adding the document using ```add_documents``` again using the same ```_id``` will cause a document to be updated.

### Get index stats
Get information about an index.

```python
results = mq.index("my-first-index").get_stats()
```

### Lexical search
Perform a keyword search.

```python
result =  mq.index("my-first-index").search('marco polo', search_method=marqo.SearchMethods.LEXICAL)
```

### Search specific fields
Using the default neural search method
```python
result = mq.index("my-first-index").search('adventure', searchable_attributes=['Title'])
```

## Multi modal and cross modal search

To power image and text search, Marqo allows users to plug and play with CLIP models from HuggingFace. **Note that if you do not configure multi modal search, image urls will be treated as strings.** To start indexing and searching with images, first create an index with a CLIP configuration, as below:

```python

settings = {
  "treat_urls_and_pointers_as_images":True,   # allows us to find an image file and index it 
  "model":"ViT-B/32"
}
response = mq.create_index("my-multimodal-index", **settings)
```

Images can then be added within documents as follows. You can use urls from the internet (for example S3) or from the disk of the machine:

```python

response = mq.index("my-multimodal-index").add_documents([{
    "My Image": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b3/Hipop%C3%B3tamo_%28Hippopotamus_amphibius%29%2C_parque_nacional_de_Chobe%2C_Botsuana%2C_2018-07-28%2C_DD_82.jpg/640px-Hipop%C3%B3tamo_%28Hippopotamus_amphibius%29%2C_parque_nacional_de_Chobe%2C_Botsuana%2C_2018-07-28%2C_DD_82.jpg",
    "Description": "The hippopotamus, also called the common hippopotamus or river hippopotamus, is a large semiaquatic mammal native to sub-Saharan Africa",
    "_id": "hippo-facts"
}])

```

You can then search using text as usual. Both text and image fields will be searched:
```python

results = mq.index("my-multimodal-index").search('animal')

```
 Setting `searchable_attributes` to the image field `['My Image'] ` ensures only images are searched in this index:

```python

results = mq.index("my-multimodal-index").search('animal',  searchable_attributes=['My Image'])

```

### Searching using an image
Searching using an image can be achieved by providing the image link. 
```python
results = mq.index("my-multimodal-index").search('https://upload.wikimedia.org/wikipedia/commons/thumb/9/96/Standing_Hippopotamus_MET_DP248993.jpg/1920px-Standing_Hippopotamus_MET_DP248993.jpg')
```


### Delete index
Delete an index.

```python
results = mq.index("my-first-index").delete()
```


### Delete documents
Delete documents.

```python
results = mq.index("my-first-index").delete_documents(ids=["article_591", "article_602"])
```



## A note when using a GPU
Depending on the class of GPU, a version of PyTorch compiled with the latest CUDA (>11.3) may be required. 
If for example, an error appears similar to the following;

```
NVIDIA #### with CUDA capability sm_86 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_37 sm_50 sm_60 sm_70.
If you want to use the NVIDIA #### GPU with PyTorch, please check the instructions at https://pytorch.org/get-started/locally/
```
then PyTorch with the appropriate CUDA should be installed. For example, to install PyTorch 1.12 with CUDA 11.6 do the following;
```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116 --upgrade
```
It should be noted that the CUDA version the current driver supports can be obtained by using the following command in the terminal;
```
$nvidia-smi
```
The respective PyTorch installation should have a CUDA version that does not exceed this. PyTorch installation instrucitons can be
found here https://pytorch.org/get-started/locally/ and previous versions with other CUDA options can be found at https://pytorch.org/get-started/previous-versions/.

## Warning

Note that you should not run other applications on Marqo's Opensearch cluster as Marqo automatically changes and adapts the settings on the cluster.

## Contributors
Marqo is a community project with the goal of making neural search accessible to the wider developer community. We are glad that you are interested in helping out! Please read [this](./CONTRIBUTING.md) to get started

## Dev set up
1. Create a virtual env ```python -m venv ./venv```
2. Activate the virtual environment ```source ./venv/bin/activate```
3. Install requirements from the requirements file: ```pip install -r requirements.txt```
4. Run tests by running the tox file. CD into this dir and then run "tox"
5. If you update dependencies, make sure to delete the .tox dir and rerun

## Merge instructions:
1. Run the full test suite (by using the command `tox` in this dir).
2. Create a pull request with an attached github issue.

The large data test will 
build Marqo from the main branch and fill indices with data. Go through and test queries 
against this data. https://github.com/S2Search/NeuralSearchLargeDataTest

<!-- start support-pitch -->


## Support

- Join our [Slack community](https://join.slack.com/t/marqo-community/shared_invite/zt-1d737l76e-u~b3Rvey2IN2nGM4wyr44w) and chat with other community members about ideas.
- Marqo community meetings (coming soon!)

<!-- end support-pitch -->
