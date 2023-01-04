<p align="center">
  <img src="assets/logo2.svg" alt="Marqo"/>
</p>

<h1 align="center">Marqo</h1>

<p align="center">
  <b>Tensor search for humans.</b>
</p>

<p align="center">
<a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
<a href="https://pypi.org/project/marqo/"><img src="https://img.shields.io/pypi/v/marqo?label=PyPI"></a>
<a href="https://github.com/marqo-ai/marqo/actions/workflows/unit_test_CI.yml"><img src="https://img.shields.io/github/actions/workflow/status/marqo-ai/marqo/unit_test_CI.yml?branch=mainline"></a>
<a href="https://pepy.tech/project/marqo"><img alt="PyPI - Downloads from pepy" src="https://static.pepy.tech/personalized-badge/marqo?period=month&units=international_system&left_color=grey&right_color=blue&left_text=downloads/month"></a>
<a align="center" href="https://join.slack.com/t/marqo-community/shared_invite/zt-1d737l76e-u~b3Rvey2IN2nGM4wyr44w"><img src="https://img.shields.io/badge/Slack-blueviolet?logo=slack&amp;logoColor=white"></a>
</p>


An open-source tensor search engine that seamlessly integrates with your applications, websites, and workflow. 

Marqo cloud ☁️  is in beta. If you're interested, apply here: https://q78175g1wwa.typeform.com/to/d0PEuRPC

## What is tensor search?

Tensor search involves transforming documents, images and other data into collections of vectors called "tensors". Representing data as tensors allows us to match queries against documents with human-like understanding of the query and document's content. Tensor search can power a variety of use cases such as:
- end user search and recommendations
- multi-modal search (image-to-image, text-to-image, image-to-text)
- chat bots and question and answer systems
- text and image classification

<p align="center">
  <img src="assets/output.gif"/>
</p>

<!-- end marqo-description -->

## Getting started

1. Marqo requires docker. To install Docker go to the [Docker Official website.](https://docs.docker.com/get-docker/)
2. Use docker to run Marqo (Mac users with M-series chips will need to [go here](#m-series-mac-users)):
```bash
docker rm -f marqo
docker pull marqoai/marqo:latest
docker run --name marqo -it --privileged -p 8882:8882 --add-host host.docker.internal:host-gateway marqoai/marqo:latest
```
3. Install the Marqo client:
```bash
pip install marqo
```
4. Start indexing and searching! Let's look at a simple example below:

```python
import marqo

mq = marqo.Client(url='http://localhost:8882')

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

- `mq` is the client that wraps the `marqo` API
- `add_documents()` takes a list of documents, represented as python dicts, for indexing
- `add_documents()` creates an index with default settings, if one does not already exist
- You can optionally set a document's ID with the special `_id` field. Otherwise, Marqo will generate one.
- If the index doesn't exist, Marqo will create it. If it exists then Marqo will add the documents to the index.
- Running this code multiple times could result in duplicate documents. To reset the index, you can delete it first using `mq.index("my-first-index").delete()`

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
            '_score': 0.61938936
        }, 
        {   
            'Title': 'The Travels of Marco Polo',
            'Description': "A 13th-century travelogue describing Polo's travels",
            '_highlights': {'Title': 'The Travels of Marco Polo'},
            '_id': 'e00d1a8d-894c-41a1-8e3b-d8b2a8fce12a',
            '_score': 0.60237324
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
Using the default tensor search method
```python
result = mq.index("my-first-index").search('adventure', searchable_attributes=['Title'])
```

### Delete documents
Delete documents.

```python
results = mq.index("my-first-index").delete_documents(ids=["article_591", "article_602"])
```

### Delete index
Delete an index.

```python
results = mq.index("my-first-index").delete()
```

## Multi modal and cross modal search

To power image and text search, Marqo allows users to plug and play with CLIP models from HuggingFace. **Note that if you do not configure multi modal search, image urls will be treated as strings.** To start indexing and searching with images, first create an index with a CLIP configuration, as below:

```python

settings = {
  "treat_urls_and_pointers_as_images":True,   # allows us to find an image file and index it 
  "model":"ViT-L/14"
}
response = mq.create_index("my-multimodal-index", **settings)
```

Images can then be added within documents as follows. You can use urls from the internet (for example S3) or from the disk of the machine:

```python

response = mq.index("my-multimodal-index").add_documents([{
    "My Image": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f2/Portrait_Hippopotamus_in_the_water.jpg/440px-Portrait_Hippopotamus_in_the_water.jpg",
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
results = mq.index("my-multimodal-index").search('https://upload.wikimedia.org/wikipedia/commons/thumb/9/96/Standing_Hippopotamus_MET_DP248993.jpg/440px-Standing_Hippopotamus_MET_DP248993.jpg')
```

## Documentation
The full documentation for Marqo can be found here [https://docs.marqo.ai/](https://docs.marqo.ai/).

## M series Mac users
Marqo does not yet support the docker-in-docker backend configuration for the arm64 architecture. This means that if you have an M series Mac, you will also need to run marqo's backend, marqo-os, locally.

To run Marqo on an M series Mac, follow the next steps.

1. In one terminal run the following command to start opensearch:

```shell
docker rm -f marqo-os; docker run -p 9200:9200 -p 9600:9600 -e "discovery.type=single-node" marqoai/marqo-os:0.0.3-arm
```

2. In another terminal run the following command to launch Marqo:
```shell
docker rm -f marqo; docker run --name marqo --privileged \
    -p 8882:8882 --add-host host.docker.internal:host-gateway \
    -e "OPENSEARCH_URL=https://localhost:9200" \
    marqoai/marqo:latest
```

## Contributors
Marqo is a community project with the goal of making tensor search accessible to the wider developer community. We are glad that you are interested in helping out! Please read [this](./CONTRIBUTING.md) to get started

## Dev set up
1. Create a virtual env ```python -m venv ./venv```
2. Activate the virtual environment ```source ./venv/bin/activate``` (on Linux or Mac) or ```./venv/Scripts/activate``` (on Windows)
3. Install requirements from the requirements file: ```pip install -r requirements.txt```
4. Run tests by running the tox file. CD into this dir and then run "tox"
5. If you update dependencies, make sure to delete the .tox dir and rerun

## Merge instructions:
1. Run the full test suite (by using the command `tox` in this dir).
2. Create a pull request with an attached github issue.


## Support

- Join our [Slack community](https://join.slack.com/t/marqo-community/shared_invite/zt-1d737l76e-u~b3Rvey2IN2nGM4wyr44w) and chat with other community members about ideas.
- Marqo community meetings (coming soon!)

### Stargazers
[![Stargazers repo roster for @marqo-ai/marqo](https://reporoster.com/stars/marqo-ai/marqo)](https://github.com/marqo-ai/marqo/stargazers)

### Forkers
[![Forkers repo roster for @marqo-ai/marqo](https://reporoster.com/forks/marqo-ai/marqo)](https://github.com/marqo-ai/marqo/network/members)


## Translations

This readme is available in the following translations:

- [English](README.md)🇬🇧
- [Français](README-translated/README-French.md)🇫🇷
- [中文 Chinese](README-translated/README-Chinese.md)🇨🇳
- [Polski](README-translated/README-Polish.md)🇵🇱
- [Українська](README-translated/README-Ukrainian.md)🇺🇦
