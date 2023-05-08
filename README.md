<p align="center">
<img src="https://uploads-ssl.webflow.com/62dfa8e3960a6e2b47dc7fae/62fdf9cef684e6f16158b094_MARQO%20LOGO-UPDATED-GREEN.svg" width="90%" height="80%">
</p>

<p align="center">
<b><a href="https://marqo.ai">Website</a> | <a href="https://docs.marqo.ai">Documentation</a> | <a href="https://demo.marqo.ai">Demos</a>  | <a href="https://join.slack.com/t/marqo-community/shared_invite/zt-1d737l76e-u~b3Rvey2IN2nGM4wyr44w">Slack Community</a> | <a href="https://q78175g1wwa.typeform.com/to/d0PEuRPC">Marqo Cloud</a>
</b>
</p>

<p align="center">
<a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
<a href="https://pypi.org/project/marqo/"><img src="https://img.shields.io/pypi/v/marqo?label=PyPI"></a>
<a href="https://github.com/marqo-ai/marqo/actions/workflows/unit_test_CI.yml"><img src="https://img.shields.io/github/actions/workflow/status/marqo-ai/marqo/unit_test_CI.yml?branch=mainline"></a>
<a align="center" href="https://join.slack.com/t/marqo-community/shared_invite/zt-1d737l76e-u~b3Rvey2IN2nGM4wyr44w"><img src="https://img.shields.io/badge/Slack-blueviolet?logo=slack&amp;logoColor=white"></a>
</p>

<p align="center">
An end-to-end vector search engine that seamlessly integrates with applications and websites. Marqo allows developers to turbocharge search functionality with the latest machine learning models, in 3 lines of code.
</p>
    
<p align="center">
    <a href="https://demo.marqo.ai/?q=smiling+with+glasses&index=boredapes"><img src="assets/demo-short.gif"></a>
</p>
 
<p align="center">
    <b>    
    <a href="https://demo.marqo.ai">Try the demo</a> | <a href="https://github.com/marqo-ai/marqo-demo">View the code</a>
    </b>
</p>





## ✨ Core Features
**⚡ Performance**
- Embeddings stored in in-memory HNSW indexes, achieving cutting edge search speeds.
- Scale to hundred-million document indexes with horizontal index sharding.
- Async and non-blocking data upload and search.

**🤖 Machine Learning**
- Use the latest machine learning models from PyTorch, Huggingface, OpenAI and more. 
- Start with a pre-configured model or bring your own.
- Built in ONNX support and conversion for faster inference and higher throughput.
- CPU and GPU support.

**☁️ Cloud-native**
- Fast deployment using Docker.
- Run Marqo multi-az and high availability.

**🌌 End-to-end**
- Build search and analytics on multiple unstructured data types such as text, image, code, video.
- Filter search results using Marqo’s query DSL.
- Store unstructred data and semi-structured metadata together in documents, using a range of supported datatypes like bools, ints and keywords.

**🍱 Managed cloud**
- Scale Marqo at the click of a button.
- Multi-az, accelerated inference.
- Marqo cloud ☁️ is in beta. If you’re interested, [apply here](https://q78175g1wwa.typeform.com/to/d0PEuRPC).

<p align="center">
    <a href="https://q78175g1wwa.typeform.com/to/d0PEuRPC"><img src="assets/join-the-cloud-beta.png" width="90%" height="20%"/></a>
</p>

## Learn more about Marqo
                                                                                                                                                       
|                                                                                               |                                                                                                                                                                                                                                                   |
| --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 📗 [Quick start](#Getting-started)                                             | Build your first application with Marqo in under 5 minutes.                                                                                                                                                                                                 |
| 🔍 [ What is tensor search?](https://medium.com/@jesse_894/introducing-marqo-build-cloud-native-tensor-search-applications-in-minutes-9cb9a05a1736) | A beginner's guide to the fundamentals of Marqo and tensor search.                                                                                                                                                                                                                          |
| 🖼 [Marqo for image data](https://medium.com/@wanli19940222/how-to-implement-text-to-image-search-on-marqo-in-5-lines-of-code-448f75bed1da)      | Building text-to-image search in Marqo in 5 lines of code.                                                                                                                                                                                              |
| 📚 [Marqo for text](https://medium.com/@pandu_95301/how-i-used-marqo-to-create-a-multilingual-legal-database-in-5-key-lines-of-code-42ba49fd0caa)           | Building a multilingual database in Marqo.                                                                                                                                                                                  |
| 🔮 [Integrating Marqo with GPT](https://medium.com/creator-fund/building-search-engines-that-think-like-humans-e019e6fb6389)             | Making GPT a subject matter expert by using Marqo as a knowledge base.                                                                                                                                                                                                                    |
| 🎨 [ Marqo for Creative AI](https://medium.com/@jesse_894/combining-stable-diffusion-with-semantic-search-generating-and-categorising-100k-hot-dogs-afeeddea9d81)                             | Combining stable diffusion with semantic search to generate and categorise 100k images of hotdogs.                                                                                                                                                                                         |
| 🦾 [Features](#-Core-Features)                           | Marqo's core features.                                                                                                                                                                                                                     |





## Getting started


1. Marqo requires docker. To install Docker go to the [Docker Official website](https://docs.docker.com/get-docker/). Ensure that docker has at least 8GB memory and 50GB storage.

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
    q="What is the best outfit to wear on the moon?", searchable_attributes=["Title", "Description"]
)

```

- `mq` is the client that wraps the `marqo` API
- `add_documents()` takes a list of documents, represented as python dicts for indexing.
- `add_documents()` creates an index with default settings, if one does not already exist.
- You can optionally set a document's ID with the special `_id` field. Otherwise, Marqo will generate one.
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

- Each hit corresponds to a document that matched the search query.
- They are ordered from most to least matching.
- `limit` is the maximum number of hits to be returned. This can be set as a parameter during search.
- Each hit has a `_highlights` field. This was the part of the document that matched the query the best.

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

result = mq.index("my-first-index").search('marco polo', search_method=marqo.SearchMethods.LEXICAL)

```

### Search specific fields

Using the default tensor search method.

```python

result = mq.index("my-first-index").search('adventure', searchable_attributes=['Title'])

```

### Multi modal and cross modal search

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
    "My Image": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png",
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

results = mq.index("my-multimodal-index").search('https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png')

```

### Searching using weights in queries
Queries can also be provided as dictionaries where each key is a query and their corresponding values are weights. This allows for more advanced queries consisting of multiple components with weightings towards or against them, queries can have negations via negative weighting.

The example below shows the application of this to a scenario where a user may want to ask a question but also negate results that match a certain semantic criterion. 

```python

import marqo
import pprint

mq = marqo.Client(url="http://localhost:8882")

mq.index("my-weighted-query-index").add_documents(
    [
        {
            "Title": "Smartphone",
            "Description": "A smartphone is a portable computer device that combines mobile telephone "
            "functions and computing functions into one unit.",
        },
        {
            "Title": "Telephone",
            "Description": "A telephone is a telecommunications device that permits two or more users to"
            "conduct a conversation when they are too far apart to be easily heard directly.",
        },
        {
            "Title": "Thylacine",
            "Description": "The thylacine, also commonly known as the Tasmanian tiger or Tasmanian wolf, "
            "is an extinct carnivorous marsupial."
            "The last known of its species died in 1936.",
        },
    ]
)

# initially we ask for a type of communications device which is popular in the 21st century
query = {
    # a weighting of 1.1 gives this query slightly more importance
    "I need to buy a communications device, what should I get?": 1.1,
    # a weighting of 1 gives this query a neutral importance
    "Technology that became prevelant in the 21st century": 1.0,
}

results = mq.index("my-weighted-query-index").search(
    q=query, searchable_attributes=["Title", "Description"]
)

print("Query 1:")
pprint.pprint(results)

# now we ask for a type of communications which predates the 21st century
query = {
    # a weighting of 1 gives this query a neutral importance
    "I need to buy a communications device, what should I get?": 1.0,
    # a weighting of -1 gives this query a negation effect
    "Technology that became prevelant in the 21st century": -1.0,
}

results = mq.index("my-weighted-query-index").search(
    q=query, searchable_attributes=["Title", "Description"]
)

print("\nQuery 2:")
pprint.pprint(results)

```

### Creating and searching indexes with multimodal combination fields
Marqo lets you have indexes with multimodal combination fields. Multimodal combination fields can combine text and images into one field. This allows scoring of documents across the combined text and image fields together. It also allows for a single vector representation instead of needing many which saves on storage. The relative weighting of each component can be set per document.

The example below demonstrates this with retrival of caption and image pairs using multiple types of queries.

```python

import marqo
import pprint

mq = marqo.Client(url="http://localhost:8882")

settings = {"treat_urls_and_pointers_as_images": True, "model": "ViT-L/14"}

mq.create_index("my-first-multimodal-index", **settings)

mq.index("my-first-multimodal-index").add_documents(
    [
        {
            "Title": "Flying Plane",
            "captioned_image": {
                "caption": "An image of a passenger plane flying in front of the moon.",
                "image": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg",
            },
        },
        {
            "Title": "Red Bus",
            "captioned_image": {
                "caption": "A red double decker London bus traveling to Aldwych",
                "image": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image4.jpg",
            },
        },
        {
            "Title": "Horse Jumping",
            "captioned_image": {
                "caption": "A person riding a horse over a jump in a competition.",
                "image": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg",
            },
        },
    ],
    # Create the mappings, here we define our captioned_image mapping 
    # which weights the image more heavily than the caption - these pairs 
    # will be represented by a single vector in the index
    mappings={
        "captioned_image": {
            "type": "multimodal_combination",
            "weights": {
                "caption": 0.3,
                "image": 0.7,
            },
        }
    },
)

# Search this index with a simple text query
results = mq.index("my-first-multimodal-index").search(
    q="Give me some images of vehicles and modes of transport. I am especially interested in air travel and commercial aeroplanes.",
    searchable_attributes=["captioned_image"],
)

print("Query 1:")
pprint.pprint(results)

# search the index with a query that uses weighted components
results = mq.index("my-first-multimodal-index").search(
    q={
        "What are some vehicles and modes of transport?": 1.0,
        "Aeroplanes and other things that fly": -1.0,
    },
    searchable_attributes=["captioned_image"],
)
print("\nQuery 2:")
pprint.pprint(results)

results = mq.index("my-first-multimodal-index").search(
    q={"Animals of the Perissodactyla order": -1.0},
    searchable_attributes=["captioned_image"],
)
print("\nQuery 3:")
pprint.pprint(results)

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

## Documentation

The full documentation for Marqo can be found here [https://docs.marqo.ai/](https://docs.marqo.ai/).

## Warning

Note that you should not run other applications on Marqo's Opensearch cluster as Marqo automatically changes and adapts the settings on the cluster.

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

Marqo is a community project with the goal of making tensor search accessible to the wider developer community. We are glad that you are interested in helping out! Please read [this](./CONTRIBUTING.md) to get started.

## Dev set up

1. Create a virtual env ```python -m venv ./venv```.

2. Activate the virtual environment ```source ./venv/bin/activate```.

3. Install requirements from the requirements file: ```pip install -r requirements.txt```.

4. Ensure you have marqo-os running with `docker run -p 9200:9200 -p 9600:9600 -e "discovery.type=single-node" --name marqo-os marqoai/marqo-os:0.0.3`

5. Run tests by running the tox file. CD into this dir and then run "tox".

6. If you update dependencies, make sure to delete the .tox dir and rerun.

## Merge instructions:

1. Run the full test suite (by using the command `tox` in this dir).

2. Create a pull request with an attached github issue.

## Support

- Join our [Slack community](https://join.slack.com/t/marqo-community/shared_invite/zt-1d737l76e-u~b3Rvey2IN2nGM4wyr44w) and chat with other community members about ideas.
- Marqo community meetings (coming soon!).

### Stargazers

[![Stargazers repo roster for @marqo-ai/marqo](https://reporoster.com/stars/marqo-ai/marqo)](https://github.com/marqo-ai/marqo/stargazers).

### Forkers

[![Forkers repo roster for @marqo-ai/marqo](https://reporoster.com/forks/marqo-ai/marqo)](https://github.com/marqo-ai/marqo/network/members).

## Translations

This readme is available in the following translations:

- [English](README.md)🇬🇧
- [Français](README-translated/README-French.md)🇫🇷
- [中文 Chinese](README-translated/README-Chinese.md)🇨🇳
- [Polski](README-translated/README-Polish.md)🇵🇱
- [Українська](README-translated/README-Ukrainian.md)🇺🇦

