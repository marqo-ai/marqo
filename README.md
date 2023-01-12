<p align="center">
<a href=""><img src="https://uploads-ssl.webflow.com/62dfa8e3960a6e2b47dc7fae/62fdf9cef684e6f16158b094_MARQO%20LOGO-UPDATED-GREEN.svg"></a>
</p>

<p align="center">
<b><a href="https://marqo.ai">Website</a> | <a href="https://docs.marqo.ai">Documentation</a> | <a href="https://demo.marqo.ai">Demos</a>  | <a href="https://join.slack.com/t/marqo-community/shared_invite/zt-1d737l76e-u~b3Rvey2IN2nGM4wyr44w">Slack Community</a>
</b>
</p>

<p align="center">
<a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
<a href="https://pypi.org/project/marqo/"><img src="https://img.shields.io/pypi/v/marqo?label=PyPI"></a>
<a href="https://github.com/marqo-ai/marqo/actions/workflows/CI.yml"><img src="https://img.shields.io/github/workflow/status/marqo-ai/marqo/CI?label=CI"></a>
<a href="https://pepy.tech/project/marqo"><img alt="PyPI - Downloads from pepy" src="https://static.pepy.tech/personalized-badge/marqo?period=month&units=international_system&left_color=grey&right_color=blue&left_text=downloads/month"></a>
<a align="center" href="https://join.slack.com/t/marqo-community/shared_invite/zt-1d737l76e-u~b3Rvey2IN2nGM4wyr44w"><img src="https://img.shields.io/badge/Slack-blueviolet?logo=slack&amp;logoColor=white"></a>
</p>

A tensor-based search and analytics engine that seamlessly integrates with your applications, websites, and workflow in 3 lines of code.

<p align="center">
    <a href="https://demo.marqo.ai/?q=smiling+with+glasses&index=boredapes"><img src="https://user-images.githubusercontent.com/115690730/211964207-235e396b-1fc5-4785-b841-1bd037ddae4f.gif"></a>
</p>
 
<p align="center">
    <b>    
    <a href="https://demo.marqo.ai">Try the demo</a>
    </b> | <b>
    <a href="https://github.com/marqo-ai/marqo-demo">Check the code</a>
    </b>
</p>
    
<p align="center">
    <a href="https://q78175g1wwa.typeform.com/to/d0PEuRPC"><img src="https://i.imgur.com/lGOKnVg.png" width="500" height="125"/></a>
</p>




## ✨ Core Features
**⚡ Performant**
- Scale to million document indexes with horizontal index sharding.
- Async and non-blocking data upload and search.
- Supports lexical and tensor search in one place.

**☁️ Cloud-native**
- Fast deployment using Docker.
- Run multi-az and high availability

**🌌 End-to-end**
- Build search and analytics on multiple unstructured data types such as text, image, code, video.
- Filter search results using Marqo’s query DSL.
- Store unstructred data and semi-structured metadata together in documents, using a range of supported datatypes like bools, ints and keywords.

**🍱 Managed cloud**
- Scale marqo at the click of a button and Marqo at million document scale with high performace, including performant management of in-memory HNSW indexes
- Multi-az, accelorated inference
- Marqo cloud ☁️ is in beta. If you’re interested, [apply here](https://q78175g1wwa.typeform.com/to/d0PEuRPC).


## Understand Marqo Better
                                                                                                                                                       
|                                                                                               |                                                                                                                                                                                                                                                   |
| --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |

| 📗 [Quick start](#Getting-started)                                           | Build your first application with Marqo in under 5 minutes                                                                                                                                                                                                 |
| 🔍 [ What is tensor search?](https://medium.com/@jesse_894/introducing-marqo-build-cloud-native-tensor-search-applications-in-minutes-9cb9a05a1736)  | A beginner's guide to the fundamentals of Marqo and tensor search                                                                                                                                                                                                                           |
| 🖼 [Marqo for image data](https://medium.com/@wanli19940222/how-to-implement-text-to-image-search-on-marqo-in-5-lines-of-code-448f75bed1da)     
| Building text-to-image search in Marqo in 5 lines of code                                                                                                                                                                                             |
|  📚 [Marqo for text](https://medium.com/@pandu_95301/how-i-used-marqo-to-create-a-multilingual-legal-database-in-5-key-lines-of-code-42ba49fd0caa)   
| Building a multilingual database in Marqo                                                                                                                                                                                 |
| 🔮 [Integrating Marqo with GPT](https://medium.com/creator-fund/building-search-engines-that-think-like-humans-e019e6fb6389)   
| Making GPT a subject matter expert by using Marqo as a knowledge base |

|  🎨 [ Marqo for Creative AI](https://medium.com/@jesse_894/combining-stable-diffusion-with-semantic-search-generating-and-categorising-100k-hot-dogs-afeeddea9d81)                           
| Combining stable diffusion with semantic search to generate and categorise 100k images of hotdogs                                                                                                                                                                                                                     |
| 🦾 [Features](#-Core-Features) | Marqo's core features                                                                                                                                                                                         |




## Getting started

1. Marqo requires docker. To install Docker go to the [Docker Official website.]([https://docs.docker.com/get-docker/](https://docs.docker.com/get-docker/)). Ensure that docker has at least 8gb memory and 50gb storage.

2. Use docker to run Marqo (Mac users with M-series chips will need to [go here](#m-series-mac-users)):

```bash

docker rm -f marqo;

docker pull marqoai/marqo:0.0.6;

docker run --name marqo -it --privileged -p 8882:8882 --add-host host.docker.internal:host-gateway marqoai/marqo:0.0.6

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

result = mq.index("my-first-index").search('marco polo', search_method=marqo.SearchMethods.LEXICAL)

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

"treat_urls_and_pointers_as_images":True, # allows us to find an image file and index it

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

results = mq.index("my-multimodal-index").search('animal', searchable_attributes=['My Image'])

```

### Searching using an image

Searching using an image can be achieved by providing the image link.

```python

results = mq.index("my-multimodal-index").search('https://upload.wikimedia.org/wikipedia/commons/thumb/9/96/Standing_Hippopotamus_MET_DP248993.jpg/440px-Standing_Hippopotamus_MET_DP248993.jpg')

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

docker rm -f marqo-os; docker run -p 9200:9200 -p 9600:9600 -e "discovery.type=single-node" marqoai/marqo-os:0.0.2-arm

```

2. In another terminal run the following command to launch Marqo:

```shell

docker rm -f marqo; docker run --name marqo --privileged \

- p 8882:8882 --add-host host.docker.internal:host-gateway \
- e "OPENSEARCH_URL=https://localhost:9200" \

marqoai/marqo:0.0.6

```

## Contributors

Marqo is a community project with the goal of making tensor search accessible to the wider developer community. We are glad that you are interested in helping out! Please read [this](./CONTRIBUTING.md) to get started

## Dev set up

1. Create a virtual env ```python -m venv ./venv```

2. Activate the virtual environment ```source ./venv/bin/activate```

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

text filtering
