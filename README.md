<p align="center">
<img src="https://uploads-ssl.webflow.com/62dfa8e3960a6e2b47dc7fae/62fdf9cef684e6f16158b094_MARQO%20LOGO-UPDATED-GREEN.svg" width="50%" height="40%">
</p>

<p align="center">
<b><a href="https://www.marqo.ai">Website</a> | <a href="https://docs.marqo.ai">Documentation</a> | <a href="https://demo.marqo.ai">Demos</a> | <a href="https://community.marqo.ai">Discourse</a>  | <a href="https://bit.ly/marqo-slack">Slack Community</a> | <a href="https://www.marqo.ai/cloud">Marqo Cloud</a>
</b>
</p>

<p align="center">
<a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
<a href="https://pypi.org/project/marqo/"><img src="https://img.shields.io/pypi/v/marqo?label=PyPI"></a>
<a href="https://github.com/marqo-ai/marqo/actions/workflows/unit_test_200gb_CI.yml"><img src="https://img.shields.io/github/actions/workflow/status/marqo-ai/marqo/unit_test_200gb_CI.yml?branch=mainline"></a>
<a align="center" href="https://bit.ly/marqo-slack"><img src="https://img.shields.io/badge/Slack-blueviolet?logo=slack&amp;logoColor=white"></a>

## Marqo

Marqo is more than a vector database, it's an end-to-end vector search engine for both text and images. Vector generation, storage and retrieval are handled out of the box through a single API. No need to bring your own embeddings. 
    
**Why Marqo?**

Vector similarity alone is not enough for vector search. Vector search requires more than a vector database - it also requires machine learning (ML) deployment and management, preprocessing and transformations of inputs as well as the ability to modify search behavior without retraining a model. Marqo contains all these pieces, enabling developers to build vector search into their application with minimal effort. A full list of features can be found [below](#-core-features).

**Why not X, Y, Z vector database?** 

Vector databases are specialized components for vector similarity and only service one component of a vector search system. They are “vectors in - vectors out”. They still require the production of vectors, management of the ML models, associated orchestration and processing of the inputs. Marqo makes this easy by being “documents in, documents out”. Preprocessing of text and images, embedding the content, storing meta-data and deployment of inference and storage is all taken care of by Marqo. 

**Quick start** 

Here is a code snippet for a minimal example of vector search with Marqo (see [Getting Started](#getting-started)):

1. Use docker to run Marqo:

```bash

docker rm -f marqo
docker pull marqoai/marqo:latest
docker run --name marqo -it -p 8882:8882 marqoai/marqo:latest

```

Note: If your `marqo` container keeps getting killed, this is most likely due to a lack of memory being allocated to Docker. Increasing the memory limit for Docker to at least 6GB (8GB recommended) in your Docker settings may fix the problem.

2. Install the Marqo client:

```bash
pip install marqo
```

3. Start indexing and searching! Let's look at a simple example below:

```python
import marqo

mq = marqo.Client(url='http://localhost:8882')

mq.create_index("my-first-index")

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
    }],
    tensor_fields=["Description"]
)

results = mq.index("my-first-index").search(
    q="What is the best outfit to wear on the moon?"
)

```

## ✨ Core Features

**🤖 State of the art embeddings**
- Use the latest machine learning models from PyTorch, Huggingface, OpenAI and more. 
- Start with a pre-configured model or bring your own.
- Built-in ONNX support and conversion for faster inference and higher throughput.
- CPU and GPU support.

**⚡ Performance**
- Embeddings stored in in-memory HNSW indexes, achieving cutting edge search speeds.
- Scale to hundred-million document indexes with horizontal index sharding.
- Async and non-blocking data upload and search.

**🌌 Documents-in-documents-out**
- Vector generation, storage, and retrieval are provided out of the box.
- Build search, entity resolution, and data exploration application with using your text and images.
- Build complex semantic queries by combining weighted search terms.
- Filter search results using Marqo’s query DSL.
- Store unstructured data and semi-structured metadata together in documents, using a range of supported datatypes like bools, ints and keywords.

**🍱 Managed cloud**
- Low latency optimised deployment of Marqo.
- Scale inference at the click of a button.
- High availability.
- 24/7 support.
- Access control.
- Learn more [here](https://www.marqo.ai/cloud).

## Integrations

Marqo is integrated into popular AI and data processing frameworks, with more on the way.

**💙 [Haystack](https://github.com/deepset-ai/haystack)**

Haystack is an open-source framework for building applications that make use of NLP technology such as LLMs, embedding models and more. This [integration](https://haystack.deepset.ai/integrations/marqo-document-store) allows you to use Marqo as your Document Store for Haystack pipelines such as retrieval-augmentation, question answering, document search and more.

**🛹 [Griptape](https://github.com/griptape-ai/griptape)**

Griptape enables safe and reliable deployment of LLM-based agents for enterprise applications, the MarqoVectorStoreDriver gives these agents access to scalable search with your own data. This integration lets you leverage open source or custom fine-tuned models through Marqo to deliver relevant results to your LLMs.

**🦜🔗 [Langchain](https://github.com/langchain-ai/langchain)**

This integration lets you leverage open source or custom fine tuned models through Marqo for LangChain applications with a vector search component. The Marqo vector store implementation can plug into existing chains such as the Retrieval QA and Conversational Retrieval QA.

**⋙ [Hamilton](https://github.com/DAGWorks-Inc/hamilton/)**

This integration lets you leverage open source or custom fine tuned models through Marqo for Hamilton LLM applications. 

## Learn more about Marqo
                                                                                                                                                       
| | |
| --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 📗 [Quick start](#Getting-started)| Build your first application with Marqo in under 5 minutes. |
| 🖼 [Marqo for image data](https://www.marqo.ai/blog/context-is-all-you-need-multimodal-vector-search-with-personalization) | Building advanced image search with Marqo. |
| 📚 [Marqo for text](https://www.marqo.ai/blog/how-i-used-marqo-to-create-a-multilingual-legal-databse-in-5-key-lines-of-code) | Building a multilingual database in Marqo. |
| 🔮 [Integrating Marqo with GPT](https://www.marqo.ai/blog/from-iron-manual-to-ironman-augmenting-gpt-with-marqo-for-fast-editable-memory-to-enable-context-aware-question-answering) | Making GPT a subject matter expert by using Marqo as a knowledge base. |
| 🎨 [ Marqo for Creative AI](https://www.marqo.ai/blog/combining-stable-diffusion-with-semantic-search-generating-and-categorising-100k-hot-dogs) | Combining stable diffusion with semantic search to generate and categorise 100k images of hotdogs. |
| 🔊 [Marqo and Speech Data](https://www.marqo.ai/blog/speech-processing) | Add diarisation and transcription to preprocess audio for Q&A with Marqo and ChatGPT. |
| 🚫 [Marqo for content moderation](https://www.marqo.ai/blog/refining-image-quality-and-eliminating-nsfw-content-with-marqo) | Building advanced image search with Marqo to find and remove content. |
| ☁️ [Getting started with Marqo Cloud](https://github.com/marqo-ai/getting_started_marqo_cloud) | Go through how to get set up and running with Marqo Cloud starting from your first time login through to building your first application with Marqo|
| 👗 [Marqo for e-commerce](https://github.com/marqo-ai/getting_started_marqo_cloud/blob/main/e-commerce-demo/README.md) | This project is a web application with frontend and backend using Python, Flask, ReactJS, and Typescript. The frontend is a ReactJS application that makes requests to the backend which is a Flask application. The backend makes requests to your Marqo cloud API.|
| 🤖 [Marqo chatbot](https://github.com/marqo-ai/getting_started_marqo_cloud/tree/main/chatbot-demo) | In this guide we will build a chat bot application using Marqo and OpenAI's ChatGPT API. We will start with an existing code base and then walk through how to customise the behaviour.|
| 🦾 [Features](#-Core-Features) | Marqo's core features. |




## Getting started


1. Marqo requires docker. To install Docker go to the [Docker Official website](https://docs.docker.com/get-docker/). Ensure that docker has at least 8GB memory and 50GB storage.

2. Use docker to run Marqo:

```bash

docker rm -f marqo
docker pull marqoai/marqo:latest
docker run --name marqo -it --privileged -p 8882:8882 --add-host host.docker.internal:host-gateway marqoai/marqo:latest

```

Note: If your `marqo` container keeps getting killed, this is most likely due to a lack of memory being allocated to Docker. Increasing the memory limit for Docker to at least 6GB (8GB recommended) in your Docker settings may fix the problem.

3. Install the Marqo client:

```bash
pip install marqo
```

4. Start indexing and searching! Let's look at a simple example below:

```python
import marqo

mq = marqo.Client(url='http://localhost:8882')

mq.create_index("my-first-index")

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
    }],
    tensor_fields=["Description"]
)

results = mq.index("my-first-index").search(
    q="What is the best outfit to wear on the moon?"
)

```

- `mq` is the client that wraps the `marqo` API.
- `create_index()` creates a new index with default settings. You have the option to specify what model to use. For example, `mq.create_index("my-first-index", model="hf/all_datasets_v4_MiniLM-L6")` will create an index with the default text model `hf/all_datasets_v4_MiniLM-L6`. Experimentation with different models is often required to achieve the best retrieval for your specific use case. Different models also offer a tradeoff between inference speed and relevancy. See [here](https://docs.marqo.ai/1.0.0/Models-Reference/dense_retrieval/) for the full list of models.
- `add_documents()` takes a list of documents, represented as python dicts for indexing.
- You can optionally set a document's ID with the special `_id` field. Otherwise, Marqo will generate one.

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
            '_highlights': [{
                'Description': 'The EMU is a spacesuit that provides environmental protection, '
                               'mobility, life support, and communications for astronauts'
            }],
            '_id': 'article_591',
            '_score': 0.61938936
        },
        {
            'Title': 'The Travels of Marco Polo',
            'Description': "A 13th-century travelogue describing Polo's travels",
            '_highlights': [{'Title': 'The Travels of Marco Polo'}],
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
    "My_Image": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png",
    "Description": "The hippopotamus, also called the common hippopotamus or river hippopotamus, is a large semiaquatic mammal native to sub-Saharan Africa",
    "_id": "hippo-facts"
}], tensor_fields=["My_Image"])

```

You can then search the image field using text.

```python

results = mq.index("my-multimodal-index").search('animal')

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

mq.create_index("my-weighted-query-index")

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
        }
    ],
    tensor_fields=["Description"]
)

# initially we ask for a type of communications device which is popular in the 21st century
query = {
    # a weighting of 1.1 gives this query slightly more importance
    "I need to buy a communications device, what should I get?": 1.1,
    # a weighting of 1 gives this query a neutral importance
    "Technology that became prevelant in the 21st century": 1.0
}

results = mq.index("my-weighted-query-index").search(q=query)

print("Query 1:")
pprint.pprint(results)

# now we ask for a type of communications which predates the 21st century
query = {
    # a weighting of 1 gives this query a neutral importance
    "I need to buy a communications device, what should I get?": 1.0,
    # a weighting of -1 gives this query a negation effect
    "Technology that became prevelant in the 21st century": -1.0
}

results = mq.index("my-weighted-query-index").search(q=query)

print("\nQuery 2:")
pprint.pprint(results)

```

### Creating and searching indexes with multimodal combination fields
Marqo lets you have indexes with multimodal combination fields. Multimodal combination fields can combine text and images into one field. This allows scoring of documents across the combined text and image fields together. It also allows for a single vector representation instead of needing many which saves on storage. The relative weighting of each component can be set per document.

The example below demonstrates this with retrieval of caption and image pairs using multiple types of queries.

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
            "caption": "An image of a passenger plane flying in front of the moon.",
            "image": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg",
        },
        {
            "Title": "Red Bus",
            "caption": "A red double decker London bus traveling to Aldwych",
            "image": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image4.jpg",
        },
        {
            "Title": "Horse Jumping",
            "caption": "A person riding a horse over a jump in a competition.",
            "image": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg",
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
                "image": 0.7
            }
        }
    },
    # We specify which fields to create vectors for. 
    # Note that captioned_image is treated as a single field.
    tensor_fields=["captioned_image"]
)

# Search this index with a simple text query
results = mq.index("my-first-multimodal-index").search(
    q="Give me some images of vehicles and modes of transport. I am especially interested in air travel and commercial aeroplanes."
)

print("Query 1:")
pprint.pprint(results)

# search the index with a query that uses weighted components
results = mq.index("my-first-multimodal-index").search(
    q={
        "What are some vehicles and modes of transport?": 1.0,
        "Aeroplanes and other things that fly": -1.0
    },
)
print("\nQuery 2:")
pprint.pprint(results)

results = mq.index("my-first-multimodal-index").search(
    q={"Animals of the Perissodactyla order": -1.0}
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

## Contributors

Marqo is a community project with the goal of making tensor search accessible to the wider developer community. We are glad that you are interested in helping out! Please read [this](./CONTRIBUTING.md) to get started.

## Dev set up

1. Create a virtual env ```python -m venv ./venv```.

2. Activate the virtual environment ```source ./venv/bin/activate```.

3. Install requirements from the requirements file: ```pip install -r requirements.txt```.

4. Run tests by running the tox file. CD into this dir and then run "tox".

5. If you update dependencies, make sure to delete the .tox dir and rerun.

## Merge instructions:

1. Run the full test suite (by using the command `tox` in this dir).

2. Create a pull request with an attached github issue.

## Support

- Ask questions and share your creations with the community on our [Discourse forum](https://community.marqo.ai).
- Join our [Slack community](https://bit.ly/marqo-slack) and chat with other community members about ideas.


