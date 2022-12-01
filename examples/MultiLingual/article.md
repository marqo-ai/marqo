# How I used Marqo to create a multilingual legal database in 5 key lines of code
![](assets/robot_lawyer.png)
*A [machine learning transformer model](https://openai.com/dall-e-2/) 
from [OpenAI](https://openai.com/) 
generated this image. A transformer model will also power the following multilingual search solution.*


The European Union has to deal with a peculiar problem - it has 24 official languages across 27 countries and these countries must abide by EU law. Experts in EU law have the complex task of navigating legal material in multiple languages.

What if there was a system where a user (like a lawyer) could search through a database of documents in their preferred language, and get the closest matching document in another? What if this user wanted to give access to this database to a colleague that uses a different language?
In this article, we present a solution that can search across multiple languages using a multilingual legal database built using Marqo, an open source tensor search engine, in just 5 key lines of code.
## The dataset 

The MultiEURLEX dataset is a collection of 65 thousand laws in 23 EU languages. EU laws are published in all member languages. This means that we may come across the same law in multiple languages.

## Scope for this proof of concept
In the interest of time and for ease of replication, this proof-of-concept will be a database to store documents from two languages: Deutsch and English. 
We will also only use the dataset's validation splits with 5000 documents from each language. 
Note that the machine learning model that Marqo will be using, _stsb-xlm-r-multilingual_ (more about this model can be found 
[here](https://www.sbert.net/docs/pretrained_models.html#multi-lingual-models) and 
[here](https://huggingface.co/sentence-transformers/stsb-xlm-r-multilingual)) 
can handle many more languages 
[than just these two](https://metatext.io/models/sentence-transformers-stsb-xlm-r-multilingual). 

The solution was run on an _ml.g4dn.2xlarge_ AWS machine. This comes with a Nvidia T4 GPU. 
The GPU speeds up the Marqo machine learning model which processes our documents as we insert them. 
These AWS machines are very easy to set up as 
[SageMaker Jupyter Notebook instances](https://aws.amazon.com/pm/sagemaker/). 

## The solution
If we were to develop this on a traditional SQL database or search engine, we'd have to manually create a translation 
layer to process the queries, or link each document with handcrafted translations. 

An example of this would be to translate all the documents into English as they are stored. The search query would also 
be translated into English, and a keyword search would be performed using a technology like Elasticsearch. However this
is problematic as a translated sentence is a lossy approximation of the source language and it introduces a significant component (real-time translation) into the system. This results in poorer search relevancy, worse latency, and additional system complexity.  

Tensor search, the technology that powers Marqo, outperforms traditional keyword search methods.

First, we set up a Marqo instance on the machine, which has docker installed. Notice the `--gpus all` option. 
This allows Marqo to use GPUs it finds on the machine. If the machine you are using doesn't have GPUs, then remove this option from the command. 
```sh
docker rm -f marqo; 
docker run --name marqo -it --privileged -p 8882:8882 --gpus all --add-host host.docker.internal:host-gateway marqoai/marqo:latest
```
We use pip to install the Marqo client (`pip install marqo`) and the datasets python package (`pip install datasets`). 
We will use the `datasets` package from [Hugging Face](https://huggingface.co/docs/datasets/index)
to import the MultiEURLEX dataset.

Then, we start work on our Python script. We start by loading the the validation splits for the English and Deutsch datasets:
```python
from datasets import load_dataset 
dataset_en = load_dataset('multi_eurlex', 'en', split="validation")
dataset_de = load_dataset('multi_eurlex', 'de', split="validation")
```
We then import Marqo and set up the client. We tell the Marqo client to connect with the Marqo Docker container that we ran earlier.

```python
from marqo import Client
mq = Client("http://localhost:8882")    
```
Then, add a line telling Marqo to create the multilingual index: 
```python
mq.create_index(index_name='my-multilingual-index', model='stsb-xlm-r-multilingual')
```
Notice that here is where we tell Marqo what model to use. After this, we'll iterate through each dataset, indexing each document as we go. 
One small adjustment we'll make is to split up text of very long documents (of over 100k chars) to make it easier to index and search. 
At the end of each loop, we call the `add_documents()` function to insert the document:
```python
mq.index(index_name='my-multilingual-index').add_documents(
    device='cuda', auto_refresh=False,
    documents=[{
                    "_id": doc_id,
                    "language": lang,
                    'text': sub_doc,
                    'celex_id': doc['celex_id'],
                    'labels': str(doc['labels'])
                }]
)
```
Here we set the device argument as `"cuda"`. This tells Marqo to use the GPU it finds on the machine to index the document. 
If you don't have a GPU, remove this argument or set it to `"cpu"`. We encourage using a GPU as it will make the `add_documents` 
process significantly faster (our testing showed a 6–12x speed up). 

We also set the `auto_refresh` argument to `False`. When indexing large volumes of data we encourage you to set this to False, as it optimises the `add_documents` process. 

And that's the indexing process! Run the script to fill up the Marqo index with documents. It took us around 45 minutes 
with an AWS _ml.g4dn.2xlarge_ machine. 

### Searching the index
![GIF of a legal search interface, with results in multiple languages](assets/fishing_search.gif)

We'll define the following search function that sets some parameters for the call to Marqo:
```python
# pprint is an inbuilt python formatter package that prints data in a readable way
import pprint 

def search(query: str):
    result = mq.index(index_name='my-multilingual-index').search(q=query, searchable_attributes=["text"])
    for res in result["hits"]:
        pprint.pprint(res["_highlights"])
```
The first thing to notice is the call to the Marqo `search()` function. We set `searchable_attributes` to the `"text"` field. 
This is because this is the field that holds the content relevant for searching.

We could print out the result straight away, but it contains the full original documents. These can be huge. Instead, 
we'll just print out the highlights from each document. These highlights also show us what part of the document Marqo 
found most relevant to the search query. We do this by printing the `_highlights` attribute from each hit.

We search by passing a string query to the search function. For the search with query string:

`"Laws about the fishing industry"`

We get the following results as the top 2 highlights: 
```
{'text': 'Consequently, catch limits and fishing effort limits for the cod stocks in the Baltic Sea should be established in accordance with the rules laid down in Council Regulation (EC) No 1098/2007 of 18 '...

{'text': '(18)\n'
         'Bei der Nutzung der Fangmöglichkeiten ist geltendes Unionsrecht uneingeschränkt zu befolgen -\n'
         'HAT FOLGENDE VERORDNUNG ERLASSEN:\n'
         'TITEL I\n'
         'GELTUNGSBEREICH UND BEGRIFFSBESTIMMUNGEN\n'
         'Artikel 1\n'...
```
The second result is from a German document. Using Google Translate, the German document's first line translates to

`When using the fishing opportunities, applicable Union law to be strictly followed`

Using Google Translate to translate the original fishing law query string into Deutsch gives us:

`"Gesetze über die Fischereiindustrie"`

Searching with this string gives us similar results to the English query. The first result is an English document, with the same highlight as the English query. Marqo identifies both queries strings as having similar meaning. 

Because we added the language code as a property of each document, we can filter for certain languages. We add a filter string to the search query:
```python
mq.index(index_name='my-multilingual-index').search(
    q=query, 
    searchable_attributes=['text'],
    filter_string='language:en'
)
```
Searching with this filter for `"Gesetze über saubere Energie"` (Google translation of `"Laws about clean energy"`) yields only English language results. The top 3 results are:
```
The electricity and water consumptions of products subject to this Regulation should be made more efficient by applying existing… 

Products subject to this Regulation should be made more energy efficient by applying existing non-proprietary cost-effective…

The electricity consumption of products subject to this Regulation should be made more efficient by applying existing non-proprietary cost-effective technologies that can reduce the combined costs of purchasing and operating these products…
```
## Conclusion

Marqo is a tensor search engine that can be deployed in just 3 lines of code and solve search problems using the latest 
ML models from HuggingFace and OpenAI. In this article I showed how I used Marqo to quickly set up a multilingual legal database.

Marqo makes tensor search easy. Without needing to be a machine learning expert, you can use cutting edge machine 
learning models to create an unrivalled search experience with minimal code. Check out the full code for the demo 
[here](eu_legal.py). Check out (and contribute, if you can!) to our open source codebase [here](https://github.com/marqo-ai/marqo). 





