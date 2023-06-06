# “Context is all you need” - multi-modal vector search with personalization 

*TL:DR We show how both text and images can be used for multi-modal search. This allows for multi-part queries, multi-modal queries, searching via prompting, promoting/suppressing content by themes, per query curation and personalization. Additionally, we show how other query independneet signals can be used to rank documents in addition to similairty. These features allow the creation and curation of high-quality search experiences, particularly for e-commerce and image heavy applications.*

<p align="center">
  <img src="assets/backpack.gif"/>
</p>
<p align="center">
    <em>An example of multi-modal search that uses multi-modal queries to curate search results based on text and images.</em>
</p>

## 1. Introduction

Often the items we want to search over contain more than just text. For example, they may also contain images or videos. These modalities other than text will often contain a wealth of information that is not captured by the text. By incorporating these other modalities into search, the relevancy of results can be improved as well as unlocking new ways to search. Examples of multi-modal search include domains like fashion and e-commerce which will have title, description as well as multiple images displaying the item. This data can also help disambiguate the subject of the image - for example if there are multiple items present like pants and a top, the text can provide the necessary context to identify the correct subject. The information contained in these data across modalities is complementary and rich. 

<p align="center">
  <img src="assets/example.png"/>
</p>
<p align="center">
    <em>An example of multi-modal "document" that contain both images and text.</em>
</p>

### 1.1 Multi-modal Search

Multi-modal search is search that operates over multiple modalities. We can think of two ways of doing multi-modal search, using multi-modal queries and multi-modal documents. For the former, the query itself may contain a combination of text and images  and the latter the document is made up of a combination of text and images.  For clarity we will stick to two modalities for now, text and images but the concepts are not restricted to just those and can be extended to video or audio (for example).

<p align="center">
  <img src="assets/stripes.gif"/>
</p>
<p align="center">
    <em>An example of multi-modal search using images and text to refine a search.</em>
</p>

### 1.2 Benefits

There are numerous benefits to this multi-modal approach. For example:

- The mulit-modal representations of documents allows for utilizing not just text or images or a combination of these. This allows complementary information to be captured that is not present in either modality.
- Using multi-modal representations allows for updatable and editable meta data for documents without re-training a model or re-indexing large amounts of data.
- Relevance feedback can be easily incorporated (in natural language) at a document level to improve or modify results.
- Curating queries with additional context allows for personalization and curation of results on a per query basis without additional models or fine-tuning.
- Curation can be performed in natural language.
- Business logic can be incorporated into the search using natural language.

## 2. Multi-modal Search in practice

In this section we will walk through a number of ways multi-modal search can be used to improve and curate results.

### 2.1 Multi-modal Queries

Multi-modal queries are queries that are made up of multiple components and/or multiple modalities. The benefit is that it effectively allows us to modify the scoring function for the approximate-knn to take into account additional similarities - for example, across multiple images or text and images.  The similairty scoring will now be against a weighted collection of items rather than a single piece of text. This allows finer grained curation of search results than by using a single part query alone.  We have seen previous examples of this earlier in the article already where both images and text are used to curate the search.

Shown below is an example of this where the query has multiple components. The first query is for an item while the second query is used to further condition the results. This acts as a “soft” or “semantic” filter. 

```python
query = {"green shirt":1.0, "short sleeves":1.0}
```

This multi-part query can be understood to be a form of manual [query expansion](https://en.wikipedia.org/wiki/Query_expansion). The animation below illustrates how the query can be used to modify search results.

<p align="center">
  <img src="assets/shirt1.gif"/>
</p>
<p align="center">
    <em>An example of multi-modal search two text queries to further refine the search.</em>
</p>


### 2.2 Negation

In the previous examples we saw how multiple queries can be used to condition the search. In those examples, the terms were being added with a positive weighting. Another way to utilise these queries is to use negative weighting terms to move away from particualr terms or concepts. Below is an example of a query with an additional negative term:

```python
query = {"green shirt":1.0, "short sleeves":1.0, "buttons":-1.0}
```

Now the search results are also moving away from the `buttons` while being drawn to the `green shirt` and `short sleeves`.

<p align="center">
  <img src="assets/shirt2.gif"/>
</p>
<p align="center">
    <em>An example of multi-modal search using negation to avoid particualr concepts - `buttons` in this case.</em>
</p>


### 2.2 Excluding low quality images

Negation can help avoid particular things when returning results, like low-quality images or ones with artifacts. Avoiding things like low-quality images or [NSFW content](https://www.marqo.ai/blog/refining-image-quality-and-eliminating-nsfw-content-with-marqo) can be easily described using natural language as seen in the exanmple query below:

```python
query = {"yellow handbag":1.0, "lowres, blurry, low quality":-1.1}
```

In the example below the initial results contain three low-quality images. These are denoted by a red mark for clarity and the poor image quality can be seen by the strong banding in the background of these images. 

<p align="center">
  <img src="assets/handbag1.gif"/>
</p>
<p align="center">
    <em>An example of multi-modal search using negation to avoid lowe quality images. The low-quality images are denoted by a red dot next to them. </em>
</p>

An alternative use is to use the same query to clean up existing data by using a positive weight. 

### 2.3 Searching with images

In the earlier examples we have seen how searching can be performed using weighted combinations of images and text. Searching with images alone can also be performed to utilize image similarity to find similar looking items.

```python
query = {image_url:1.0}
```

It can also be easily extended in the same way as with text to include multiple multi-modal terms.

```python
query = {image_url:1.0, "RED":1.0}
```

<p align="center">
  <img src="assets/backpack1.gif"/>
</p>


### 2.4 Conditional search with popular or liked items

Another way to utilize the multi-modal queries is for things like query expansion. In this example, each query will be expanded by adding additional variations of the query. However, in this case we will be not using variations on the query itself, but instead pre-compute expansions for particular queries (i.e. head queries).

```python
query = {"backpack":1.0}                        query = {"backpack":1.0}
context_vector1 = [.1, ...,.-.8]                context_vector2 = [-.01, ...,.3]
```

<p align="center">
  <img src="assets/context.png"/>
</p>

<p align="center">
  <img src="assets/backpack2.gif"/>
</p>



### 2.5 Searching as prompting

An alternative method to constructing multi-part queries is to append specific characteristics or styles to the end of a query. For example, additional descriptors can be appended to a query to curate the results.

```python 
query = {"handbag, bold colors, vibrant":1.0}
```

<p align="center">
  <img src="assets/handbag2.gif"/>
</p>


```python
query = {"cozy sweater, xmas, festive, holidays":1.0}
```

<p align="center">
  <img src="assets/sweater1.gif"/>
</p>


### 2.7 Ranking with other signals

We can also rank with other signals, not just the similarity. For example document specific values can be used to multiply or bias the vector similarity score. This allows for things like previous sales or popularity to impact the ranking.  In the example below, we have calculated an [aesthetic score](https://github.com/LAION-AI/aesthetic-predictor) and we can bias the score using this document (but query independent) field.

```python
query = {"yellow handbag":1.0}
```

<p align="center">
  <img src="assets/handbag1.png"/>
</p>


```python
query = {"yellow handbag":1.0}
score_modifiers = { 
        "add_to_score": 
            [
              {"field_name": "aesthetic_score", "weight": 0.02}]
       }
```

<p align="center">
  <img src="assets/handbag2.png"/>
</p>


### 2.8 **Multi-modal Entities**

Multi modal entities or items are just that - representations that take into account multiple pieces of information. These can be images or text or some combination of both. Examples include using multiple display images for ecommerce. Using multiple images can aid retrieval and help disambiguating between the item for sale and other items in the images. If a multi-modal model like CLIP is used, then the different modalities can be used together as they live in the same latent space.

```python
document = {"combined_text_image": 
		         {
                "image1":"https://some_image1.png",
								"image2":"https://some_image2.png",
								"image3":"https://some_image3.png", 
								"title": "Fresh and Versatile: The Green Cotton T-Shirt for Effortless Style"
								"description": "Crafted from high-quality cotton fabric, this t-shirt offers a soft and breathable feel, ensuring all-day comfort."
							}
						}
```

<p align="center">
  <img src="assets/multim.png"/>
</p>


## 3. Detailed Example

In the next section we will demonstrate how all of the above concepts can be implemented using Marqo.

### **3.1 Dataset**

The dataset consists of ~220,000 e-commerce products with images, text and some meta-data. The items span many categories of items, from clothing and watches to bags, backpacks and wallets. Along with the images they also have an [aesthetic score](https://github.com/LAION-AI/aesthetic-predictor), caption, and price. We will use all these features in the following example. Some images from the dataset are below.

### **3.2 Installing Marqo**

The first thing to do is start [Marqo](https://github.com/marqo-ai/marqo). To start, we can run the following [docker command](https://marqo.pages.dev/0.0.21/) from a terminal (for M-series Mac users see [here](https://marqo.pages.dev/0.0.21/m1_mac_users/)).

```bash
docker pull marqoai/marqo:latest
docker rm -f marqo
docker run --name marqo -it --privileged -p 8882:8882 --add-host host.docker.internal:host-gateway marqoai/marqo:latest
```

The next step is to install the python client (a REST API is also [available](https://docs.marqo.ai/0.0.21/)).

```bash
pip install marqo
```

### **3.3 Loading the data**

The first step is to load the data. The images are hosted on s3 for easy access. We use a file that contains all the image pointers as well as the meta data for them (found [here](https://marqo-overall-demo-assets.s3.us-west-2.amazonaws.com/ecommerce_meta_data.csv)). 

```python
filename = "ecommerce_meta_data.csv"
data = pd.read_csv(filename)
data['_id'] = data['s3_http']
documents = data[['s3_http', '_id', 'price', 'blip_large_caption', 'aesthetic_score']].to_dict(orient='records')
```

###  **3.4 Create the index**

Now we have the data prepared, we can setup the index. We will use a ViT-L-14 from open clip as the model. This model is very good to start with. It is recommended to use a GPU (at least 4GB VRAM) otherwise a [smaller model](https://marqo.pages.dev/0.0.21/Models-Reference/dense_retrieval/#open-clip) can be used (although results may be worse).  

```python
client = Client()
    
index_name = 'multimodal'
settings = {
            "index_defaults": {
                "treat_urls_and_pointers_as_images": True,
                "model": "open_clip/ViT-L-14/laion2b_s32b_b82k",
                "normalize_embeddings": True,
            },
        }
    
 response = mq.create_index(index_name, settings_dict=settings)
```

###  **3.5 Add images to the index**

Now we can add images to the index which can then be searched over. We can also select the device we want to use and also which fields in the data to embed.

```python
device = 'cuda'
non_tensor_fields = ['_id', 'price', 'blip_large_caption', 'aesthetic_score']

res = client.index(index_name).add_documents(documents, client_batch_size=64, non_tensor_fields=non_tensor_fields, device=device)
```

###  **3.6 Searching**

Now the images are indexed, we can start searching.

```python
query = "green shirt"
res = client.index(index_name).search(query, searchable_attributes=['s3_http'], device=device, limit=10)
```

###  **3.7 Searching as prompting**

Like in the examples above, it is easy to do more specific searches by adopting a similar style to prompting. 

```python
query = "cozy sweater, xmas, festive, holidays"    
res = client.index(index_name).search(query, searchable_attributes=['s3_http'], device=device, limit=10)
```

###  **3.8 Searching with semantic filters**

```python
query = {"green shirt":1.0, "short sleeves":1.0}
res = client.index(index_name).search(query, searchable_attributes=['s3_http'], device=device, limit=10)
```

###  **3.9 Searching with negation**

Remove buttons from a long sleeve shirt examples

```python
query = {"green shirt":1.0, "short sleeves":1.0, "buttons":-1.0}
res = client.index(index_name).search(query, searchable_attributes=['s3_http'], device=device, limit=10)
```

###  **3.10 Searching with images**

```python
image_context_url = "https://marqo-overall-demo-assets.s3.us-west-2.amazonaws.com/ecommerce/red_backpack.jpg"
query = {image_context_url:1.0}
res = client.index(index_name).search(query, searchable_attributes=['s3_http'], device=device, limit=10)
```

###  **3.11 Searching with multi-modal queries** 

```python
# skateboard
image_context_url = "https://marqo-overall-demo-assets.s3.us-west-2.amazonaws.com/ecommerce/71iPk9lfhML._SL1500_.jpg"

query = {"backpack":1.0, image_context_url:1.0}
res = client.index(index_name).search(query, searchable_attributes=['s3_http'], device=device, limit=10)

# trees/hiking
image_context_url = "https://marqo-overall-demo-assets.s3.us-west-2.amazonaws.com/ecommerce/trees.jpg"

query = {"backpack":1.0, image_context_url:1.0}
res = client.index(index_name).search(query, searchable_attributes=['s3_http'], device=device, limit=10)
```

###  **3.12 Searching with ranking**

We can now extend the search to also include document specific values to boost the ranking of documents in addition to the vector similarity. 

```python
query = {"yellow handbag":1.0}
    
# we define the extra document specific data to use for ranking
# multiple fields can be used to multiply or add to the vector similairty score
score_modifiers = { 
        "add_to_score": 
            [
            {"field_name": "aesthetic_score", "weight": 0.02}]
        }
    
res = client.index(index_name).search(query, searchable_attributes=['s3_http'], device=device, limit=10, score_modifiers=score_modifiers)

# now get the aggregate aesthetic score
print(sum(r['aesthetic_score'] for r in res['hits']))
    
# and compare to the non ranking version
res = client.index(index_name).search(query, searchable_attributes=['s3_http'], device=device, limit=10)

print(sum(r['aesthetic_score'] for r in res['hits']))
```

###  **3.13 Searching with popular or liked products**

```python
# we create another index to create a context vector
index_name = 'multimodal-context'
settings = {
            "index_defaults": {
                "treat_urls_and_pointers_as_images": True,
                "model": "open_clip/ViT-L-14/laion2b_s32b_b82k",
                "normalize_embeddings": True,
            },
        }
    
response = client.create_index(index_name, settings_dict=settings)
    
# create the document that will be created from multiple images
document1 = {"_id":"1",
                "multimodal": 
                    {
                        "top_1":"https://marqo-overall-demo-assets.s3.us-west-2.amazonaws.com/ecommerce/blue_backpack.jpg", 
                        "top_2":"https://marqo-overall-demo-assets.s3.us-west-2.amazonaws.com/ecommerce/dark_backpack.jpeg", 
                        "top_3":'https://marqo-overall-demo-assets.s3.us-west-2.amazonaws.com/ecommerce/green+_backpack.jpg',
                        "top_4":"https://marqo-overall-demo-assets.s3.us-west-2.amazonaws.com/ecommerce/red_backpack.jpg"
                    }
               }

# create the document that will be created from multiple images
document2 = {"_id":"2",
                "multimodal": 
                    {
                        "top_1":"https://marqo-overall-demo-assets.s3.us-west-2.amazonaws.com/ecommerce/office_1.jpg", 
                        "top_2":"https://marqo-overall-demo-assets.s3.us-west-2.amazonaws.com/ecommerce/office2.webp", 
                        "top_3":'https://marqo-overall-demo-assets.s3.us-west-2.amazonaws.com/ecommerce/office_3.jpeg',
                        "top_4":"https://marqo-overall-demo-assets.s3.us-west-2.amazonaws.com/ecommerce/office_4.jpg"
                    }
               }

    
# define how we want to comnbined
mappings1 = {"multimodal": 
               {"type": "multimodal_combination",
                   "weights": {"top_1": 0.40,
                               "top_2": 0.30,
                               "top_3": 0.20,
                               "top_4": 0.10,
                            }}}

# define how we want to comnbined
mappings2 = {"multimodal": 
               {"type": "multimodal_combination",
                   "weights": {"top_1": 0.25,
                               "top_2": 0.25,
                               "top_3": 0.25,
                               "top_4": 0.25,
                            }}}

    
# index the document             
res = client.index(index_name).add_documents([document1], device=device, mappings=mappings1)
    
# index the other using a different mappings
res = client.index(index_name).add_documents([document2], device=device, mappings=mappings2)
    
# retrieve the embedding to use as a context for search
indexed_documents = client.index(index_name).get_documents([document1['_id'], document2['_id']] , expose_facets=True)
    
# get the embedding
context_vector1 = indexed_documents['results'][0]['_tensor_facets'][0]['_embedding']
context_vector2 = indexed_documents['results'][1]['_tensor_facets'][0]['_embedding']
    
# create the context for the search
context1 = {"tensor":
                [
                  {'vector':context_vector1, 'weight':0.50}                  
                ]
            }

# create the context for the search
context2 = {"tensor":
                [
                  {'vector':context_vector2, 'weight':0.50}                  
                ]
            }

# now search
query = {"backpack":1.0}
res1 = client.index(index_name).search(query, searchable_attributes=['s3_http'], device=device, limit=10, context=context1)
    
res2 = client.index(index_name).search(query, searchable_attributes=['s3_http'], device=device, limit=10, context=context2)
```

### 4. Conclusion

To summarise, we have shown how vector search can be easily modified to enable a number of useful


