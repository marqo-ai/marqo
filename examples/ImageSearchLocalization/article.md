# Image search with localization and open-vocabulary reranking using Marqo, yolox, CLIP and OWL-ViT

TL;DR: Here we show how image search can be evolved to add localization and re-ranking by leveraging Marqo, yolox, CLIP and OWL-ViT. Adding the extra dimension of localization can improve retrieval performance and enable new use cases for image search while also helping with explainability. Re-ranking with an open vocabulary detection model allows for even finer-grained localsiation. The first part of the article covers background information while the second part contains working code (also found [here](https://github.com/marqo-ai/marqo/blob/mainline/examples/ImageSearchLocalization/index_all_data.py)).

<p align="center">
  <img src="article/1 FxTYCI7KBP0Zjjvcab0Azw.gif"/>
</p>

## Introduction

Image search has come a long way. Originally if you wanted to search a collection of images you would use keyword based search across manually curated meta-data. Vector representations followed and provided an avenue for more direct ways to query the image content. This has developed even more with the advent of cross-modal models like CLIP that allow searching images with natural language. Here we show it can be evolved further using a modern search stack that adds localization and the ability to re-rank.

## Image Search
Popular modern image search (or image retrieval) is often based on embedding images into a latent space (e.g. by transforming images into vectors and tensors). A query is embedded into the same space while search results are found by finding the closest matching embedding and returning their corresponding images.

<p align="center">
  <img src="article/1 AZLGFdtTksvNFtnTGhpQ_Q.png"/>
</p>

This single-stage retrieval based on embeddings is the same one that has been popular in many natural language processing applications like information retrieval, question and answering and chatbots. In many of these applications the matching documents are not just presented as part of the results but the part of the text that is the best match is also highlighted. This highlighting is what we can bring to image search via localization.

<p align="center">
  <img src="article/1 YZ3i9BPKoH2YedCpWAi7BA.png"/>
</p>

## Image Search + Localisation

There are a number of ways to get localization in image search. There is a strong latency:relevancy trade off as more sophisticated methods take longer to process. However, there are two broad categories of localization - (1) heuristic - where a heuristic is used to obtain localization and, (2) model - where another ML model is used to provide the localization. The localization can also happen at the time of indexing ('index-time partitioning') or after an initial set of search results have been returned ("search-time localization"). The latter is akin to a second stage re-ranker from traditional two stage retrieval systems.

### Index-time Partitioning

<p align="center">
  <img src="article/1 i_cfOYloD77oKfN8Kl5LKw.gif"/>
</p>

Here we will explain index-time partitioning for localization. In the indexing step, the image is partitioned into patches. Each of these patches and the original image are embedded and then stored in an index. This has the advantage that the time penalty is (mostly) paid off when indexing instead of searching.
In the retrieval step the query is embedded and compared not just to the original image but to all the patches as well. This now allows the location of the sub-image to also be returned.

<p align="center">
  <img src="article/1 khwH2Qd9JUscD32PpU2oKw.png"/>
</p>

Variations of this approach can also be used to do 'augment-time-indexing'. Instead of the image being broken into sub-patches it is augmented any number of times using any number of operations. Each of these augmented images are then stored in the same way the sub-patches were.

### Heuristic partitioning methods
As explained above, one of the simplest ways to get localisation in image search is to partition the image into patches. This is done using a rule or heuristic to crop the image into other sub-images and store the embeddings for each of these patches. The simplest scheme for images is to split the image into an N x M equally sized patches and embed those. More sophisticated methods can be performed by other machine learning models like object detectors.

<p align="center">
  <img src="article/1 L2Aln_Wc6IyEmDqYdcKEvQ.png"/>
</p>

##S# Model based partitioning methods
For the model based approaches, ideally we want "important" or relevant parts of the image to be detected by a model and proposed as the sub-images. Different use cases will have different requirements but surprisingly we can do some pretty good stuff with pretty generic models. To achieve this we can exploit some properties of object detectors and attention (i.e. transformers)  based models.

<p align="center">
  <img src="article/1 VMk4UqhEcL0_47A6Yg48rw.png"/>
</p>

For example, in two-stage detectors like Faster-RCNN the first stage consists of a region-proposal netwrok (RPN). The RPN is used to propose regions of the image that contains objects of interest and happens before any fine-grained classification occurs. The RPN is trainable and can be used to propose possible interesting parts of an image. Alternatives are to simply use a fast lighter-weight detector like yolo and make the output boxes the proposed regions (ignoring class) or use "objectness" scores to rank the proposed (now class agnostic) boxes. Finally, other alternatives exist like models that output "saliency" maps which can be obtained from supervised learning or through self-supervised methods like DINO. DINO has the added benefit that since it is self-supervised, it makes fine-tuning on custom datasets simple and a way to provide domain specific localisation.

## Re-ranking
An alternative approach to single-stage retrieval is two-stage retrieval. Two-stage retrieval consists of an initial retrieval of documents which are then reranked by a localization model or heuristic. The first stage is where the initial candidate documents are retrieved and the second stage can re-rank (i.e. re-order) the results based on another model or heuristic.

<p align="center">
  <img src="article/1 gqR9VyNJiK6Cpb57UYSmtw.gif"/>
</p>

One of the reasons for this type of architecture is to strike a balance between speed and relevancy. For example, the first stage can trade off speed and relevancy to provide a fast initial selection while the re-ranker can then provide a better final ordering by using a different model. The re-ranker can be used to add additional diversity or context (e.g. personalisation) to the results ranking or to add other things like localization (for images or videos). The diagram above has an illustrated example of this - the first stage retrieval of images comes from dense embeddings (e.g. from CLIP) while the second-stage re-ranker re-orders them based on a second (different) model.

### Search-time localization as re-ranking
As we saw earlier, localization can be introduced by dividing the images into patches at index-time and then searching across the image and child patches. An alternative approach is to defer the localization to the second stage via a reranker.

<p align="center">
  <img src="article/1 B_0tNnrDERSdmPBGqrBUEw.png"/>
</p>

There are multiple ways to do this, for example, you could do what was done at indexing time and divide each image in the retrieved results. However, doing that on its own ignores the crucial thing that we have now - the query. If we blindly divide the images and try and then match the query to the patches the additional information from the query is not used as effectively as it could. Instead, the proposal mechanism can be conditioned on the query. From this the results can then be re-ordered, for example by using the score that comes from the proposed regions.

<p align="center">
  <img src="article/1 XNf6OvML7_yQHWwPXMJV2w.png"/>
</p>

Conditioning the proposals based on the query has its roots in tasks like visual question and answering. The way this differs from other object detection problems is that the output is no longer restricted to a fixed vocabulary of objects but can take free form queries ('open vocabulary'). One good candidate model for this is OWL-ViT (Vision Transformer for Open-World Localization). OWL-ViT is a zero-shot text-conditioned object detection model. OWL-ViT uses CLIP as its backbone, while a vision transformer and a causal language model are used for the visual and text features respectively. Open-vocabulary classification is enabled by replacing the classification output with the class-name embeddings obtained from the text model.

# Putting it all together

In the previous section it was explained how image search works in general and how localization can be incorporated at both index and search time. In this section a full example with working code will be used to demonstrate each of these things in practice.

## Image dataset
For this example we are using about 10,000 images of various everyday objects. Here are some example images:

<p align="center">
  <img src="article/1 z8VEz9X_Ye2CJ4E-WEPqMg.png"/>
</p>

We are going to index this dataset using a couple of different methods and then search with and without the localization based reranker.

## Starting Marqo

We will be using Marqo to do the image search with localization that was explained previously (full code is also here). To start Marqo run the following from your terminal (assuming a cuda compatible GPU):

```
docker run --name marqo -it --privileged -p 8882:8882 --gpus all --add-host host.docker.internal:host-gateway -e MARQO_MODELS_TO_PRELOAD='[]' marqoai/marqo:0.0.10
```

If no GPU is available remove the --gpus all flag from the above command.

## Preparing the documents

We can either use the s3 urls directly or you can download the images and use them locally (see here for details). For now we will use the urls directly and create the documents for indexing.

```python
import pandas as pd
import os

df = pd.read_csv('files.csv')
documents = [{"image_location":s3_uri, '_id':os.path.basename(s3_uri)} for s3_uri in df['s3_uri']]
```

## Indexing with localization
Now we have the document we are going to index them using no index-time localization, using DINO and using yolox. We setup the client and the base settings.

```python
from marqo import Client
client = Client()

# setup the settings so we can comapre the different methods
patch_methods = [None, "dino-v2", "yolox"]

settings = {
    "index_defaults": {
        "treat_urls_and_pointers_as_images": True,
        "image_preprocessing": {
            "patch_method": None
        },
        "model":"ViT-B/32",
        "normalize_embeddings":True,
    },
}
```

To use the different methods we change the method name. We will iterate through each method and index the images in a different index.

```python
for patch_method in patch_methods:

    index_name = f"visual_search-{str(patch_method).lower()}"     
    
    settings['index_defaults']['image_preprocessing']['patch_method'] = patch_method
   
    response = client.create_index(index_name, settings_dict=settings)
    
    # index the documents on the GPU using multiple processes
    response = client.index(index_name).add_documents(documents, device='cuda', 
                                server_batch_size=50, processes=2)
```

If no GPU is available, set device='cpu'. 

## Searching with localization

Now we will demonstrate how to use the two different methods to get localization in image search.

### Search using index time localization 
We can now perform some searches against our indexed data and see the localization.

```python
response = client.index(index_name).search("brocolli", device="cuda")
print(response['hits'][0])
```

We can see in the highlights field the coordinates of the bounding box that best matched the query.

```python
bbox = response['hits']['_highlights']['image_location']
print(bbox)
```

The top six results are shown below with their corresponding top bounding box highlight.

<p align="center">
  <img src="article/1 466FOKNrrrzFnszM7qVZyQ.png"/>
</p>

The method here uses a pre-trained yolox model to propose the bounding boxes at indexing time and each of the sub-images are indexed alongside the original. Some filtering and non-max suppression (NMS) is applied and the maximum number of proposals per image is capped at ten. The class agnostic scores are used for the NMS. We can see the results from another method which is named dino-v2.

<p align="center">
  <img src="article/1 GDZZbl072NLDV--5yC_UoQ.png"/>
</p>

Dino-v2 uses base transformer models from DINO which is a self supervised representation learning method. Apart from being used as a pre-training step the attention maps from these models tend to focus on object within the images. These attention maps can be used to determine the salient or important parts of the images. The nice thing about this method is it is self-supervised and does not require labels or bounding boxes. It is also amenable to fine-tuning on domain specific data to provide better localization for specific tasks. The difference between dino-v1 and dino-v2 is that the proposals for v2 are generated per attention map, while v1 uses a summed attention map. This means the v1 generates fewer proposals than v2 (and means less storage is required).

## Search using search time localization
As described earlier, the alternative way to get localization is to have an object detector acting as a reranker and localizer. In Marqo we can specify the model here for re-ranking. The re-ranking model is OWL-ViT. OWL-ViT is an open vocabulary object detector that generates proposals after conditioning with a text prompt (or query). This conditional localisation is ideal to use as a reranker since we have the query to condition the model with for localisation.

```python
response = client.index(index_name).search("brocolli", device="cuda", 
        searchable_attributes=['image_location'], reranker="owl/ViT-B/32")
print(response['hits'][0])
```

The localisation provided by the reranker does not require any index time localisation. It can even be used with lexical search which does not use any embeddings for the first stage retrieval. 
We can see in the highlights field the coordinates of the bounding box that best matched the query after reranking,

```python
bbox = response['hits']['_highlights']['image_location']
print(bbox)
```

and we can plot these results as well. The localisation is better here as the proposals are done in conjunction with the query.

<p align="center">
  <img src="article/1 nWO8ksPJ3sLZeY4EUiGUgQ.png"/>
</p>

# Conclusion
We have shown how using a two-stage retrieval system can enable multiple avenues for adding localisation to image search. We showed how yolox and DINO could be leveraged to provide index time localisation. OWL-ViT was shown as a second stage reranker that also provides localisation. The methods discussed allow for a variety of trade-offs, including speed and relevency. To see how many more applications like this can be built, check out Marqo!