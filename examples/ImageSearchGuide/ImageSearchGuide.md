# Search Image with Text Using Marqo

*A step-to-step guide on how to search image with text (Text-to-Image Search) using [Marqo](https://www.marqo.ai/).*

![An example to show how to search image using text.](asset/example.png)


[Marqo](https://www.marqo.ai/) is a tensor-based search and analytics engine that supports multi-modal search. It can be seamlessly 
integrates with your applications, websites, and workflows. In this article, we will
introduce how to set up your own Text-to-Image search engine using marqo. All the codes are available at on our [Github](imagesearchguide.ipynb).

## Set up
In this article, we select 5 images from the [coco dataset](https://cocodataset.org/#home) as examples.
<p float="left">
  <img src="./data/image3.jpg" width="80" />
  <img src="./data/image2.jpg" width="80" /> 
  <img src="./data/image1.jpg" width="50" />
  <img src="./data/image4.jpg" width="50" /> 
  <img src="./data/image5.jpg" width="80" />
</p>

First, we need to run marqo in docker using the following command. This test is done on a x64 linux machine, for Mac users with M-series chips
please check [here](https://github.com/marqo-ai/marqo#m-series-mac-users).

```
docker rm -f marqo
docker pull marqoai/marqo:latest
docker run --name marqo -it --privileged -p 8882:8882 --add-host host.docker.internal:host-gateway marqoai/marqo:latest
```

Now, we can create a new environment and install the Marqo client by:
```
conda create -n marqo-client python=3.8
conda activate marqo-client

pip install marqo matplotlib
```
Now, open your python and check the installation is successful by:
```python
import marqo
mq = marqo.Client("http://localhost:8882")

mq.get_marqo()
```
and you should have the output as:
```python
{'message': 'Welcome to Marqo', 'version': '0.0.10'}
```
By the time this article is written, we are using marqo with version 0.0.10.

Now, you can download our examples images from [Github](./data). You should have the following directory diagram:
![directory_diagram](./asset/directory_diagram.png)

Done, you have finished all the set-up, let do the real search!

## Search with marqo

First, we need to create a marqo index that provides you the access to all the necessary operations, e.g., indexing, searching. We also provide
necessary settings based on hardware devices.

```python
index_name = 'image-search-guide'

settings = {
        "model": "ViT-L/14",
        "treat_urls_and_pointers_as_images": True,
        }

mq.create_index(index_name, **settings)
```
__Note__: We __MUST__ set `"treat_urls_and_pointers_as_imges": True` to enable the multi-modal search feature. As for the model, we need to 
select a model from CLIP families to 






Now, we need to add the images to the created index, which is a little tricky. 






Finally, we can do the search and see the returned the results:







## Fnal word