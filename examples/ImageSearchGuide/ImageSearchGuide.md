# How to implement text-to-image search on Marqo - in 5 lines of code

*A step-to-step guide on how to search images with text (Text-to-Image Search) using [Marqo](https://www.marqo.ai/).*

![An example to show how to search image using text.](asset/example.png)


[Marqo](https://www.marqo.ai/) is an open-source tensor-based search engine that supports multi-modal search. In this article, we will
introduce how to set up your own text-to-image search engine using marqo. The full code is available on Marqo's github [Github](imagesearchguide.ipynb).

## Set up

### Install marqo
In this article, we select 5 images from the [coco dataset](https://cocodataset.org/#home) as examples.
<p align="center">
  <img src="./data/image3.jpg" width="150" />
  <img src="./data/image2.jpg" width="150" /> 
  <img src="./data/image1.jpg" width="110" />
  <img src="./data/image4.jpg" width="100" /> 
  <img src="./data/image5.jpg" width="150" />
</p>

First, we need to run marqo in docker using the following command. This test is done on a x64 linux machine, for Mac users with M-series chips
please check [here](https://github.com/marqo-ai/marqo#m-series-mac-users).

```
docker rm -f marqo
docker pull marqoai/marqo:0.0.10
docker run --name marqo -it --privileged -p 8882:8882 --add-host host.docker.internal:host-gateway marqoai/marqo:0.0.10
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

### Download images

Now, you can download our examples images from [Github](./data). You should have the following directory diagram:

<p align="center">
 <img src ="./asset/directory_diagram.png"/>
</p>


Done, you have finished all the set-up, let do the real search!

## Search with marqo

### Create index

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
__Note__: To accomplish this multi-modal search task, we __MUST__ set `"treat_urls_and_pointers_as_imges": True` to enable the multi-modal search feature. As for the `model`, we need to 
select a model from [__CLIP families__](https://docs.marqo.ai/0.0.10/Models-Reference/dense_retrieval/) (`"ViT-L/14"` in this case).

### Access local images
Now, we need to add the images to the created index, which is a little tricky. Marqo is running in the docker, so it will not be able to access
the local images.
One solution is to upload all the images to Github and access them through urls. This is OK in this case as we only have 5 images. However, if we think big,
are you really going to upload and download 1 million images with a larger dataset? I guess the answer is __NO__, so here is the solution.

We can put the local images in a local server for easier access from marqo in dock by
```python
import subprocess
local_dir = "./data"
pid = subprocess.Popen(['python3', '-m', 'http.server', '8222', '--directory', local_dir], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
```

With this step, marqo can access you image easily using http request, we just need to tell marqo where the image is:
```python
import glob
import os

# Find all the local images
locators = glob.glob(local_dir + '*.jpg')

# Generate docker path for local images
docker_path = "http://localhost:8222/"
image_docker = [docker_path + os.path.basename(f) for f in locators]

print(image_docker)
```
```python
['http://localhost:8222/image4.jpg',
 'http://localhost:8222/image1.jpg',
 'http://localhost:8222/image3.jpg',
 'http://localhost:8222/image5.jpg',
 'http://localhost:8222/image2.jpg']
```

All the local image are on a local server now.

### Add images to index












### Conduct searching
Finally, we can do the search and see the returned the results:







## Final word
