import marqo
from pprint import pprint

mq = marqo.Client("http://localhost:8882")

####################################################
### STEP 1: Download Data
####################################################

# Download the data from [here](https://github.com/marqo-ai/marqo/tree/2.0.0/examples/ImageSearchGuide/data)
# store it in a data/ directory

#####################################################
### STEP 2. Start Marqo
#####################################################

# Follow the instructions here https://github.com/marqo-ai/marqo/tree/2.0.0
"""
docker rm -f marqo
docker pull marqoai/marqo:2.0.0
docker run --name marqo -it -p 8882:8882 --add-host host.docker.internal:host-gateway marqoai/marqo:2.0.0
"""

####################################################
### STEP 3: Index Data
####################################################

index_name = 'image-search-guide'

try:
    mq.index(index_name).delete()
except:
    pass

settings = {
    "model": "open_clip/ViT-B-32/laion2b_s34b_b79k",
    "treatUrlsAndPointersAsImages": True,
}

mq.create_index(index_name, settings_dict=settings)

####################################################
### STEP 4: Access Local Images
####################################################

import subprocess

local_dir = "./data/"
pid = subprocess.Popen(['python3', '-m', 'http.server', '8222', '--directory', local_dir], stdout=subprocess.DEVNULL,
                       stderr=subprocess.STDOUT)

import glob
import os

# Find all the local images
locators = glob.glob(local_dir + '*.jpg')

# Generate docker path for local images
docker_path = "http://host.docker.internal:8222/"
image_docker = [docker_path + os.path.basename(f) for f in locators]

print(image_docker)

"""
output:
['http://host.docker.internal:8222/image4.jpg',
'http://host.docker.internal:8222/image1.jpg',
'http://host.docker.internal:8222/image3.jpg',
'http://host.docker.internal:8222/image0.jpg',
'http://host.docker.internal:8222/image2.jpg']
"""

####################################################
### STEP 5: Add Images to the Index
####################################################

documents = [{"image_docker": image, "_id": str(idx)} for idx, image in enumerate(image_docker)]

print(documents)

res = mq.index(index_name).add_documents(
    documents, client_batch_size=1,
    tensor_fields=["image_docker"]
)

pprint(res)

####################################################
### STEP 6: Search using Marqo
####################################################

search_results = mq.index(index_name).search("A rider on a horse jumping over the barrier")
print(search_results)

####################################################
### STEP 7: Visualize the Output
####################################################

import requests
from PIL import Image
from IPython.display import display

fig_path = search_results["hits"][0]["image_docker"].replace(docker_path, local_dir)
display(Image.open(fig_path))