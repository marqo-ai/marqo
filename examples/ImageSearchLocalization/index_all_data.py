#####################################################
### STEP 0. Import and define any helper functions
#####################################################

from marqo import Client
import os
import pandas as pd 
from utils import download_file, extract_zip
import glob
import subprocess
#####################################################
### STEP 1. start Marqo
#####################################################

# Follow the instructions here https://github.com/marqo-ai/marqo

#####################################################
### STEP 2. Get the data for indexing
#####################################################


# this will pull directly from the s3 bucket if True, otherwise it will pull for local indexing
use_remote = False

data = pd.read_csv('files.csv', index_col=0)

local_dir = os.getcwd() + '/images/'
if not use_remote:
    zip_file_url = 'https://marqo-public-datasets.s3.us-east-2.amazonaws.com/demos/ImageSearchLocalisation/images.zip'
    zip_file = download_file(zip_file_url, local_dir=local_dir)
    extract_zip(zip_file, local_dir=local_dir)
    # alternatively you can download the files individually
    #downloaded = download_parallel(urls=data['s3_uri'].tolist(), local_dir=local_dir)
    locators = glob.glob(local_dir + '*.jpg')
    docker_path = 'http://host.docker.internal:8222/'

    # we start an image server for easier access from within docker
    pid = subprocess.Popen(['python3', '-m', 'http.server', '8222', '--directory', local_dir], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    locators = [docker_path + os.path.basename(f) for f in locators]
else:
    # now we create our documents for indexing - a list of python dicts with keys as field names
    locators = data['s3_uri']


documents = [{"image_location":s3_uri, '_id':os.path.basename(s3_uri)} for s3_uri in locators]

# if you have the images locally, see the instructions 
# here https://marqo.pages.dev/Advanced-Usage/images/ for the best ways to index 


#####################################################
### STEP 3. Create the index(s)
######################################################

client = Client()

# setup the settings so we can comapre the different methods
index_name_prefix = "visual-search"
patch_methods = ["dino/v1", "dino/v2", "frcnn", None, "yolox"]
model_name = "ViT-B/32"
n_processes = 3
batch_size = 50

# set this to false if you do not want to delete the previous index of the same name
delete_index = True

settings = {
    "index_defaults": {
        "treat_urls_and_pointers_as_images": True,
        "image_preprocessing": {
            "patch_method": None
        },
        "model":None,
        "normalize_embeddings":True,
    },
}

for patch_method in patch_methods:

    suffix = '' if patch_method is None else f"-{patch_method.replace('/','-')}"
    index_name = index_name_prefix + suffix
    
    # update the settings we want to use
    settings['index_defaults']['model'] = model_name
    settings['index_defaults']['image_preprocessing']['patch_method'] = patch_method

    # optionally delete the index if it exists
    if delete_index:
        try:
            client.delete_index(index_name)
        except:
            print("index does not exist, cannot delete")
    
    # create the index with our settings
    response = client.create_index(index_name, settings_dict=settings)


    response = client.index(index_name).add_documents(documents, device='cuda', 
                                server_batch_size=batch_size, processes=n_processes)




# #####################################################
# ### STEP 4. Search
# ######################################################

# response = client.index(index_name).search("house", reranker= "google/owlvit-base-patch32")

# rank = 0

# bbox = list(response['hits'][rank]['_highlights'].values())[0]
# local_file = files_map[response['hits'][rank]['_id']]
# image = Image.open(local_file)

# image_cropped = image.crop(bbox)

# image_cropped.show()