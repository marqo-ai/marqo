import pandas as pd
import marqo
from pprint import pprint

mq = marqo.Client("http://localhost:8882")
device = 'cpu' # either 'cpu' or 'cuda' (only works on GPU enabled machines)

############################################################################
############################# Step 1. Enriching
############################################################################
"""
Here we ddemonstrate how to perform two different types of enriching operations. only images are currently supported.
1. attribute-extraction - generate binary labels for classifying if something is rpesent or not
2. question-answer  - open ended question answer that can produce free-form answers
"""

######################################
# Method 1. Attribute extraction #####
"""
Attribute extraction provides binary labels for particualr attributes    
"""

# these are the documents we want to enrich
# they must be a 
documents = [
    {
        "Description": "A photo of a hippo",
        "Image location": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png"
   },
   {
       "Description": "A photo of a hippo status",
       "Image location": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png"
   }
]

# now we can call the enrich endpoint on our documents
enriched_ae = mq.enrich(
    documents=documents,
  enrichment=
    {
                "task": "attribute-extraction",
                "to": ["Is_bathroom", "Is_Bedroom", "Is_Office", "Is_Yard", "Is_Hippo"],
                "kwargs": {
                    "attributes": [{"string": "Bathroom, Bedroom, Office, Yard, Hippo"}],
                    "image_field": {"document_field": "Image location"}
                },
            },
device=device)

pprint(enriched_ae['documents'])

######################################
# Method 2. Question-answer      #####
"""
question answer provides answers to open ended questions   
"""

enriched_qa = mq.enrich(
    documents=[
        {
            "Description": "A photo of a hippo",
            "Image location": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png"
        },
        {
           "Description": "A photo of a hippo status",
           "Image location": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png"
        }
    ],
    enrichment=
    {
        "task": "question-answer",
        "to": ["Astronaut count", "Hippo count"], 
        "kwargs": {
            "query": [
                {"string":"How many astronauts in this image?"},
                {"string":"How many Hippos in this image?"}],
            "image_field": {"document_field":"Image location"}
        },
    },
device=device)

############################################################################
############################# Step 2. Save to csv (optional)
############################################################################

# add them to a dataframe and write to a csv
df_ae = pd.DataFrame(enriched_ae['documents'])
df_ae.to_csv("data_ae.csv")


# add them to a dataframe and write to a csv
df_qa = pd.DataFrame(enriched_qa['documents'])
df_qa.to_csv("data_qa.csv")

############################################################################
############################# Step 3. Index them into marqo
############################################################################

settings = {
 "treat_urls_and_pointers_as_images":True,
 "model":"ViT-B/32" # the best performance is ViT-L/14 but check your RAM.
}

index_name = 'test_enrichment'

mq.create_index(index_name, **settings)

mq.index(index_name).add_documents(enriched_ae['documents'], device=device)

mq.index(index_name).search("hippo", filter_string="Is_Hippo:true")

mq.index(index_name).search("hippo", filter_string="Is_Hippo:true and Is_Bedroom:true")