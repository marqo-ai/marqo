import pandas as pd
import os

from marqo import Client


if __name__ == "__main__":

    #######################################################################
    ############        Install Marqo and the python client    ############
    #######################################################################
    
    # run the following from the terminal
    # see https://marqo.pages.dev/0.0.21/
    
    """
    docker pull marqoai/marqo:latest
    docker rm -f marqo
    docker run --name marqo -it --privileged -p 8882:8882 --add-host host.docker.internal:host-gateway marqoai/marqo:latest
    
    pip install marqo
    """
    
    #######################################################################
    ############        Read in the data                       ############
    #######################################################################
    
    N = 100 # the number of samples to use (full size is ~220k)
    
    filename = "https://marqo-overall-demo-assets.s3.us-west-2.amazonaws.com/ecommerce_meta_data.csv"
    data = pd.read_csv(filename, nrows=N)
    data['_id'] = data['s3_http']
    documents = data[['s3_http', '_id', 'price', 'blip_large_caption', 'aesthetic_score']].to_dict(orient='records')
    
    print("Finished reading data...")
    
    #######################################################################
    ############        Create the index                       ############
    #######################################################################
    
    # https://marqo.pages.dev/0.0.21/
    client = Client()
    
    # https://marqo.pages.dev/0.0.21/API-Reference/indexes/
    index_name = 'multimodal'
    settings = {
            "index_defaults": {
                "treat_urls_and_pointers_as_images": True,
                "model": "open_clip/ViT-L-14/laion2b_s32b_b82k",
                "normalize_embeddings": True,
            },
        }
    
    # if the index already exists it will cause an error
    res = client.create_index(index_name, settings_dict=settings)
    print("Finished creating index...")

    #######################################################################
    ############        Index the data (image only)            ############
    #######################################################################
    
    # https://marqo.pages.dev/0.0.21/API-Reference/documents/
    device = 'cpu' # change to 'cuda' if GPU is available 
    res = client.index(index_name).add_documents(documents, client_batch_size=64, tensor_fields=["s3_http"], device=device)

    print("Finished indexing data...")

    #######################################################################
    ############        Search                                 ############
    #######################################################################
    
    # https://marqo.pages.dev/0.0.21/API-Reference/search/
    query = "green shirt"
    res = client.index(index_name).search(query, searchable_attributes=['s3_http'], device=device, limit=10)
    
    
    #######################################################################
    ############        Searching as prompting                 ############
    #######################################################################

    query = "cozy sweater, xmas, festive, holidays"    
    res = client.index(index_name).search(query, searchable_attributes=['s3_http'], device=device, limit=10)

    #######################################################################
    ############        Searching with semantic filters        ############
    #######################################################################
    
    # https://marqo.pages.dev/0.0.21/API-Reference/search/#query-q
    query = {"green shirt":1.0, "short sleeves":1.0}
    res = client.index(index_name).search(query, searchable_attributes=['s3_http'], device=device, limit=10)

    
    #######################################################################
    ############        Searching with negation                ############
    #######################################################################

    query = {"green shirt":1.0, "short sleeves":1.0, "buttons":-1.0}
    res = client.index(index_name).search(query, searchable_attributes=['s3_http'], device=device, limit=10)

    #######################################################################
    ############        Searching with images                  ############
    #######################################################################

    image_context_url = "https://marqo-overall-demo-assets.s3.us-west-2.amazonaws.com/ecommerce/red_backpack.jpg"
    query = {image_context_url:1.0}
    res = client.index(index_name).search(query, searchable_attributes=['s3_http'], device=device, limit=10)
    
    #######################################################################
    ############        Searching with multi-modal queries     ############
    #######################################################################

    # skateboard
    image_context_url = "https://marqo-overall-demo-assets.s3.us-west-2.amazonaws.com/ecommerce/71iPk9lfhML._SL1500_.jpg"

    query = {"backpack":1.0, image_context_url:1.0}
    res = client.index(index_name).search(query, searchable_attributes=['s3_http'], device=device, limit=10)

    # trees/hiking
    image_context_url = "https://marqo-overall-demo-assets.s3.us-west-2.amazonaws.com/ecommerce/trees.jpg"

    query = {"backpack":1.0, image_context_url:1.0}
    res = client.index(index_name).search(query, searchable_attributes=['s3_http'], device=device, limit=10)

    #######################################################################
    ############        Searching with ranking                 ############
    #######################################################################

    query = {"yellow handbag":1.0}
    
    # https://marqo.pages.dev/0.0.21/API-Reference/search/#score-modifiers
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

    print("Finished searching data...")

    #######################################################################
    ############        Guided search with popular products    ############
    #######################################################################
    
    # we create another index to create a context vector
    # we create another index to create a context vector
    index_name_context = 'multimodal-context'
    settings = {
                "index_defaults": {
                    "treat_urls_and_pointers_as_images": True,
                    "model": "open_clip/ViT-L-14/laion2b_s32b_b82k",
                    "normalize_embeddings": True,
                },
            }

    res = client.create_index(index_name_context, settings_dict=settings)

    # https://marqo.pages.dev/0.0.21/Advanced-Usage/document_fields/#multimodal-combination-object
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

    # https://marqo.pages.dev/0.0.21/API-Reference/mappings/
    # define how we want to comnbined
    mappings1 = {"multimodal": {"type": "multimodal_combination",
                               "weights": {"top_1": 0.40,
                                           "top_2": 0.30,
                                           "top_3": 0.20,
                                           "top_4": 0.10,
                                          }}}

    # define how we want to comnbined
    mappings2 = {"multimodal": {"type": "multimodal_combination",
                               "weights": {"top_1": 0.25,
                                           "top_2": 0.25,
                                           "top_3": 0.25,
                                           "top_4": 0.25,
                                          }}}

    
    # index the document             
    res = client.index(index_name_context).add_documents([document1], tensor_fields=["multimodal"], device=device, mappings=mappings1)

    # index the other using a different mappings
    res = client.index(index_name_context).add_documents([document2], tensor_fields=["multimodal"], device=device, mappings=mappings2)

    # retrieve the embedding to use as a context for search
    indexed_documents = client.index(index_name_context).get_documents([document1['_id'], document2['_id']] , expose_facets=True)

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
    
    
    #######################################################################
    ############        Indexing as multi-modal objects        ############
    #######################################################################
    
    # we will create a new index for the multimodal objects
    index_name_mm_objects = 'multimodal-objects'
    settings = {
                "index_defaults": {
                    "treat_urls_and_pointers_as_images": True,
                    "model": "open_clip/ViT-L-14/laion2b_s32b_b82k",
                    "normalize_embeddings": True,
                },
            }

    res = client.create_index(index_name_mm_objects, settings_dict=settings) 
    print(res)
    
    # now create the multi-modal field in the documents
    for doc in documents:
        doc['multimodal'] = {
                            'blip_large_caption':doc['blip_large_caption'],
                            's3_http':doc['s3_http'],
                            }

    # define how we want to combine the fields
    mappings = {"multimodal": 
                             {"type": "multimodal_combination",
                              "weights": 
                                 {"blip_large_caption": 0.20,
                                   "s3_http": 0.80,
                                 }
                             }
                    }

    # now index
    res = client.index(index_name_mm_objects).add_documents(documents, client_batch_size=64, tensor_fields=["multimodal"], device=device, mappings=mappings)    

    # now search
    query = "red shawl"
    res = client.index(index_name_mm_objects).search(query, searchable_attributes=['multimodal'], device=device, limit=10)
