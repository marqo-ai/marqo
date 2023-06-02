import pandas as pd
import os

from marqo import Client


if __name__ == "__main__":

    #######################################################################
    ############        Read in the data                       ############
    #######################################################################
    
    filename = "ecommerce_meta_data.csv"
    data = pd.read_csv(filename)
    data['_id'] = data['s3_http']
    documents = data[['s3_http', '_id', 'price', 'blip_large_caption', 'aesthetic_score']].to_dict(orient='records')
    
    #######################################################################
    ############        Create the index                       ############
    #######################################################################
    
    client = Client()
    
    index_name = 'multimodal'
    settings = {
            "index_defaults": {
                "treat_urls_and_pointers_as_images": True,
                "model": "open_clip/ViT-L-14/laion2b_s32b_b82k",
                "normalize_embeddings": True,
            },
        }
    
    res = client.create_index(index_name, settings_dict=settings)

    #######################################################################
    ############        Index the data (image only)            ############
    #######################################################################
    
    device = 'cuda'
    non_tensor_fields = ['_id', 'price', 'blip_large_caption', 'aesthetic_score']

    res = client.index(index_name).add_documents(documents, client_batch_size=64, non_tensor_fields=non_tensor_fields, device=device)

    
    #######################################################################
    ############        Search                                 ############
    #######################################################################

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

    #######################################################################
    ############        Guided search with popular products    ############
    #######################################################################
    
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
    res = client.index(index_name).search(query, searchable_attributes=['s3_http'], device=device, limit=10, context=context1)
    
    query = {"backpack":1.0}
    res = client.index(index_name).search(query, searchable_attributes=['s3_http'], device=device, limit=10, context=context2)
