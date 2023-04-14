import marqo
import pprint
import pandas as pd

import marqo.tensor_search.tensor_search

mq = marqo.Client(url='http://localhost:8882') # Connection to Marqo Docker Container

dataset_path = "http://localhost:8222/" # Place your file path here (directory where http server is setup)

def load_index(index_name: str, number_data: int) -> None:
    try:
        shirt_data = pd.read_csv('clothing-dataset/images.csv').head(number_data)[['image','label','kids']].to_dict('records')
        
        # dataset came from this link: https://github.com/alexeygrigorev/clothing-dataset-small
        # the .csv file has the following headers:
        # image, sender_id, label, kids
        # (image name, id of the sender who sent the pictures from sender_id, what kind of clothing it is, whether or not the clothing is for kids)
        # Dataset Example:.
        # 4285fab0-751a-4b74-8e9b-43af05deee22,124,Not sure,False
        # 70045b01-b350-4918-be74-2f627290ad7a,95,Skirt,False
        

        for data in shirt_data:
            path = "http://host.docker.internal:8222/clothing-dataset/images/" + data['image'] + ".jpg"
            data['image'] = path
            
        settings = {
            "treat_urls_and_pointers_as_images":True,   # allows us to find an image file and index it
            "model":"ViT-B/16"
        }
            
        mq.create_index(index_name, **settings)

        mq.index(index_name).add_documents(shirt_data)

        print("Index successfully created.")

    except Exception as e:
        print("Index already exists.")

def delete_index(index_name: str):
    try:
        mq.index(index_name).delete()
        print("Index successfully deleted.")
    except Exception as e:
        print("Index does not exist.")


def delete_doc_from_index(index_name:str, doc_ids:list[str]):
    results = marqo.tensor_search.tensor_search.delete_documents(ids=doc_ids)
    return results

def search_index_text(index_name:str, query_text: str, search_method: str):
    results = mq.index(index_name).search(
        q=query_text,
        search_method=search_method,
    )
    
    # Marqo also has other features such as searhcing based on a specific attribute field and query fitlering
    # refer to the documentation on how these features work (https://marqo.pages.dev/)
    return results

def search_index_image(index_name:str, image_name: str):
    # make sure the image is located inside the directory in which the python http server is running

    image_path = "http://host.docker.internal:8222/" + image_name

    results = mq.index(index_name).search(image_path)
    
    return results

def get_index_stats(index_name: str) -> dict:
    results = mq.index(index_name).get_stats()
    return results



def main():
    print("Welcome to Marqo Demo!")
    while True:
        action = int(input('''
What would you like to do?
1) Create an Index
2) Delete an Index
3) Search from an Index
4) Show Index Stats
5) Delete a document from an Index
6) Quit

Action: '''))

        if action == 1:
            index_name = input("Index name: ")
            no_of_items = int(input("No. of items in dataset: "))

            load_index(index_name, no_of_items)
        elif action == 2:
            index_name = input("Index name: ")
            
            delete_index(index_name)
        elif action == 3:
            index_name = input("Index name: ")
            search_type = input("Search Type (Text, Image): ")

            if search_type == 'Text':
                search_mode = str(input("Search Mode: (Lexical, Tensor)"))
                query_text = str(input("Query Text: "))

                results = search_index_text(index_name, query_text, search_mode.upper())
                
                pprint.pprint(results)
            elif search_type == 'Image':
                image_name = str(input("Image name (include MIME type .jpg or .png): "))

                results = search_index_image(index_name, image_name)

                pprint.pprint(results)
            
        elif action == 4:
            index_name = input("Index name: ")
            get_index_stats(index_name)

        elif action == 5:
            index_name = input("Index name: ")
            no_of_docs = int(input("No. of documents to delete: "))
            doc_ids = []

            for i in range(no_of_docs):
                doc_id = input("Document ID: ")
                doc_ids.append(doc_id)

            delete_doc_from_index(index_name, doc_ids)

        else:
            print("Goodbye")
            break

main()
