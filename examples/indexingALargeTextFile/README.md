## Indexing a Large Text File

In cases where you want queries to return only a specific part of a document it can make sense to deconstruct documents into components before indexing.

### Disambiguating the Term 'Document'

Before we continue with this example we must disambiguate the term document. In Marqo, a document referring to a thing that gets indexed, this may be an image, and image and text pair, multiple paragraphs of text, a sentence or some other combination of the above. 

For some use cases this can differ from the common usage of the word 'document'. This is the case for this example where a large 'document' (text file) is broken into sentence and each sentence is indexed as a document.

From here on, we will refer to the input text file as 'source material' and the entries that we put into the index as 'documents'.

### Document Size Limits

By default [marqo imposes a size limit of 100,000 bytes](https://docs.marqo.ai/latest/Advanced-Usage/configuration/). While this can be adjusted - often querying large documents is not as useful as querying parts of documents.

### Running the example

1. Clone the repository.
1. Run Marqo:
    ```
    docker rm -f marqo;docker run --name marqo -it --privileged -p 8882:8882 --add-host host.docker.internal:host-gateway marqoai/marqo:latest   
    ```    
    For more instructions refer to the <a href="https://github.com/marqo-ai/marqo#getting-started">getting started</a> guide.

2. Run the `indexing_a_large_text_file.py` script via the following command (Note it can take a bit of time to index depending on the computer):
    ```
    python3 indexing_a_large_text_file.py
    ```
