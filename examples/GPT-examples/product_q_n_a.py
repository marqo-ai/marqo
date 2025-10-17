from marqo import Client
import pandas as pd
import numpy as np

from langchain_openai import OpenAI
from langchain.docstore.document import Document
from langchain.chains import LLMChain

from dotenv import load_dotenv

from utilities import (
    load_data,
    extract_text_from_highlights,
    qna_prompt,
    predict_ce,
    get_sorted_inds
)

load_dotenv()

if __name__ == "__main__":

    #############################################################
    #       STEP 0: Install Marqo
    #############################################################

    # run the following docker commands from the terminal to start marqo
    # docker rm -f marqo
    # docker pull marqoai/marqo:2.0.0
    # docker run --name marqo -it -p 8882:8882 --add-host host.docker.internal:host-gateway marqoai/marqo:2.0.0


    #############################################################
    #       STEP 1: Setup Marqo
    #############################################################


    mq = Client()
    index_name = "iron-docs"

    # (optinally) delete if it already exists
    try:
        mq.index(index_name).delete()
    except:
        pass

    # we can set some specific settings for the index. if they are not provided, sensible defaults are used
    index_settings = {
        "model": "flax-sentence-embeddings/all_datasets_v4_MiniLM-L6",
        "normalizeEmbeddings": True,
        "textPreprocessing": {
            "splitLength": 3,
            "splitOverlap": 1,
            "splitMethod": "sentence"
        },
    }

    # create the index with custom settings
    mq.create_index(index_name, settings_dict=index_settings)

    #############################################################
    #       STEP 2: Load the data
    #############################################################

    df = load_data()

    # turn the data into a dict for indexing
    documents = df.to_dict(orient='records')

    #############################################################
    #       STEP 3: Index the data
    #############################################################

    # index the documents
    indexing = mq.index(index_name).add_documents(documents, tensor_fields=["cleaned_text"], client_batch_size=64)

    #############################################################
    #       STEP 4: Search the data
    #############################################################

    # try a generic search
    q = "what is the rated voltage"

    results = mq.index(index_name).search(q)
    print(results['hits'][0])

    #############################################################
    #       STEP 5: Make it chatty
    #############################################################

    highlights, texts = extract_text_from_highlights(results, token_limit=150)
    docs = [Document(page_content=f"Source [{ind}]:" + t) for ind, t in enumerate(texts)]
    llm = OpenAI(temperature=0.9)
    chain_qa = LLMChain(llm=llm, prompt=qna_prompt())
    llm_results = chain_qa.invoke({"summaries": docs, "question": results['query']}, return_only_outputs=True)
    print(llm_results['text'])

    #############################################################
    #       STEP 6: Score the references
    #############################################################

    score_threshold = 0.20
    top_k = 3
    scores = predict_ce(llm_results['text'], texts)
    inds = get_sorted_inds(scores)
    scores = scores.cpu().numpy()
    scores = [np.round(s[0], 2) for s in scores]
    references = [(str(np.round(scores[i], 2)), texts[i]) for i in inds[:top_k] if scores[i] > score_threshold]
    df_ref = pd.DataFrame(references, columns=['score', 'sources'])
    print(df_ref)
