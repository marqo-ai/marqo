"""The API entrypoint for Neural Search"""
from models.api_objects import SearchQuery, AddDocuments
from fastapi import FastAPI
from typing import Union
from fastapi import FastAPI
from marqo.neural_search import neural_search
from marqo import config
from typing import List, Dict
import os

c = config.Config(url=f'https://admin:admin@{os.environ["OPENSEARCH_IP"]}:9200')


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/indexes/{index_name}/search")
async def search(search_query: SearchQuery, index_name: str):
    return neural_search.search(
        config=c, text=search_query.q,
        index_name=index_name)


@app.post("/indexes/{index_name}/documents")
async def add_documents(docs: List[Dict], index_name: str, refresh: bool = True):
    """add_documents endpoint"""
    return neural_search.add_documents(
        config=c, docs=docs,
        index_name=index_name, auto_refresh=refresh
    )

# try these curl commands:
"""
curl -XPOST  'http://localhost:8000/indexes/my-irst-ix/documents?refresh=true' -H 'Content-type:application/json' -d '
[
    {
        "Title": "Honey is a delectable food stuff",
        "Desc" : "some boring description",
        "_id": "honey_facts_119"
    }, {
        "Title": "Space exploration",
        "Desc": "mooooon! Space!!!!",
        "_id": "moon_fact_145"
    }
]'
"""

"""
curl -XPOST  http://localhost:8000/indexes/my-irst-ix/search -H 'Content-type:application/json' -d '{
    "q": "what do bears eat?"
}'
"""