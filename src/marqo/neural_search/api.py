from fastapi import FastAPI
from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from marqo.neural_search import neural_search
from marqo import config
import os

c = config.Config(url=f'https://admin:admin@{os.environ["OPENSEARCH_IP"]}:9200')

class SearchQuery(BaseModel):
    q: str
    index_name: str


class AddDocuments(BaseModel):
    docs: list
    index_name: str


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/search")
async def search(search_query: SearchQuery):
    return neural_search.search(
        config=c, text=search_query.q,
        index_name=search_query.index_name)


@app.post("/add-documents")
async def search(add_docs: AddDocuments):
    return neural_search.add_documents(
        config=c, docs=add_docs.docs,
        index_name=add_docs.index_name, auto_refresh=True
    )

# try these curl commands:
"""
curl -XPOST  http://localhost:8000/add-documents -H 'Content-type:application/json' -d '{
"docs": [
    {
        "Title": "Honey is a delictable food stuff",
        "Desc" : "some boring description"
    }, {
        "Title": "Space exploration",
        "Desc": "mooooon! Space!!!!"
    }],
"index_name": "my-irst-ix"
}'
"""

"""
curl -XPOST  http://localhost:8000/search -H 'Content-type:application/json' -d '{
    "q": "what do bears eat?",
    "index_name": "my-irst-ix"
}'
"""