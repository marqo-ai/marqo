"""
1. cd into your Marqo repo
2. Run Marqo with your desired env vars:
```bash
docker rm -f marqo &&
  DOCKER_BUILDKIT=1 docker build . -t marqo_docker_0 &&
  docker run --name marqo --privileged -p 8882:8882 --add-host host.docker.internal:host-gateway \
  -e "MARQO_MAX_INDEX_FIELDS=123" -e "MARQO_MAX_DOC_BYTES=12345" -e "MARQO_MAX_RETRIEVABLE_DOCS=21" marqo_docker_0
```
3. In a Python terminal, running the following script:
"""
import marqo
from marqo.errors import MarqoWebError
mq = marqo.Client()

index_name = "my-tes-index-1"
max_fields_limit = 123

# TEST FIELD COUNT LIMIT
res_1 = mq.index(index_name).add_documents(
    documents=[{f"f{i}": "some content" for i in range(max_fields_limit)},
               {f"f{i}": "some content" for i in range(max_fields_limit // 2 + 1)}
               ], auto_refresh=True, 
)
assert not res_1['errors']
try:
    res_2 = mq.index(index_name).add_documents(
        documents=[{'fx': "blah"}], auto_refresh=True
    )
    raise AssertionError
except MarqoWebError as e:
    pass

# TEST TOO MANY DOC BYTES
max_doc_size = 12345

update_res = mq.index(index_name=index_name).add_documents(
    documents=[
            {"_id": "123", 'f1': "edf " * (max_doc_size // 4)},
            {"f1": "abc " * ((max_doc_size // 4) - 500)},
            {"_id": "456", "f1": "exc " * (max_doc_size // 4)},
          ],
    auto_refresh=True)
items = update_res['items']
assert update_res['errors']
assert 'error' in items[0] and 'error' in items[2]
assert 'doc_too_large' == items[0]['code'] and ('doc_too_large' == items[0]['code'])
assert items[1]['result'] == 'created'
assert 'error' not in items[1]

# TEST TOO MANY DOCS RETRIEVED
max_retrievable_docs = 21

docs = [
    {"f0": "some string", "_id": f"doc_{i}"}
    for i in range(100)
]
update_res = mq.index(index_name=index_name).add_documents(
    documents=docs,
    auto_refresh=True)

limit_search = mq.index(index_name=index_name).get_documents(
    document_ids=[docs[i]['_id'] for i in range(max_retrievable_docs)])
assert len(limit_search['results']) == max_retrievable_docs

try:
    oversized_search = mq.index(index_name=index_name).get_documents(
        document_ids=[docs[i]['_id'] for i in range(max_retrievable_docs + 1)])
    raise AssertionError
except MarqoWebError as e:
    pass

mq.delete_index(index_name)
print("Passed environment variable tests")

