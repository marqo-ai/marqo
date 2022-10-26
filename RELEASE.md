# Release 0.0.6

## New features
- Health check endpoint: `GET /health`. An endpoint that can be used to inspect the status of Marqo and Marqo's backend (Marqo-os)
- Marqo can be launched with environment variables that define limits around maximum number of fields per index, maximum document size and the maximum numnber of documents that can be retrieved.
- README translations: 
  - Chinese 🇨🇳 (by [@wanliAlex](https://github.com/wanliAlex))
  - French 🇫🇷 (by [@rym-oualha](https://github.com/rym-oualha))
  - Ukrainian 🇺🇦 (by [@dmyzlata](https://github.com/dmyzlata))

## Breaking API changes
The home `/` json response has been updated. If you have logic that reads the endpoint root, please update it. 

## Non-breaking data model changes
- Each text field just creates a top level Marqo-os text field, without any keywords 
- Very large fields get their tensor_facet keywords ignored, rather than Marqo-OS preventing the doc being indexed
- Tensor facets can no longer have _id as a filtering field

## Bug fixes and minor changes
- FastAPI single threaded concurrency
- Refactoring out old code
- Get documents by IDs and lexical search and no longer returns vectors if expose_facets isn't specified

## Caveats
- If a large request to add_documents or update_documents results in a document adding fields such that the index field limit is exceeded, the entire operation will fail (without resilience). Mitigate this sending `add_documents` and `update_documents` requests with smaller batches of documents. 
- Fixing asynchronous indexing means that large scale ingestion can overwhelm Marqo's resources (this dependent on the machine running Marqo). For ingesting large batches of data we recommend: 
  - server side batch size of 10
  - client side batch size of 10 - 50








## Special thanks to our community contributors:

[@rym-oualha](https://github.com/rym-oualha), [@dmyzlata](https://github.com/dmyzlata)

# Release 0.0.5
<!--SMALL BLURB ABOUT RELEASE-->
Added Open CLIP models and added features to the get document endpoint.

## New features
<!--NON BREAKING CHANGES GO HERE-->
- Added Open CLIP models ([#116](https://github.com/marqo-ai/marqo/pull/116)). 
Read about usage [here](https://marqo.pages.dev/Models-Reference/dense_retrieval/#open-clip)
- Added the ability to get multiple documents by ID 
([#122](https://github.com/marqo-ai/marqo/pull/122)). 
Read about usage [here](https://marqo.pages.dev/API-Reference/documents/#get-multiple-documents)
- Added the ability to get document tensor facets through the get document endpoint 
([#122](https://github.com/marqo-ai/marqo/pull/122)). 
Read about usage [here](https://marqo.pages.dev/API-Reference/documents/#example_2)

# Release 0.0.4

<!--SMALL BLURB ABOUT RELEASE-->
Adding the attributesToRetrieve to the search endpoint and added the update documents endpoints

## New features
<!--NON BREAKING CHANGES GO HERE-->
- Added the AttributesToRetrieve option to the search endpoint ([55e5ac6](https://github.com/marqo-ai/marqo/pull/103))
- Added the PUT documents endpoint ([ce1306a](https://github.com/marqo-ai/marqo/pull/117))
