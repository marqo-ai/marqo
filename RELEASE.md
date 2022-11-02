# Release 0.0.7

## Bug fixes and minor changes
- 429 (too many request errors) are propagated from Marqo-os to the user properly

# Release 0.0.6

## New features
- Health check endpoint: `GET /health`. An endpoint that can be used to inspect the status of Marqo and Marqo's backend (Marqo-os). Read about usage [here](https://docs.marqo.ai/API-Reference/health/).
- Marqo can be launched with environment variables that define limits around maximum number of fields per index, maximum document size and the maximum numnber of documents that can be retrieved. Read about usage [here](https://docs.marqo.ai/Advanced-Usage/configuration/).
- README translations: 
  - Chinese 🇨🇳 (by [@wanliAlex](https://github.com/wanliAlex))
  - French 🇫🇷 (by [@rym-oualha](https://github.com/rym-oualha))
  - Ukrainian 🇺🇦 (by [@dmyzlata](https://github.com/dmyzlata))
  - Polish 🇵🇱 (by [@MichalLuck](https://github.com/MichalLuck))

## Breaking API changes
- The home `/` json response has been updated. If you have logic that reads the endpoint root, please update it. 
- The Python client's `add_documents()` and `update_documents()` `batch_size` parameter has been replaced by `server_batch_size` and `client_batch_size` parameters

## Non-breaking data model changes
- Each text field just creates a top level Marqo-os text field, without any keywords 
- Very large fields get their tensor_facet keywords ignored, rather than Marqo-OS preventing the doc being indexed
- Tensor facets can no longer have _id as a filtering field

## Bug fixes and minor changes
- FastAPI single threaded concurrency
- Refactoring out old code
- Get documents by IDs and lexical search and no longer returns vectors if expose_facets isn't specified
- Fixed batching bug in Python client

## Caveats
- If a large request to add_documents or update_documents results in a document adding fields such that the index field limit is exceeded, the entire operation will fail (without resilience). Mitigate this sending `add_documents` and `update_documents` requests with smaller batches of documents. 
- For optimal indexing of large volumes of images, we recommend that the images are hosted on the same region and cloud provider as Marqo.

## Contributor shout-outs

- For their translation work: [@rym-oualha](https://github.com/rym-oualha), [@dmyzlata](https://github.com/dmyzlata), [@wanliAlex](https://github.com/wanliAlex), [@dmyzlata](https://github.com/dmyzlata), [@MichalLuck](https://github.com/MichalLuck)
- For reporting bugs and requesting features: [@kdewald](https://github.com/kdewald), [@llermaly](https://github.com/llermaly)
- To our 900+ star gazers and 30+ forkers


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
