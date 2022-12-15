# Release 0.0.10

## New features 
- Generic model support (https://github.com/marqo-ai/marqo/pull/179). Create an index with your favourite SBERT-type models from HuggingFace! Read about usage [here](https://marqo.pages.dev/0.0.10/Models-Reference/dense_retrieval/#generic-models)
- Visual search update 2. (https://github.com/marqo-ai/marqo/pull/214). Search-time image reranking and open-vocabulary localization, based on users' queries, is now available with the Owl-ViT model. **Locate the part of the image corresponding to your query!** 
- Visual search update 1. (https://github.com/marqo-ai/marqo/pull/214). Better image patching. In addition to faster-rcnn, you can now use yolox or attention based (DINO) region proposal as a patching method at indexing time. This allows localization as the sub patches of the image can be searched.

Check out [this article](https://medium.com/@jesse_894/image-search-with-localization-and-open-vocabulary-reranking-using-marqo-yolox-clip-and-owl-vit-9c636350bf66) about how this update makes image search awesome.

## Bug fixes
- Fixed imports and outdated Python client usage in Wikipedia demo (https://github.com/marqo-ai/marqo/pull/216) 

## Contributor shout-outs
- Thank you to [@georgewritescode](https://github.com/georgewritescode) for debugging and updating the Wikipedia demo
- Thank you to our 1.8k stargazers and 60+ forkers!


# Release 0.0.9
## Optimisations 
- Set k to limit to for Marqo-os search queries (https://github.com/marqo-ai/marqo/pull/219)
- Reduced the amount of metadata returned from Marqo-os, on searches (https://github.com/marqo-ai/marqo/pull/218)

## Non-breaking data model changes
- Set default kNN m value to 16 (https://github.com/marqo-ai/marqo/pull/222)

## Bug fixes
- Better error messages when downloading an image fails (https://github.com/marqo-ai/marqo/pull/198)
- Bug where filtering wouldn't work on fields with spaces (https://github.com/marqo-ai/marqo/pull/213), resolving https://github.com/marqo-ai/marqo/issues/115


# Release 0.0.8

## New features
- Get indexes endpoint: `GET /indexes` ([#181](https://github.com/marqo-ai/marqo/pull/181)). Use this endpoint to inspect
existing Marqo indexes. 
Read about usage [here](https://docs.marqo.ai/API-Reference/indexes/#list-indexes).
- Non-tensor fields([#161](https://github.com/marqo-ai/marqo/pull/161)). 
During the indexing phase, mark fields as non-tensor to prevent tensors being created for them. 
This helps speed up indexing and reduce storage for fields where keyword search is good enough. For example: email, name 
and categorical fields. These fields can still be used for filtering. 
Read about usage [here](https://docs.marqo.ai/API-Reference/documents/#query-parameters).
- Configurable preloaded models([#155](https://github.com/marqo-ai/marqo/pull/155)).
Specify which machine learning model to load as Marqo starts. This prevents a delay during initial search and index commands after 
Marqo starts. Read about usage [here](https://docs.marqo.ai/Advanced-Usage/configuration/#preloading-models).
- New [example](https://github.com/marqo-ai/marqo/tree/mainline/examples/GPT3NewsSummary) 
and [article](https://medium.com/creator-fund/building-search-engines-that-think-like-humans-e019e6fb6389): 
use Marqo to provide context for up-to-date GPT3 news summary generation 
([#171](https://github.com/marqo-ai/marqo/pull/171), [#174](https://github.com/marqo-ai/marqo/pull/174)).
Special thanks to [@iain-mackie](https://github.com/iain-mackie) for this example. 

## Bug fixes and minor changes
- Updated developer guide ([#164](https://github.com/marqo-ai/marqo/pull/164))
- Updated requirements which prevented Marqo being built as an arm64 image ([#173](https://github.com/marqo-ai/marqo/pull/173))
- Backend updated to use marqo-os:0.0.3 ([#183](https://github.com/marqo-ai/marqo/pull/183))
- Default request timeout has been increased from 2 to 75 seconds ([#184](https://github.com/marqo-ai/marqo/pull/184))

## Contributor shout-outs
- For work on the GPT3 news summary generation example: [@iain-mackie](https://github.com/iain-mackie)
- For contributing the non-tensor fields feature: [@jeadie](https://github.com/jeadie)
- Thank you to our users who raise issues and give us valuable feeback  
- Thank you to our 1.4k+ star gazers and 50+ forkers!

# Release 0.0.7

## Bug fixes and minor changes
- 429 (too many request errors) are propagated from Marqo-os to the user properly ([#150](https://github.com/marqo-ai/marqo/pull/150))

# Release 0.0.6

## New features
- Health check endpoint: `GET /health`. An endpoint that can be used to inspect the status of Marqo and Marqo's backend (Marqo-os) 
([#128](https://github.com/marqo-ai/marqo/pull/128)). Read about usage [here](https://docs.marqo.ai/API-Reference/health/).
- Marqo can be launched with environment variables that define limits around maximum number of fields per index, maximum document size and the maximum number of documents that can be retrieved 
([#135](https://github.com/marqo-ai/marqo/pull/135)). Read about usage [here](https://docs.marqo.ai/Advanced-Usage/configuration/).
- README translations: 
  - Chinese 🇨🇳 (by [@wanliAlex](https://github.com/wanliAlex), [#133](https://github.com/marqo-ai/marqo/pull/133))
  - Polish 🇵🇱 (by [@MichalLuck](https://github.com/MichalLuck), [#136](https://github.com/marqo-ai/marqo/pull/136))
  - Ukrainian 🇺🇦 (by [@dmyzlata](https://github.com/dmyzlata), [#138](https://github.com/marqo-ai/marqo/pull/138))
  - French 🇫🇷 (by [@rym-oualha](https://github.com/rym-oualha), [#147](https://github.com/marqo-ai/marqo/pull/147))

## Breaking API changes
- The home `/` json response has been updated. If you have logic that reads the endpoint root, please update it ([#128](https://github.com/marqo-ai/marqo/pull/128)). 
- The Python client's `add_documents()` and `update_documents()` `batch_size` parameter has been replaced by `server_batch_size` and `client_batch_size` parameters 
([py-marqo#27](https://github.com/marqo-ai/py-marqo/pull/27)), ([py-marqo#28](https://github.com/marqo-ai/py-marqo/pull/28))

## Non-breaking data model changes
- Each text field just creates a top level Marqo-os text field, without any keywords 
([#135](https://github.com/marqo-ai/marqo/pull/135))
- Very large fields get their tensor_facet keywords ignored, rather than Marqo-OS preventing the doc being indexed
([#135](https://github.com/marqo-ai/marqo/pull/135))
- Tensor facets can no longer have _id as a filtering field
([#135](https://github.com/marqo-ai/marqo/pull/135))

## Bug fixes and minor changes
- FastAPI runs with better concurrency ([#128](https://github.com/marqo-ai/marqo/pull/128))
- Get documents by IDs and lexical search and no longer returns vectors if expose_facets isn't specified
- Fixed batching bug in Python client
([py-marqo#28](https://github.com/marqo-ai/py-marqo/pull/28))

## Caveats
- If a large request to add_documents or update_documents results in a document adding fields such that the index field limit is exceeded, the entire operation will fail (without resilience). Mitigate this sending `add_documents` and `update_documents` requests with smaller batches of documents. 
- For optimal indexing of large volumes of images, we recommend that the images are hosted on the same region and cloud provider as Marqo.

## Contributor shout-outs

- For their translation work: [@rym-oualha](https://github.com/rym-oualha), [@dmyzlata](https://github.com/dmyzlata), [@wanliAlex](https://github.com/wanliAlex), [@dmyzlata](https://github.com/dmyzlata), [@MichalLuck](https://github.com/MichalLuck)
- For raising issues and helping with READMEs: [@kdewald](https://github.com/kdewald), [@llermaly](https://github.com/llermaly), [@namit343](https://github.com/namit343)
- Thank you to our 900+ star gazers and 30+ forkers


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
