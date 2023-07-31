# Release 1.1.0

## New features

- New field `numberOfVectors` in the `get_stats` response object (https://github.com/marqo-ai/marqo/pull/553). 
This field counts all vectors from all documents in a given index. See [here](https://docs.marqo.ai/1.1.0/API-Reference/stats/) for detailed usage.

- New per-index health check endpoint `GET /indexes/{index-name}/health` (https://github.com/marqo-ai/marqo/pull/552). 
This replaces the cluster-level health check endpoint, `GET /health`,
which is deprecated and will be removed in Marqo 2.0.0. See [here](https://docs.marqo.ai/1.0.0/API-Reference/health/) for detailed usage.

## Bug fixes and minor changes

- Improved image download validation and resource management (https://github.com/marqo-ai/marqo/pull/551). Image downloading in Marqo is more stable and resource-efficient now.

- Adding documents now returns an error when `tensorFields` is not specified explicitly (https://github.com/marqo-ai/marqo/pull/554). This prevents users accidentally creating unwanted tensor fields.

## Contributor shout-outs

- Thank you for the Vibrant discussion in our and [forum](https://community.marqo.ai/). 
We love hearing your questions and about your use cases.


# Release 1.0.0

## Breaking Changes

- New parameter `tensor_fields` will replace `non_tensor_fields` in the `add_documents` endpoint (https://github.com/marqo-ai/marqo/pull/538). Only fields in `tensor_fields` will have embeddings generated, offering more granular control over which fields are vectorised. See [here](https://docs.marqo.ai/1.0.0/API-Reference/documents/#add-or-replace-documents) for the full list of `add_documents` parameters and their usage. The `non_tensor_fields` parameter is deprecated and will be removed in a future release. Calls to `add_documents` with neither of these parameters specified will now fail.

- Multiple tensor field optimisation ([#530](https://github.com/marqo-ai/marqo/pull/530)). This optimisation results in faster and more stable searches across multiple tensor fields. Please note that indexed documents will now have a different internal document structure, so documents indexed with previous Marqo versions cannot be searched with this version, and vice versa.

- The `add_documents` endpoint's request body is now an object, with the list of documents under the `documents` key ([#535](https://github.com/marqo-ai/marqo/pull/535)). The query parameters `use_existing_tensors`, `image_download_headers`, `model_auth`, and `mappings` have been moved to the body as optional keys, and support for these parameters in the query string is deprecated. This change results in shorter URLs and better readability, as values for these parameters no longer need to be URL-encoded. See [here](https://docs.marqo.ai/1.0.0/API-Reference/documents/#add-or-replace-documents) for the new `add_documents` API usage. Backwards compatibility is supported at the moment but will be removed in a future release.

- Better validation for index creation with custom models (https://github.com/marqo-ai/marqo/pull/530). When creating an index with a `model` not in the registry, Marqo will check if `model_properties` is specified with a proper `dimension`, and raise an error if not. See [here](https://docs.marqo.ai/1.0.0/Models-Reference/bring_your_own_model) for a guide on using custom models. This validation is now done at index creation time, rather than at add documents or search time.

- Stricter `filter_string` syntax for `search` ([#530](https://github.com/marqo-ai/marqo/pull/530)). The `filter_string` parameter must have special Lucene characters escaped with a backslash (`\`) to filter as expected. This will affect filtering on field names or content that contains special characters. See [here](https://lucene.apache.org/core/2_9_4/queryparsersyntax.html) for more information on special characters and see [here](https://docs.marqo.ai/1.0.0/query_dsl) for a guide on using Marqo filter strings.

- Removed server-side batching (`batch_size` parameter) for the `add_documents` endpoint ([#527](https://github.com/marqo-ai/marqo/pull/527)). Instead, client-side batching is encouraged (use `client_batch_size` instead of `server_batch_size` in the python client).

## New Features
- Multi-field pagination ([#530](https://github.com/marqo-ai/marqo/pull/530)). The `offset` parameter in `search` can now be used to paginate through results spanning multiple `searchable_attributes`. This works for both `TENSOR` and `LEXICAL` search. See [here](https://docs.marqo.ai/1.0.0/API-Reference/search/#search-result-pagination) for a guide on pagination.
- Optimised default index configuration (https://github.com/marqo-ai/marqo/pull/540).

## Bug Fixes & Minor Changes
- Removed or updated all references to outdated features in the examples and the README (https://github.com/marqo-ai/marqo/pull/529).
- Enhanced bulk search test stability (https://github.com/marqo-ai/marqo/pull/544).

## Contributor shout-outs
- Thank you to our 3.2k stargazers!
- We've finally come to our first major release, Marqo 1.0.0! Thanks to all our users and contributors, new and old, for your feedback and support to help us reach this huge milestone. We're excited to continue building Marqo with you. Happy searching!


# Release 0.1.0

## New features
- Telemetry. Marqo now includes various timing metrics for the `search`, `bulk_search` and `add_documents` endpoints
when the query parameter `telemetry=True` is specified (https://github.com/marqo-ai/marqo/pull/506). The metrics will be
returned in the response body and provide a breakdown of latencies for various stages of the API call.
- Consolidate default device to CUDA when available (https://github.com/marqo-ai/marqo/pull/508). By default,
Marqo now uses CUDA devices for search and indexing if available.
See [here](https://docs.marqo.ai/0.1.0/API-Reference/search/#query-parameters) for more information. This helps ensure
you get the best indexing and search experience without having to explicitly add the device parameter to search and
add_documents calls.
- Model download integrity verification (https://github.com/marqo-ai/marqo/pull/502). Model files are validated and
removed if corrupted during download. This helps ensure that models are not loaded if they are corrupted.

## Breaking changes
- Remove deprecated `add_or_update_documents` endpoint (https://github.com/marqo-ai/marqo/pull/517).
- Disable automatic index creation. Marqo will no longer automatically create an index if it does not exist 
(https://github.com/marqo-ai/marqo/pull/516).
Attempting to add documents to a non-existent index will now result in an error. This helps provide more certainty about
the properties of the index you are adding documents to, and also helps prevent accidental indexing to the wrong index.
- Remove parallel indexing (https://github.com/marqo-ai/marqo/pull/523). Marqo no longer supports server-side parallel
indexing. This helps deliver a more stable and efficient indexing experience. Parallelisation can still be implemented
by the user.

## Bug fixes and minor changes
- Improve error messages (https://github.com/marqo-ai/marqo/pull/494, https://github.com/marqo-ai/marqo/pull/499).
- Improve API request validation (https://github.com/marqo-ai/marqo/pull/495).
- Add new multimodal search example (https://github.com/marqo-ai/marqo/pull/503).
- Remove autocast for CPU to speed up vectorisation on ARM64 machines (https://github.com/marqo-ai/marqo/pull/491).
- Enhance test stability (https://github.com/marqo-ai/marqo/pull/514).
- Ignore `.kibana` index (https://github.com/marqo-ai/marqo/pull/512).
- Improve handling of whitespace when indexing documents (https://github.com/marqo-ai/marqo/pull/521).
- Update CUDA version to 11.4.3 (https://github.com/marqo-ai/marqo/pull/525).

## Contributor shout-outs
- Thank you to our 3.1k stargazers!

# Release 0.0.21

## New features
- Load custom SBERT models from cloud storage with authentication (https://github.com/marqo-ai/marqo/pull/474). 
Marqo now supports fetching your fine-tuned public and private SBERT models from Hugging Face and AWS s3. Learn more about using your own SBERT model [here](https://docs.marqo.ai/0.0.21/Models-Reference/bring_your_own_model/#bring-your-own-hugging-face-sbert-models). For instructions on loading a private model using authentication, check
[model auth during search](https://docs.marqo.ai/0.0.19/API-Reference/search/#model-auth) and 
[model auth during add_documents](https://docs.marqo.ai/0.0.19/API-Reference/documents/#model-auth).

- Bulk search score modifier and context vector support (https://github.com/marqo-ai/marqo/pull/469). 
Support has been added for [score modifiers](https://docs.marqo.ai/0.0.21/API-Reference/search/#score-modifiers) 
and [context vectors](https://docs.marqo.ai/0.0.21/API-Reference/search/#context) to our bulk search API. 
This can help enhance throughput and performance for certain workloads. Please see [documentation](https://docs.marqo.ai/0.0.21/API-Reference/bulk/) for usage. 

## Bug fixes and minor changes
- README enhancements (https://github.com/marqo-ai/marqo/pull/482, https://github.com/marqo-ai/marqo/pull/481).

## Contributor shout-outs
- A special thank you to our 3.0k stargazers!


# Release 0.0.20

## New features
- Custom model pre-loading (https://github.com/marqo-ai/marqo/pull/475). Public CLIP and OpenCLIP models specified by URL can now be loaded on Marqo startup via the `MARQO_MODELS_TO_PRELOAD` environment variable. These must be formatted as JSON objects with `model` and `model_properties`.
  See [here (configuring pre-loaded models)](https://marqo.pages.dev/0.0.20/Advanced-Usage/configuration/#configuring-preloaded-models) for usage.

## Bug fixes and minor changes
- Fixed arm64 build issue caused by package version conflicts (https://github.com/marqo-ai/marqo/pull/478)


# Release 0.0.19

## New features
- Model authorisation(https://github.com/marqo-ai/marqo/pull/460). Non-public OpenCLIP and CLIP models can now be loaded 
  from Hugging Face and AWS s3 via the `model_location` settings object and `model_auth`. 
  See [here (model auth during search)](https://docs.marqo.ai/0.0.19/API-Reference/search/#model-auth)
  and [here (model auth during add_documents)](https://docs.marqo.ai/0.0.19/API-Reference/documents/#model-auth) for usage.
- Max replicas configuration (https://github.com/marqo-ai/marqo/pull/465). 
  Marqo admins now have more control over the max number of replicas that can be set for indexes on the Marqo instance.
  See [here](https://docs.marqo.ai/0.0.19/Advanced-Usage/configuration/#configuring-usage-limits) for how to configure this.

## Breaking changes
- Marqo now allows for a maximum of 1 replica per index by default (https://github.com/marqo-ai/marqo/pull/465).

## Bug fixes and minor changes
- README improvements (https://github.com/marqo-ai/marqo/pull/468)
- OpenCLIP version bumped (https://github.com/marqo-ai/marqo/pull/461)
- Added extra tests (https://github.com/marqo-ai/marqo/pull/464/)
- Unneeded files are now excluded in Docker builds (https://github.com/marqo-ai/marqo/pull/448, https://github.com/marqo-ai/marqo/pull/426)

## Contributor shout-outs
- Thank you to our 2.9k stargazers!
- Thank you to community members for the increasingly exciting discussions on our Slack channel. 
  Feedback, questions and hearing about use cases helps us build a great open source product.
- Thank you to [@jalajk24](https://github.com/jalajk24) for the PR to exclude unneeded files from Docker builds!


# Release 0.0.18

## New features
- New E5 model type is available (https://github.com/marqo-ai/marqo/pull/419). E5 models are state of the art general-purpose text embedding models that obtained the best results on the MTEB benchmark when released in Dec 2022. Read more about these models [here](https://docs.marqo.ai/0.0.18/Models-Reference/dense_retrieval/#text).
- Automatic model ejection (https://github.com/marqo-ai/marqo/pull/372). Automatic model ejection helps prevent out-of-memory (OOM) errors on machines with a larger amount of CPU memory (16GB+) by ejecting the least recently used model. 
- Speech processing article and example (https://github.com/marqo-ai/marqo/pull/431). [@OwenPendrighElliott](https://github.com/OwenPendrighElliott) demonstrates how you can build and query a Marqo index from audio clips. 

## Optimisations 
- Delete optimisation (https://github.com/marqo-ai/marqo/pull/436). The `/delete` endpoint can now handle a higher volume of requests.
- Inference calls can now execute in batches, with batch size configurable by an environment variable (https://github.com/marqo-ai/marqo/pull/376).

## Bug fixes and minor changes
- Configurable max value validation for HNSW graph parameters (https://github.com/marqo-ai/marqo/pull/424). See [here](https://docs.marqo.ai/0.0.18/Advanced-Usage/configuration/#other-configurations) for how to configure.
- Configurable maximum number of tensor search attributes (https://github.com/marqo-ai/marqo/pull/430). See [here](https://docs.marqo.ai/0.0.18/Advanced-Usage/configuration/#other-configurations) for how to configure.
- Unification of vectorise output type (https://github.com/marqo-ai/marqo/pull/432)
- Improved test pipeline reliability (https://github.com/marqo-ai/marqo/pull/438, https://github.com/marqo-ai/marqo/pull/439)
- Additional image download tests (https://github.com/marqo-ai/marqo/pull/402, https://github.com/marqo-ai/marqo/pull/442)
- Minor fix in the Iron Manual example (https://github.com/marqo-ai/marqo/pull/440)
- Refactored HTTP requests wrapper (https://github.com/marqo-ai/marqo/pull/367)

## Contributor shout-outs
- Thank you to our 2.8k stargazers!
- Thank you community members raising issues and discussions in our Slack channel. 
- Thank you [@jess-lord](https://github.com/jess-lord) and others for raising issues

# Release 0.0.17 

## New features
- New parameters that allow tweaking of Marqo indexes' underlying HNSW graph. `ef_construction` and `m`  can be defined at index time (https://github.com/marqo-ai/marqo/pull/386, https://github.com/marqo-ai/marqo/pull/420, https://github.com/marqo-ai/marqo/pull/421), giving you more control over the relevancy/speed tradeoff. See usage and more details [here](https://docs.marqo.ai/0.0.17/API-Reference/indexes/#example_1).
- Score modification fields (https://github.com/marqo-ai/marqo/pull/414). Rank documents using knn similarity in addition to document metadata ( https://github.com/marqo-ai/marqo/pull/414). This allows integer or float fields from a document to bias a document's score during the knn search and allows additional ranking signals to be used. Use cases include giving more reputable documents higher weighting and de-duplicating search results. See usage [here](https://docs.marqo.ai/0.0.17/API-Reference/search/#score-modifiers).

## Bug fixes and minor changes
- Added validation for unknown parameters during bulk search (https://github.com/marqo-ai/marqo/pull/413).
- Improved concurrency handling when adding documents to an index as it's being deleted (https://github.com/marqo-ai/marqo/pull/407).
- Better error messages for multimodal combination fields (https://github.com/marqo-ai/marqo/pull/395).
- Examples of recently added features added to README (https://github.com/marqo-ai/marqo/pull/403).

## Contributor shout-outs
- Thank you to our 2.6k stargazers.
- Thank you to [@anlrde](https://github.com/anlrde), [@strich](https://github.com/strich), [@feature-hope](https://github.com/feature-hope), [@bazuker](https://github.com/bazuker) for raising issues!


## Release 0.0.16

## New features
- Bulk search (https://github.com/marqo-ai/marqo/pull/363, https://github.com/marqo-ai/marqo/pull/373). 
Conduct multiple searches with just one request. This improves search throughput in Marqo by parallelising multiple search queries in a single API call. 
The average search time can be decreased up to 30%, depending on your devices and models. 
Check out the usage guide [here](https://docs.marqo.ai/0.0.16/API-Reference/bulk)
- Configurable number of index replicas (https://github.com/marqo-ai/marqo/pull/391). 
You can now configure how many replicas to make for an index in Marqo using the `number_of_replicas` parameter. Marqo makes 1 replica by default.
We recommend having at least one replica to prevent data loss.
See the usage guide [here](https://docs.marqo.ai/0.0.16/API-Reference/indexes/#body-parameters)
- Use your own vectors during searches (https://github.com/marqo-ai/marqo/pull/381). Use your own vectors as context for your queries. 
Your vectors will be incorporated into the query using a weighted sum approach, 
allowing you to reduce the number of inference requests for duplicated content.
Check out the usage guide [here](https://docs.marqo.ai/0.0.16/API-Reference/search/#context)

## Bug fixes and minor changes
- Fixed a bug where some Open CLIP models were unable to load checkpoints from the cache (https://github.com/marqo-ai/marqo/pull/387).
- Fixed a bug where multimodal search vectors are not combined based on expected weights (https://github.com/marqo-ai/marqo/pull/384).
- Fixed a bug where multimodal document vectors are not combined in an expected way. `numpy.sum` was used rather than `numpy.mean`.  (https://github.com/marqo-ai/marqo/pull/384).
- Fixed a bug where an unexpected error is thrown when `using_existing_tensor = True` and documents are added with duplicate IDs (https://github.com/marqo-ai/marqo/pull/390).
- Fixed a bug where the index settings validation did not catch the `model` field if it is in the incorrect part of the settings json (https://github.com/marqo-ai/marqo/pull/365).
- Added missing descriptions and requirement files on our [GPT-examples](https://github.com/marqo-ai/marqo/tree/mainline/examples/GPT-examples) (https://github.com/marqo-ai/marqo/pull/349).  
- Updated the instructions to start Marqo-os (https://github.com/marqo-ai/marqo/pull/371).
- Improved the Marqo start-up time by incorporating the downloading of the punkt tokenizer into the dockerfile (https://github.com/marqo-ai/marqo/pull/346).

## Contributor shout-outs
- Thank you to our 2.5k stargazers.
- Thank you to [@ed-muthiah](https://github.com/ed-muthiah) for submitting a PR (https://github.com/marqo-ai/marqo/pull/349) 
that added missing descriptions and requirement files on our [GPT-examples](https://github.com/marqo-ai/marqo/tree/mainline/examples/GPT-examples).

# Release 0.0.15

## New features 
- Multimodal tensor combination (https://github.com/marqo-ai/marqo/pull/332, https://github.com/marqo-ai/marqo/pull/355). Combine image and text data into a single vector! Multimodal combination objects can be added as Marqo document fields. For example, this can be used to encode text metadata into image vectors. See usage [here](https://docs.marqo.ai/0.0.15/Advanced-Usage/document_fields/#multimodal-combination-object).

## Bug fixes
- Fixed a bug that prevented CLIP's device check from behaving as expected (https://github.com/marqo-ai/marqo/pull/337)
- CLIP utils is set to use the OpenCLIP default tokenizer so that long text inputs are truncated correctly (https://github.com/marqo-ai/marqo/pull/351). 

## Contributor shout-outs:
- Thank you to our 2.4k stargazers
- Thank you to [@ed-muthiah](https://github.com/ed-muthiah), [@codebrain](https://github.com/codebrain) and others for raising issues.


# Release 0.0.14

## New features 
- `use_existing_tensors` flag, for `add_documents` (https://github.com/marqo-ai/marqo/pull/335). Use existing Marqo tensors to autofill unchanged tensor fields, for existing documents. This lets you quickly add new metadata while minimising inference operations. See usage [here](https://docs.marqo.ai/0.0.14/API-Reference/documents/#query-parameters).
- `image_download_headers` parameter for `search` and `add_documents` (https://github.com/marqo-ai/marqo/pull/336). Index and search non-publicly available images. Add image download auth information to `add_documents` and `search` requests. See usage [here](https://docs.marqo.ai/0.0.14/API-Reference/image_downloads/).

## Optimisations
- The index cache is now updated on intervals of 2 seconds (https://github.com/marqo-ai/marqo/pull/333), rather than on every search. This reduces the pressure on Marqo-OS, allowing for greater search and indexing throughput. 

## Bug fixes
- Helpful validation errors for invalid index settings (https://github.com/marqo-ai/marqo/pull/330). Helpful error messages allow for a smoother getting-started experience. 
- Automatic precision conversion to `fp32` when using `fp16` models on CPU (https://github.com/marqo-ai/marqo/pull/331). 
- Broadening of the types of image download errors gracefully handled. (https://github.com/marqo-ai/marqo/pull/321)


# Release 0.0.13

## New features
- Support for custom CLIP models using the OpenAI and OpenCLIP architectures (https://github.com/marqo-ai/marqo/pull/286). Read about usage [here](https://docs.marqo.ai/0.0.13/Models-Reference/dense_retrieval/#generic-clip-models).
- Concurrency throttling (https://github.com/marqo-ai/marqo/pull/304). Configure the number of allowed concurrent indexing and search threads. Read about usage [here](https://docs.marqo.ai/0.0.13/Advanced-Usage/configuration/#configuring-throttling).
- Configurable logging levels (https://github.com/marqo-ai/marqo/pull/314). Adjust log output for your debugging/log storage needs. See how to configure log level [here](https://docs.marqo.ai/0.0.13/Advanced-Usage/configuration/#configuring-log-level).
- New array datatype (https://github.com/marqo-ai/marqo/pull/312). You can use these arrays as a collection of tags to filter on! See usage [here](https://docs.marqo.ai/0.0.13/Advanced-Usage/document_fields/#array).
- Boost tensor fields during search (https://github.com/marqo-ai/marqo/pull/300). Weight fields as higher and lower relative to each other during search. Use this to get a mix of results that suits your use case. See usage [here](https://docs.marqo.ai/0.0.13/API-Reference/search/#boost).
- Weighted multimodal queries (https://github.com/marqo-ai/marqo/pull/307). You can now search with a dictionary of weighted queries. If searching an image index, these queries can be a weighted mix of image URLs and text. See usage [here](https://docs.marqo.ai/0.0.13/API-Reference/search/#query-q).
- New GPT-Marqo integration [example](https://github.com/marqo-ai/marqo/tree/mainline/examples/GPT-examples) and [article](https://www.marqo.ai/blog/from-iron-manual-to-ironman-augmenting-gpt-with-marqo-for-fast-editable-memory-to-enable-context-aware-question-answering). Turn your boring user manual into a question-answering bot, with an optional persona, with GPT + Marqo!
- Added new OpenCLIP models to Marqo (https://github.com/marqo-ai/marqo/pull/299)

## Optimisations
- Concurrent image downloads (https://github.com/marqo-ai/marqo/pull/281, https://github.com/marqo-ai/marqo/pull/311)
- Blazingly fast `fp16` ViT CLIP models (https://github.com/marqo-ai/marqo/pull/286). See usage [here](https://docs.marqo.ai/0.0.13/Models-Reference/dense_retrieval/#openai-float16)
- Reduction of data transfer between Marqo and Marqo-os (https://github.com/marqo-ai/marqo/pull/300)
- We see a 3.0x indexing speedup, and a 1.7x search speedup, using the new `fp16/ViT-L/14` CLIP model, compared to the previous release using `ViT-L/14`.  

## Bug fixes 
- Fixed 500 error when creating an index while only specifying `number_of_shards`(https://github.com/marqo-ai/marqo/pull/293)
- Fixed model cache management no parsing reranker model properties properly (https://github.com/marqo-ai/marqo/pull/308)  

## Contributor shout-outs
- Thank you to our 2.3k stargazers
- Thank you to [@codebrain](https://github.com/codebrain) and others for raising issues.


# Release 0.0.12

## New features
- Multilingual CLIP (https://github.com/marqo-ai/marqo/pull/267). Search images in the language you want! Marqo now incorporates [open source multilingual CLIP models](https://github.com/FreddeFrallan/Multilingual-CLIP). A list of available multilingual CLIP models are available [here](https://docs.marqo.ai/0.0.12/Models-Reference/dense_retrieval/#multilingual-clip). 
- Exact text matching (https://github.com/marqo-ai/marqo/pull/243, https://github.com/marqo-ai/marqo/pull/288). Search for specific words and phrases using double quotes (`" "`) in lexical search. See usage [here](https://docs.marqo.ai/0.0.12/API-Reference/search/#lexical-search-exact-matches).  

## Optimisations 
- Search speed-up (https://github.com/marqo-ai/marqo/pull/278). Latency reduction from Marqo-os indexes reconfigurations. 

## Contributor shout-outs
Thank you to our 2.2k stargazers and 80+ forkers!

# Release 0.0.11

## New features 
- Pagination (https://github.com/marqo-ai/marqo/pull/251). Navigate through pages of results. Provide an extensive end-user search experience without having to keep results in memory! See usage [here](https://docs.marqo.ai/0.0.11/API-Reference/search/#search-result-pagination) 
- The `/models` endpoint (https://github.com/marqo-ai/marqo/pull/239). View what models are loaded, and on what device. This lets Marqo admins examine loaded models and prune unneeded ones. See usage [here](https://docs.marqo.ai/0.0.11/API-Reference/models/)
- The `/device` endpoint (https://github.com/marqo-ai/marqo/pull/239). See resource usage for the machine Marqo is running on. This helps Marqo admins manage resources on remote Marqo instances. See usage [here](https://docs.marqo.ai/0.0.11/API-Reference/device/)
- The index settings endpoint (`/indexes/{index_name}/settings`)(https://github.com/marqo-ai/marqo/pull/248). See the model and parameters used by each index. See usage [here](https://docs.marqo.ai/0.0.11/API-Reference/settings/). 
- Latency log outputs (https://github.com/marqo-ai/marqo/pull/242). Marqo admins have better transparency about the latencies for each step of the Marqo indexing and search request pipeline
- ONNX CLIP models are now available (https://github.com/marqo-ai/marqo/pull/245). Index and search images in Marqo with CLIP models in the faster, and open, ONNX format - created by Marqo's machine learning team. These ONNX CLIP models give Marqo up to a 35% speedup over standard CLIP models. These ONNX CLIP models are open sourced by Marqo. Read about usage [here](https://docs.marqo.ai/0.0.11/Models-Reference/dense_retrieval/#onnx-clip).
- New simple [image search](https://github.com/marqo-ai/marqo/blob/mainline/examples/ImageSearchGuide/ImageSearchGuide.md) guide (https://github.com/marqo-ai/marqo/pull/253, https://github.com/marqo-ai/marqo/pull/263). 


## Contributor shout-outs
- ⭐️ We've just hit over 2.1k GitHub stars! ⭐️ So an extra special thanks to our stargazers and contributors who make Marqo possible. 

# Release 0.0.10

## New features 
- Generic model support (https://github.com/marqo-ai/marqo/pull/179). Create an index with your favourite SBERT-type models from HuggingFace! Read about usage [here](https://marqo.pages.dev/0.0.10/Models-Reference/dense_retrieval/#generic-models)
- Visual search update 2. (https://github.com/marqo-ai/marqo/pull/214). Search-time image reranking and open-vocabulary localization, based on users' queries, is now available with the Owl-ViT model. **Locate the part of the image corresponding to your query!** Read about usage [here](https://docs.marqo.ai/0.0.10/Models-Reference/reranking/) 
- Visual search update 1. (https://github.com/marqo-ai/marqo/pull/214). Better image patching. In addition to faster-rcnn, you can now use yolox or attention based (DINO) region proposal as a patching method at indexing time. This allows localization as the sub patches of the image can be searched. Read about usage [here](https://docs.marqo.ai/0.0.10/Preprocessing/Images/). 

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
