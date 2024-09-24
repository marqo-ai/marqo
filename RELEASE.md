# Release 2.12.1

## Bug fixes and minor changes
- Fix a bug where when `treatUrlsAndPointersAsImages` is unset and `treatUrlsAndPointersAsMedia` is set, Marqo returns an error where `treatUrlsAndPointersAsImages` cannot be `False` when `treatUrlsAndPointersAsMedia` is `True` 
- Add new video-audio model `LanguageBind/Video_V1.5_FT_Audio_FT` to the model registry.


# Release 2.12.0

## New features
- Add support for video and audio modalities using LanguageBind models ([#931](https://github.com/marqo-ai/marqo/pull/931)). You can now index, embed, and search with video and audio files using Marqo, extending your search capabilities beyond text and images. 
- Load OpenCLIP models from HuggingFace Hub ([#939](https://github.com/marqo-ai/marqo/pull/939)). Support loading OpenCLIP models directly from HuggingFace by providing a model name with the hf-hub: prefix. This simplifies model integration and expands your options.
- Load custom OpenCLIP checkpoints with different image preprocessors ([#939](https://github.com/marqo-ai/marqo/pull/939)). Allow loading a custom OpenCLIP checkpoint with a different image preprocessor by providing imagePreprocessor in the model properties. This offers greater flexibility in model selection and customization.


## Bug fixes and minor changes
- Fix tokenizer loading for custom OpenCLIP checkpoints ([#939](https://github.com/marqo-ai/marqo/pull/939)). The correct tokenizer is now applied when custom OpenCLIP model checkpoints are loaded.
- Improve error handling for image_pointer fields in structured indexes ([#944](https://github.com/marqo-ai/marqo/pull/944)). Structured indexes now have targeted error reporting for non-image content in image_pointer fields. This improvement prevents batch failures and provides clearer feedback to users.

## Contributor shout-outs
- Shoutouts to our valuable 4.5k stargazers!
- Thanks a lot for the discussion and suggestions in our community. We love to hear your thoughts and requests. Join our [Slack channel](https://join.slack.com/t/marqo-community/shared_invite/zt-2jm456s90-1pFxdE5kDQt5imqddXUIcw) and [forum](https://community.marqo.ai/) now.

# Release 2.11.4

## Bug fixes and minor changes
- Fix duplication of results in RRF hybrid search ([#957](https://github.com/marqo-ai/marqo/pull/957)). Resolved an issue where some results in Reciprocal Rank Fusion (RRF) hybrid search were duplicated, ensuring more accurate and unique search results.

# Release 2.11.3

## Bug fixes and minor changes
- Support S3 custom model without explicit credentials ([#948](https://github.com/marqo-ai/marqo/pull/948)).

# Release 2.11.2

## Bug fixes and minor changes
- Fix an issue where CUDA was not automatically selected as the default device for the `embed` endpoint, even when available [#941](https://github.com/marqo-ai/marqo/pull/941).

# Release 2.11.1

## Bug fixes and minor changes
- Added a default User-Agent header (`Marqobot/1.0`) and enabled automatic redirection handling when downloading images ([#932](https://github.com/marqo-ai/marqo/pull/932)). This enhancement allows Marqo to correctly process image URLs that require a `User-Agent` header or redirection.

# Release 2.11.0

## New features

- Hybrid Search for unstructured indexes (`"searchMethod": "HYBRID”`) ([#912](https://github.com/marqo-ai/marqo/pull/912)). Marqo now supports hybrid search for unstructured indexes, combining lexical and tensor search (e.g., using reciprocal rank fusion - RRF) to provide the best relevance possible. See usage [here](https://docs.marqo.ai/2.11/API-Reference/Search/search/#example-2-creating-and-searching-an-unstructured-index-hybrid-search-with-model-deployed-within-marqo). Please note that hybrid search only works on a fresh Marqo 2.11.0 instance without state transfer for now. This is a limitation that we will address in the next release.
- [Marqo Terraform provider](https://github.com/marqo-ai/terraform-provider-marqo) is now available on both [OpenTofu Registry](https://github.com/opentofu/registry/blob/main/providers/m/marqo-ai/marqo.json) and [Terraform Registry](https://registry.terraform.io/providers/marqo-ai/marqo/latest). See usage [here](https://docs.marqo.ai/2.11/Cloud-Reference/opentofu_provider/)

## Bug fixes and minor changes

- Improve the error handling of batch add/update/get documents API ([#911](https://github.com/marqo-ai/marqo/pull/911)). Now each document in a batch request has its individual response status with detailed error message. See details [here](https://docs.marqo.ai/2.11/API-Reference/Documents/add_or_replace_documents/#response)
- Fix incorrect or missing prefixes for some models in the registry ([#917](https://github.com/marqo-ai/marqo/pull/917)). This change improves all BGE models, all Snowflake models, and multilingual-e5-large-instruct. For example, `snowflake-arctic-embed-l` model has 34% improvement in NDCG@10 on the Arguana benchmark and 153% improvement in NDCG@10 on the FIQA benchmark.
- Increase the maxHits and maxOffset limit to 1,000,000 in the default query profile. ([#914](https://github.com/marqo-ai/marqo/pull/914)). This allows user to override `MARQO_MAX_SEARCH_LIMIT` and `MARQO_MAX_SEARCH_OFFSET` environment variables to large values up to one million. Please note that this is an advanced setting and very large values aren’t normally recommended.
- Fix a bug that causes 400 error when using hybrid search with LEXICAL retrieval method and TENSOR ranking method and `scoreModifiersLexical` ([#922](https://github.com/marqo-ai/marqo/pull/922)).

## Contributor shout-outs

- Huge shoutout to all our 4.4k stargazers! We’ve come a long way as a team and as a community, so a huge thanks to everyone who continues to support Marqo.
- Feel free to keep on sharing questions and feedback on our [forum](https://community.marqo.ai/) and [Slack channel](https://marqo-community.slack.com/join/shared_invite/zt-2b4nsvbd2-TDf8agPszzWH5hYKBMIgDA#/shared-invite/email)! If you have any more inquiries or thoughts, please don’t hesitate to reach out.

# Release 2.10.2

## Bug fixes and minor changes
- Fix an issue where CUDA was not automatically selected as the default device for the `embed` endpoint, even when available [#941](https://github.com/marqo-ai/marqo/pull/941).

# Release 2.10.1

## Bug fixes and minor changes
- Improve the clarity of the error message when Marqo can not download the provided image ([#905](https://github.com/marqo-ai/marqo/pull/905)).
- Improve the error message in hybrid search to avoid confusion ([#900](https://github.com/marqo-ai/marqo/pull/900)).
- Fix a bug where a `500` error is returned when an unsupported search method is provided. Marqo now correctly returns a `400` error ([#899](https://github.com/marqo-ai/marqo/pull/899)).
- Fix a bug where a `500` error is returned when an invalid image URL with non-ASCII characters is provided. Marqo now encodes the image URL correctly ([#908](https://github.com/marqo-ai/marqo/pull/908)).

# Release 2.10.0

## New features

- Hybrid Search (`"searchMethod": "HYBRID”`) (https://github.com/marqo-ai/marqo/pull/845). Marqo now supports hybrid search, combining lexical and tensor search (using reciprocal rank fusion) to provide the best relevance possible. See usage [here](https://docs.marqo.ai/2.10/API-Reference/Search/search/#hybrid-parameters).
- Lexical Search score modifiers (https://github.com/marqo-ai/marqo/pull/884). Score modifiers are now supported for lexical search. Score modifiers are applied on all matches, not just the top k retrieved, resulting in more relevant hits. See usage [here](https://docs.marqo.ai/2.10/API-Reference/Search/search/#score-modifiers).

## Bug fixes and minor changes

- Increase unstructured index default `filterStringMaxLength` to 50, from 20 (https://github.com/marqo-ai/marqo/pull/887). Maximum length of string fields to be used in query filters now defaults to 50 characters long.

## Contributor shout-outs

- Huge shoutout to all our 4.3k stargazers! We’ve come a long way as a team and as a community, so a huge thanks to everyone who continues to support Marqo.
- Feel free to keep on sharing questions and feedback on our [forum](https://community.marqo.ai/) and [Slack channel](https://marqo-community.slack.com/join/shared_invite/zt-2b4nsvbd2-TDf8agPszzWH5hYKBMIgDA#/shared-invite/email)! If you have any more inquiries or thoughts, please don’t hesitate to reach out.

# Release 2.9.0

## New features
- Numeric map data type. Add numeric map data types, available for filtering and score modification ([#851](https://github.com/marqo-ai/marqo/pull/851)). You can now store a map/dictionary of numeric value and use these in your filters and score modifiers, or simply retrieve these with your documents. See the new types [here](https://docs.marqo.ai/2.9/API-Reference/Indexes/create_structured_index/?h=struct#fields). For usage in search, see [here](https://docs.marqo.ai/2.9/API-Reference/Search/search/?h=map#example_4).  This is supported only for indexes created with Marqo 2.9 or later.
- Double and long score modifier fields. Support double and long in map and standard numeric fields for score modifiers in both structured and unstructured indexes ([#851](https://github.com/marqo-ai/marqo/pull/851)). You can now use double values with full precision as score modifiers, as well as integers with guaranteed precision up to `2^53 - 1` (increased from `2^24 - 1`), with only negligible precision loss for larger values. For details on these new types, see the documentation [here](https://docs.marqo.ai/2.9/API-Reference/Indexes/create_structured_index/?h=struct#fields).

## Bug fixes and minor changes
- Fix the bug in score modifiers where missing score modifiers in docs used in `multiply_score_by` lead to the multiplication of scores by `0` instead of by `1` ([#851](https://github.com/marqo-ai/marqo/pull/851)).
- Improve upgrade stability ([#874](https://github.com/marqo-ai/marqo/pull/874)). Fix failure of state transfer between some versions of Marqo due to Vespa binaries being copied with state. For more information, see the documentation [here](https://docs.marqo.ai/2.9/Guides/Advanced-Usage/transferring_state/?h=transfer)
- Improve the model warmup strategy on instances with CUDA ([#877](https://github.com/marqo-ai/marqo/pull/877)). Marqo now requires less memory to warmup the models when spinning up .
- Improve create/delete index resilience to partial failures ([#866](https://github.com/marqo-ai/marqo/pull/866)). You can now bring Marqo to a consistent state by repeating the operation until getting a `200` response.

## Contributor shout-outs
- Shoutouts to our valuable 4.3k stargazers!
- Thanks a lot for the discussion and suggestions in our community. We love to hear your thoughts and requests. Join our [Slack channel](https://join.slack.com/t/marqo-community/shared_invite/zt-2jm456s90-1pFxdE5kDQt5imqddXUIcw) and [forum](https://community.marqo.ai/) now.

# Release 2.8.2

## Bug fixes and minor changes
- Fix an issue in Marqo where loading some models (e.g., `open_clip/xlm-roberta-base-ViT-B-32/laion5b_s13b_b90k`) is unsuccessful. 
This was resolved by upgrading the `transformers` and `optimum` packages. ([#868](https://github.com/marqo-ai/marqo/pull/868))

# Release 2.8.1

## Bug fixes and minor changes
- Fix a bug in Marqo where a 500 error is returned for the entire batch of documents when encountering an invalid document ID during image downloading. 
Marqo now correctly returns an error and rejects the invalid document, 
allowing successful indexing of other valid documents with a 200 response. ([#860](https://github.com/marqo-ai/marqo/pull/860))

# Release 2.8.0

## New features
- Improve `add_documents` memory efficiency and throughput for CLIP and Open_CLIP models when indexing documents with images when no patch method is used ([#849](https://github.com/marqo-ai/marqo/pull/849)). The image downloading and preprocessing logic has been improved. Marqo now converts the images to tensors directly after downloading. In our tests, the memory usage has been reduced by 37.5% and the throughput has been increased by 7.5% (subject to your settings). Marqo is also more stable when indexing documents in a multi-threading scenario.
- Add support for pre-warming patch models ([#847](https://github.com/marqo-ai/marqo/pull/847)). See usage [here](https://docs.marqo.ai/2.8/Guides/Advanced-Usage/configuration/#configuring-preloaded-patch-models).

## Bug fixes and minor changes
- Replace the requests package with pycurl for faster image downloads ([#814](https://github.com/marqo-ai/marqo/pull/814)). Marqo now downloads images 2-3x faster in our tests and the overall `add_documents` throughput is increased by 7.5%

## Contributor shout-outs
- Shoutouts to our valuable 4.2k stargazers!
- Thanks a lot for the discussion and suggestions in our community. We love to hear your thoughts and requests. Join our [Slack channel](https://join.slack.com/t/marqo-community/shared_invite/zt-2jm456s90-1pFxdE5kDQt5imqddXUIcw) and [forum](https://community.marqo.ai/) now.


# Release 2.7.2

## Bug fixes and minor changes
- Fix an issue causing an error during the Marqo shutdown process (https://github.com/marqo-ai/marqo/pull/850). Marqo now shuts down properly without encountering errors.

# Release 2.7.1

## Bug fixes and minor changes
- Resolve an issue where Marqo could not create or delete an index when not connected to the Zookeeper server (https://github.com/marqo-ai/marqo/pull/848). Users can now create or delete an index without needing to connect to the Zookeeper server. However, please note that without the Zookeeper server, your request is not protected in concurrent scenarios. For guidance on configuring your Zookeeper server, refer to [this documentation](https://docs.marqo.ai/2.7/Guides/Advanced-Usage/configuration/#configure-backend-communication).

# Release 2.7.0

## New features
- Update Open CLIP version and support new families of models, e.g., `MetaCLIP`, `DatacompCLIP`  ([#833](https://github.com/marqo-ai/marqo/pull/833)). Update the Open CLIP version to `2.24.0`  which includes new and state-of-the-art multimodal models. You can choose these models to build your index. Check [here](https://github.com/marqo-ai/marqo/releases/%5B%3Chttps://docs.marqo.ai/2.7/Guides/Models-Reference/list_of_models/#open-clip%3E%5D(%3Chttps://docs.marqo.ai/2.6/Guides/Models-Reference/list_of_models/#open-clip%3E)) for the available models.
- Support lexical search with only a filter (https://github.com/marqo-ai/marqo/pull/840). Marqo now supports a match-all query (`"*"`) with a filter in lexical search. This allows you to search your documents solely based on the filter content without considering the relevance. This is a community-requested feature (https://github.com/marqo-ai/marqo/issues/770, https://github.com/marqo-ai/marqo/issues/771) and we love to hear from our users.

## Bug fixes and minor changes
- Improve the thread safety of index creation and deletion operations (https://github.com/marqo-ai/marqo/pull/838). Marqo now returns an `operation_conflict_error(409)` if users try to delete or create an index when there is another index creation or deletion in progress.
- Fix a bug that an empty string lexical search query (`""`) returns a 500 error (https://github.com/marqo-ai/marqo/pull/840). Marqo now returns an empty search result for such a query.
- Address verbose logging at the `WARNING` level when `attributes_to_retrieve` excludes fields required to build highlights. (https://github.com/marqo-ai/marqo/pull/837)

## Contributor shout-outs
- Shoutouts to our valuable 4.2k stargazers!
- Thanks [@jesse-lord](https://github.com/jess-lord) and [@afroozsheikh](https://github.com/afroozsheikh) for requesting valuable features to improve Marqo!
- Thanks a lot for the discussion and suggestions in our community. We love to hear your thoughts and requests. Join our [Slack channel](https://join.slack.com/t/marqo-community/shared_invite/zt-2jm456s90-1pFxdE5kDQt5imqddXUIcw) and [forum](https://community.marqo.ai/) now.


# Release 2.6.0

## New features 
- Support for custom and default prefixes ([#821](https://github.com/marqo-ai/marqo/pull/821) and [#832](https://github.com/marqo-ai/marqo/pull/832)) in the index creation, adding documents, search, and embed endpoints. See usage for index creation [here](https://docs.marqo.ai/2.6/API-Reference/Indexes/create_index), adding documents [here](https://docs.marqo.ai/2.6/API-Reference/Documents/add_or_replace_documents), search [here](https://docs.marqo.ai/2.6/API-Reference/Search/search), and embed [here](https://docs.marqo.ai/2.6/API-Reference/Embed/embed).

## Bug fixes and minor changes
- Improved recommender with structured indexes ([#830](https://github.com/marqo-ai/marqo/pull/830))
- Better handling of image download errors ([#829](https://github.com/marqo-ai/marqo/pull/829)). Image download errors will now return a 200 overall and log errors per document.

## Contributor Shout-outs
- Shoutout to all our 4.2k stargazers! Thanks for continuing to use our product and helping Marqo grow.
- Keep on sharing your questions and feedback on our [forum](https://community.marqo.ai/) and [Slack channel](https://marqo-community.slack.com/join/shared_invite/zt-2b4nsvbd2-TDf8agPszzWH5hYKBMIgDA#/shared-invite/email)! If you have any more inquiries or thoughts, please don’t hesitate to reach out.
 
# Release 2.5.1

## Bug fixes and minor changes
- More stable `recommend` endpoint (https://github.com/marqo-ai/marqo/pull/825).
- Change error code when using `IN` filter operator on an unstructured index (https://github.com/marqo-ai/marqo/pull/823).
- New index settings validation endpoint for Cloud use (https://github.com/marqo-ai/marqo/pull/809).

# Release 2.5.0
## New features
- New ‘embed’ endpoint (`POST /indexes/{index_name}/embed`) (https://github.com/marqo-ai/marqo/pull/803). Marqo can now perform inference and return the embeddings for a single piece or list of content, where content can be either a string or weighted dictionary of strings. See usage [here](https://docs.marqo.ai/2.5/API-Reference/Embed/embed/). 
- New ‘recommend’ endpoint (`POST /indexes/{index_name}/recommend`) (https://github.com/marqo-ai/marqo/pull/816). Given a list of existing document IDs, Marqo can now recommend similar documents by performing a search on interpolated vectors from the documents. See usage [here](https://docs.marqo.ai/2.5/API-Reference/Search/recommend/). 
- Add Inference Cache to speed up frequent search and embed requests (https://github.com/marqo-ai/marqo/pull/802). Marqo now caches embeddings generated during inference. The cache size and type can be configured with `MARQO_INFERENCE_CACHE_SIZE` and `MARQO_INFERENCE_CACHE_TYPE`. See configuration instructions [here](https://docs.marqo.ai/2.5/Guides/Advanced-Usage/configuration/#configuring-cache). 
- Add configurable search timeout (https://github.com/marqo-ai/marqo/pull/813). Backend timeout now defaults to 1s, but can be configured with the environment variable `VESPA_SEARCH_TIMEOUT_MS`. See configuration instructions [here](https://docs.marqo.ai/2.5/Guides/Advanced-Usage/configuration/#configuring-usage-limits). 
- More informative `get_cuda_info` response (https://github.com/marqo-ai/marqo/pull/811). New keys: `utilization` `memory_used_percent` have been added for easier tracking of cuda device status. See [here](https://docs.marqo.ai/2.5/API-Reference/Device/get_cuda_information/) for more information.

## Bug fixes and minor changes
- Upgraded `open_clip_torch`, `timm`, and `safetensors` for access to new models (https://github.com/marqo-ai/marqo/pull/810) 

## Contributor shout-outs
- Shoutout to all our 4.1k stargazers! Thanks for continuing to use our product and helping Marqo grow.
- Keep on sharing your questions and feedback on our [forum](https://community.marqo.ai/) and [Slack channel](https://marqo-community.slack.com/join/shared_invite/zt-2b4nsvbd2-TDf8agPszzWH5hYKBMIgDA#/shared-invite/email)! If you have any more inquiries or thoughts, please don’t hesitate to reach out.

# Release 2.4.3

## Bug fixes and minor changes
- Fix incorrect Marqo version number (https://github.com/marqo-ai/marqo/pull/805). Version number updated from 2.4.1 to 2.4.3

# Release 2.4.2

## Bug fixes and minor changes
- Better response for truncated images in `add_documents` (https://github.com/marqo-ai/marqo/pull/797). Truncated images no longer cause a 500 error. The individual document will fail and return a 400 error in add docs response (full response will be a 200).

# Release 2.4.1

## Bug fixes and minor changes
- Improve telemetry memory management (https://github.com/marqo-ai/marqo/pull/800).

# Release 2.4.0

## New features
- Add `IN` operator to the query filter string DSL (https://github.com/marqo-ai/marqo/pull/790, https://github.com/marqo-ai/marqo/pull/793, & https://github.com/marqo-ai/marqo/pull/795). 
For structured indexes, you can now use the `IN` keyword to restrict text and integer fields to be within a list of values. See usage [here](https://docs.marqo.ai/2.4.0/Guides/query_dsl/#in-queries). 
 
- Add `no_model` option for index creation (https://github.com/marqo-ai/marqo/pull/789). This allows for indexes that do no vectorisation, 
providing easy use of custom vectors with no risk of accidentally mixing them up with Marqo-generated vectors. See usage [here](https://docs.marqo.ai/2.4.0/Guides/Models-Reference/list_of_models/#no-model). 
- Optional `q` parameter for the search endpoint if context vectors are provided. (https://github.com/marqo-ai/marqo/pull/789). 
This is particularly useful when using context vectors to search across your documents that have custom vector fields. See usage [here](https://docs.marqo.ai/2.4.0/API-Reference/Search/search/#query-q).

## Bug fixes and minor changes
- Improve error message for defining `tensorFields` when adding documents to a structured index (https://github.com/marqo-ai/marqo/pull/788). 

## Contributor shout-outs
- A huge thank you to all our 4.1k stargazers! We appreciate all of you continuing to use our product and helping Marqo grow.
- Thanks for sharing your questions and feedback on our [forum](https://community.marqo.ai/) and 
[Slack channel](https://marqo-community.slack.com/join/shared_invite/zt-2b4nsvbd2-TDf8agPszzWH5hYKBMIgDA#/shared-invite/email)! 
If you have any more inquiries or thoughts, please don’t hesitate to reach out.

# Release 2.3.0

## New features
- New `update_documents` API (https://github.com/marqo-ai/marqo/pull/773). Structured indexes now support high throughput partial updates to non-tensor fields. Unstructured indexes do not support partial updates. See usages [here](https://docs.marqo.ai/2.3.0/API-Reference/Documents/update_documents/)
- The custom vectors feature is now supported again for both structured and unstructured indexes (https://github.com/marqo-ai/marqo/pull/777). You can now add externally generated vectors to Marqo documents. See usages [here](https://docs.marqo.ai/2.3.0/API-Reference/Documents/add_or_replace_documents/#mappings)

## Bug fixes and minor changes
- Fix an issue where non-default distance metrics are not configured correctly with unstructured indexes (https://github.com/marqo-ai/marqo/pull/772).
- Introduce a guide for running Marqo open source in production environments, offering insights and best practices (https://github.com/marqo-ai/marqo/pull/775).
- Remove outdated examples from the README to improve clarity and relevance (https://github.com/marqo-ai/marqo/pull/766).

## Contributor shout-outs
- A huge thank you to all our 4k stargazers! This is a new milestone for Marqo!
- Stay connected and share your thoughts on our [forum](https://community.marqo.ai/) and [Slack channel](https://marqo-community.slack.com/join/shared_invite/zt-2b4nsvbd2-TDf8agPszzWH5hYKBMIgDA#/shared-invite/email)! Your insights, questions, and feedback are always welcome and highly appreciated.

# Release 2.2.3

## New features
- Add configurable search timeout (https://github.com/marqo-ai/marqo/pull/843). Backend timeout now defaults to 1s, but can be configured with the environment variable `VESPA_SEARCH_TIMEOUT_MS`. See configuration instructions [here](https://docs.marqo.ai/2.5/Guides/Advanced-Usage/configuration/#configuring-usage-limits). 

# Release 2.2.2

## Bug fixes and minor changes
- Improve telemetry memory management (https://github.com/marqo-ai/marqo/pull/804).

# Release 2.2.1

## Bug fixes and minor changes
- Fix response code for vector store timeout, change it from 429 to 504 (https://github.com/marqo-ai/marqo/pull/763)

# Release 2.2.0

## New features
- Support filtering on document ID with structured indexes. This was already supported with unstructured indexes ([#749](https://github.com/marqo-ai/marqo/pull/749))
- New structured index data types: `long`, `double`, `array<long>` and `array<double>` for a higher precision and range of values Available for indexes created with Marqo 2.2+ ([#722](https://github.com/marqo-ai/marqo/pull/722))
- Higher precision numeric fields for unstructured indexes. Unstructured indexes created with Marqo 2.2+ will use double precision floats and longs for a higher precision and wider range of values ([#722](https://github.com/marqo-ai/marqo/pull/722))
- Numeric value range validation. Values that are out of range for the field type will now receive a 400 validation error when adding documents. ([#722](https://github.com/marqo-ai/marqo/pull/722))

## Bug fixes and minor changes
- Fix unstructured index bug where filtering for boolean-like strings (e.g., `"true"`) would not work as expected ([#709](https://github.com/marqo-ai/marqo/pull/709))
- Better handling of vector store timeouts. Marqo will now return a 429 (throttled) error message when the backend vector store is receiving more traffic than it can handle([#758](https://github.com/marqo-ai/marqo/pull/758))
- Improved error logging. Stack trace will now always be logged ([#745](https://github.com/marqo-ai/marqo/pull/745))
- Better API 500 error message. Marqo will no longer return verbose error messages in the API response ([#751](https://github.com/marqo-ai/marqo/pull/751))
- Default index model is now hf/e5-base-v2 ([#710](https://github.com/marqo-ai/marqo/pull/710))
- Improve error messages ([#746](https://github.com/marqo-ai/marqo/pull/746), [#747](https://github.com/marqo-ai/marqo/pull/747))
- Improve error handling at startup when vector store is not ready. Marqo will now start and wait for vector store to become available ([#752](https://github.com/marqo-ai/marqo/pull/752))

## Contributor shout-outs
- A huge thank you to all our 3.9k stargazers!
- Thank you [@Dmitri](https://marqo-community.slack.com/team/U06GL2R5NMT) for helping us identify the issue with running Marqo on older AMD64 processors!


# Release 2.1.0

## New features
- Search result maximum limit and offset greatly increased. Maximum `limit` parameter increased from 400 to 1,000, `offset` increased from 1,000 to 10,000. Maximum value for `MARQO_MAX_RETRIEVABLE_DOCS` configuration is now 10,000 ([#735](https://github.com/marqo-ai/marqo/pull/735)​​, [#737](https://github.com/marqo-ai/marqo/pull/737)​​). See search `limit` and `offset` usage [here](https://docs.marqo.ai/2.1.0/API-Reference/Search/search/#limit)

## Bug fixes and minor changes
- Improved the Marqo bootstrapping process to address unexpected API behaviour when no index has been created yet (https://github.com/marqo-ai/marqo/pull/730).
- Improved validation for `create_index` settings (https://github.com/marqo-ai/marqo/pull/717, https://github.com/marqo-ai/marqo/pull/734). Using `dependent_fields` as a request body parameter will now raise a 400 validation error.
- Improved data parsing for documents in unstructured indexes (https://github.com/marqo-ai/marqo/pull/732). 
- Made vector store layer config upgrades and rollbacks easier ([#735](https://github.com/marqo-ai/marqo/pull/735)​​, [#736](https://github.com/marqo-ai/marqo/pull/736)​​). 
- Readme improvements (https://github.com/marqo-ai/marqo/pull/729). 


# Release 2.0.1

## Bug fixes and minor changes
- Improved stability of `use_existing_tensors` feature in `add_documents` (https://github.com/marqo-ai/marqo/pull/725).
- Improved readability of Marqo start-up logs (https://github.com/marqo-ai/marqo/pull/719).
- Removed obsolete examples ([#721](https://github.com/marqo-ai/marqo/pull/721), [#723](https://github.com/marqo-ai/marqo/pull/723)).


# Release 2.0.0
## New features
* Significant queries-per-second (QPS) and latency improvements in addition to reduced memory and storage requirements. 
Get a higher QPS and a lower latency for the same infrastructure cost, or get the same performance for much cheaper! 
In our large-scale experiments, we have achieved 2x QPS improvement, 2x speed-up in P50 search latency and 2.3x
speed-up in P99 search latency, compared to previous Marqo versions.
* Significantly improved recall. You can now get up to 99% recall (depending on your dataset and configuration) without 
sacrificing performance.
* Support for bfloat16 numeric type. Index 2x more vectors with the same amount of memory for a minimal reduction in
recall and performance.
* Structured index. You can now create structured indexes, which provide better data validation, higher performance, 
better recall and better memory efficiency.
* New API search parameter  `efSearch`. Search API now accepts an optional `efSearch` parameter which allows you to 
fine-tune the underlying HNSW search. Increase efSearch to improve recall at a minor cost of QPS and latency. See 
[here](https://docs.marqo.ai/2.0.0/API-Reference/Search/search/#body) for usage. 
* Exact nearest neighbour search. Set `"approximate": false` in the Search API body to perform an exact nearest 
neighbour search. This is useful for calculating recall and finding the best `efSearch` for your dataset. 
See [here](https://docs.marqo.ai/2.0.0/API-Reference/Search/search/#body) for usage. 
* New approximate nearest neighbour space types. Marqo now supports `euclidean`, `angular`, `dotproduct`,
`prenormalized-angular`, and `hamming` distance metrics. L1, L2 and Linf distance metrics are no longer supported.
The distance metric determines how Marqo calculates the closeness between indexed documents and search queries.
* Easier local runs. Simply run `docker run -p 8882:8882 marqoai/marqo:2.0.0` to start Marqo locally on both 
ARM64 (M-series Macs) and AMD64 machines.

## Breaking changes
* Create index API no longer accept the `index_defaults` parameter. Attributes previously defined in this object, 
like `textPreprocessing`, are now moved out to the top level settings object. 
See [here](https://docs.marqo.ai/2.0.0/API-Reference/Indexes/create_index/) for details.
* Create index API's `filterStringMaxLength` parameter determines the maximum length of strings that are indexed for 
filtering (default value 20 characters). This limitation does not apply to structured indexes. 
See [here](https://docs.marqo.ai/2.0.0/API-Reference/Indexes/create_index/) for details.
* Most APIs now require camel case request bodies and return camel case responses. See 
[create index](https://docs.marqo.ai/2.0.0/API-Reference/Indexes/create_index/), 
[search](https://docs.marqo.ai/2.0.0/API-Reference/Search/search/) and 
[add documents](https://docs.marqo.ai/2.0.0/API-Reference/Documents/add_or_replace_documents/) for a few examples.
* New Marqo configuration parameters See [here](https://docs.marqo.ai/2.0.0/Guides/Advanced-Usage/configuration/) for 
usage.
* Search response `_highlights` attribute is now a list of dictionaries. 
See [here](https://docs.marqo.ai/2.0.0/API-Reference/Search/search/#response-200-ok) for new usage.
* Add documents multimodal fields are defined as normal fields and not dictionaries. Furthermore, the mappings object 
is optional for structured indexes. See [here](https://docs.marqo.ai/2.0.0/API-Reference/Documents/add_or_replace_documents/) for usage.
* Add documents does not accept the `refresh` parameter anymore.
* The following features are available in Marqo 1.5, but are not supported by Marqo 2.0 and will be added in future 
releases:
  * Separate models for search and add documents
  * Prefixes for text chunks and queries
  * Configurable document count limit for add documents. There is a non-configurable limit of 128 in Marqo 2.0.
  * Custom (externally generated) vectors and `no_model` option for index creation.
  * Optional Search API `q` parameter when searching with context vectors.

## Contributor shout-outs
* Thank you to the community for your invaluable feedback, which drove the prioritisation for this major release.
* A warm thank you to all our 3.9k stargazers.

# Release 1.5.1
## Bug fixes and minor changes
- Adding `no_model` to `MARQO_MODELS_TO_PRELOAD` no longer causes an error on startup. Preloading process is simply skipped for this model [#657](https://github.com/marqo-ai/marqo/pull/657).


# Release 1.5.0
## New Features
- Separate model for search and add documents (https://github.com/marqo-ai/marqo/pull/633). Using the `search_model` and `search_model_properties` key in `index_defaults` allows you to specify a model specifically to be used for searching. This is useful for using a different model for search than what is used for add_documents. Learn how to use `search_model` [here](https://docs.marqo.ai/1.5.0/API-Reference/Indexes/create_index/#search-model).
- Prefixes for text chunks and queries enabled to improve retrieval for specific models (https://github.com/marqo-ai/marqo/pull/643). These prefixes are defined at the `model_properties` level, but can be overriden at index creation, add documents, or search time. Learn how to use prefixes for `add_documents` [here](https://docs.marqo.ai/1.5.0/API-Reference/Documents/add_or_replace_documents/#text-chunk-prefix) and `search` [here](https://docs.marqo.ai/1.5.0/API-Reference/Search/search/#text-query-prefix).

## Bug fixes and minor changes
- Upgraded `open_clip_torch`, `timm`, and `safetensors` for access to new models (https://github.com/marqo-ai/marqo/pull/646). 
- Documents containing multimodal objects that encounter errors in processing are rejected with a 400 error (https://github.com/marqo-ai/marqo/pull/631). 
- Updated README: More detailed explanations, fixed formatting issues (https://github.com/marqo-ai/marqo/pull/629/files, https://github.com/marqo-ai/marqo/pull/642/files).  

## Contributor shout-outs
- A huge thank you to all our 3.7k stargazers!
- Thanks everyone for continuing to participate in our [forum](https://community.marqo.ai/)! Keep all your insights, questions, and feedback coming!


# Release 1.4.0

## Breaking Changes
- Configurable document count limit for `add_documents()` calls (https://github.com/marqo-ai/marqo/pull/592). This mitigates Marqo getting overloaded 
due to add_documents requests with a very high number of documents. If you are adding documents in batches larger than the default (64), you will now 
receive an error. You can ensure your add_documents request complies to this limit by setting the Python client’s `client_batch_size` or changing this 
limit via the  `MARQO_MAX_ADD_DOCS_COUNT` variable. Read more on configuring the doc count limit [here](https://marqo.pages.dev/1.4.0/Guides/Advanced-Usage/configuration/#configuring-usage-limits).
- Default `refresh` value for `add_documents()` and `delete_documents()` set to `false` (https://github.com/marqo-ai/marqo/pull/601). This prevents 
unnecessary refreshes, which can negatively impact search and add_documents performance, especially for applications that are 
constantly adding or deleting documents. If you search or get documents immediately after adding or deleting documents, you may still get some extra 
or missing documents. To see results of these operations more immediately, simply set the `refresh` parameter to `true`. Read more on this parameter 
[here](https://marqo.pages.dev/1.4.0/API-Reference/Documents/add_or_replace_documents/#query-parameters).

## New Features
- Custom vector field type added (https://github.com/marqo-ai/marqo/pull/610). You can now add externally generated vectors to Marqo documents! See 
usage [here](https://marqo.pages.dev/1.4.0/Guides/Advanced-Usage/document_fields/#custom-vector-object).
- `no_model` option added for index creation (https://github.com/marqo-ai/marqo/pull/617). This allows for indexes that do no vectorisation, providing 
easy use of custom vectors with no risk of accidentally mixing them up with Marqo-generated vectors. See usage [here](https://marqo.pages.dev/1.4.0/API-Reference/Indexes/create_index/#no-model).
- The search endpoint's `q` parameter is now optional if `context` vectors are provided. (https://github.com/marqo-ai/marqo/pull/617). This is 
particularly useful when using context vectors to search across your documents that have custom vector fields. See usage [here](https://marqo.pages.dev/1.4.0/API-Reference/Search/search/#context).
- Configurable retries added to backend requests (https://github.com/marqo-ai/marqo/pull/623). This makes `add_documents()` and `search()` requests 
more resilient to transient network errors. Use with caution, as retries in Marqo will change the consistency guarantees for these endpoints. For more 
control over retry error handling, you can leave retry attempts at the default value (0) and implement your own backend communication error handling. 
See retry configuration instructions and how it impacts these endpoints' behaviour [here](https://marqo.pages.dev/1.4.0/Guides/Advanced-Usage/configuration/#configuring-marqo-os-request-retries).
- More informative `delete_documents()` response (https://github.com/marqo-ai/marqo/pull/619). The response object now includes a list of document 
ids, status codes, and results (success or reason for failure). See delete documents usage [here](https://marqo.pages.dev/1.4.0/API-Reference/Documents/delete_documents/).
- Friendlier startup experience (https://github.com/marqo-ai/marqo/pull/600). Startup output has been condensed, with unhelpful log messages removed. 
More detailed logs can be accessed by setting `MARQO_LOG_LEVEL` to `debug`.

## Bug fixes and minor changes
- Updated README: added Haystack integration, tips, and fixed links (https://github.com/marqo-ai/marqo/pull/593, https://github.com/marqo-ai/marqo/pull/602, https://github.com/marqo-ai/marqo/pull/616). 
- Stabilized test suite by adding score modifiers search tests (​​https://github.com/marqo-ai/marqo/pull/596) and migrating test images to S3 (https://github.com/marqo-ai/marqo/pull/594). 
- `bulk` added as an illegal index name (https://github.com/marqo-ai/marqo/pull/598). This prevents conflicts with the `/bulk` endpoint.
- Unnecessary `reputation` field removed from backend call (https://github.com/marqo-ai/marqo/pull/609).
- Fixed typo in error message (https://github.com/marqo-ai/marqo/pull/615).

## Contributor shout-outs
- A huge thank you to all our 3.7k stargazers!
- Shoutout to @TuanaCelik for helping out with the Haystack integration!
- Thanks everyone for keeping our [forum](https://community.marqo.ai/) busy. Don't hesitate to keep posting your insights, questions, and feedback!


# Release 1.3.0

## New features

- New E5 models added to model registry (https://github.com/marqo-ai/marqo/pull/568). E5 V2 and Multilingual E5 models are now available for use. The new E5 V2 models outperform their E5 counterparts in the BEIR benchmark, as seen [here](https://github.com/microsoft/unilm/tree/master/e5#english-pre-trained-models). See all available models [here](https://marqo.pages.dev/1.2.0/Models-Reference/dense_retrieval/).
- Dockerfile optimisation (https://github.com/marqo-ai/marqo/pull/569). A pre-built Marqo base image results in reduced image layers and increased build speed, meaning neater docker pulls and an overall better development experience.


## Bug fixes and minor changes

- Major README overhaul (https://github.com/marqo-ai/marqo/pull/573). The README has been revamped with up-to-date examples and easier to follow instructions.
- New security policy (https://github.com/marqo-ai/marqo/pull/574).
- Improved testing pipeline (https://github.com/marqo-ai/marqo/pull/582 & https://github.com/marqo-ai/marqo/pull/586). Tests now trigger on pull request updates. This results in safer and easier merges to mainline.
- Updated requirements files. Now the `requirements.dev.txt` should be used to install requirements for development environments (https://github.com/marqo-ai/marqo/pull/569). Version pins for `protobuf` & `onnx` have been removed while a version pin for `anyio` has been added (https://github.com/marqo-ai/marqo/pull/581, & https://github.com/marqo-ai/marqo/pull/589).
- General readability improvements (https://github.com/marqo-ai/marqo/pull/577, https://github.com/marqo-ai/marqo/pull/578, https://github.com/marqo-ai/marqo/pull/587, & https://github.com/marqo-ai/marqo/pull/580)

## Contributor shout-outs

- A huge thank you to all our 3.5k stargazers!
- Shoutout to @vladdoster for all the useful spelling and grammar edits!
- Thanks everyone for keeping our [forum](https://community.marqo.ai/) bustling. Don't hesitate to keep posting your insights, questions, and feedback!


# Release 1.2.0

## New features

- Storage status in health check endpoint (https://github.com/marqo-ai/marqo/pull/555 & https://github.com/marqo-ai/marqo/pull/559). The `GET /indexes/{index-name}/health` endpoint's `backend` object will now return the boolean `storage_is_available`, to indicate if there is remaining storage space. If space is not available, health status will now return `yellow`. See [here](https://marqo.pages.dev/1.2.0/API-Reference/health/) for detailed usage.

- Score Modifiers search optimization (https://github.com/marqo-ai/marqo/pull/566). This optimization reduces latency for searches with the `score_modifiers` parameter when field names or weights are changed. See [here](https://marqo.pages.dev/1.2.0/API-Reference/search/#score-modifiers) for detailed usage.

## Bug fixes and minor changes

- Improved error message for full storage (https://github.com/marqo-ai/marqo/pull/555 & https://github.com/marqo-ai/marqo/pull/559). When storage is full, Marqo will return `400 Bad Request` instead of `429 Too Many Requests`.
- Searching with a zero vector now returns an empty list instead of an internal error (https://github.com/marqo-ai/marqo/pull/562).

## Contributor shout-outs

- A huge thank you to all our 3.3k stargazers!
- Thank you for all the continued discussion in our [forum](https://community.marqo.ai/). Keep all the insights, questions, and feedback coming!


# Release 1.1.0

## New features

- New field `numberOfVectors` in the `get_stats` response object (https://github.com/marqo-ai/marqo/pull/553). 
This field counts all vectors from all documents in a given index. See [here](https://docs.marqo.ai/1.1.0/API-Reference/stats/) for detailed usage.

- New per-index health check endpoint `GET /indexes/{index-name}/health` (https://github.com/marqo-ai/marqo/pull/552). 
This replaces the cluster-level health check endpoint, `GET /health`,
which is deprecated and will be removed in Marqo 2.0.0. See [here](https://docs.marqo.ai/1.1.0/API-Reference/health/) for detailed usage.

## Bug fixes and minor changes

- Improved image download validation and resource management (https://github.com/marqo-ai/marqo/pull/551). Image downloading in Marqo is more stable and resource-efficient now.

- Adding documents now returns an error when `tensorFields` is not specified explicitly (https://github.com/marqo-ai/marqo/pull/554). This prevents users accidentally creating unwanted tensor fields.

## Contributor shout-outs

- Thank you for the vibrant discussion in our [forum](https://community.marqo.ai/). We love hearing your questions and about your use cases.


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
