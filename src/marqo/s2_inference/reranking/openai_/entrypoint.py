import copy

from marqo.s2_inference.errors import RerankerError
from marqo.s2_inference.reranking.openai_ import gpt3, gpt3_pre_summarise


def gpt_entrypoint(model_name: str, reranker_properties: dict, search_result: dict, query: str,
                   searchable_attributes: list):
    lowered_model_name = model_name.lower()

    tasks = {
        gpt3.GptQuestionAnswering.task_name: gpt3.GptQuestionAnswering,
        gpt3.GptFreeform.task_name: gpt3.GptFreeform,
        gpt3.GptSummariser.task_name: gpt3.GptSummariser,
        gpt3.GptReorder.task_name: gpt3.GptReorder
    }
    task_name = lowered_model_name.split("/")[-1]
    try:
        reranker = tasks[task_name](reranker_properties=reranker_properties)
    except KeyError:
        raise RerankerError(
            f"Encountered unknown OpenAI task: `{model_name}`. Please check our documentation for available "
            f"OpenAI tasks: https://docs.marqo.ai/latest")
    try:
        api_key = reranker_properties['api_key']
    except KeyError:
        raise RerankerError(f"OpenAI API key not found in reRankerProperties")
    try:
        if "pre_summarise" in reranker_properties:
            if reranker_properties['pre_summarise'] is True:
                default_pre_summariser_args = gpt3_pre_summarise.PreSummariseArgs(api_key=api_key)
                pre_summariser_args = default_pre_summariser_args._asdict()
            else:  # dict
                pre_summariser_args = copy.deepcopy(reranker_properties['pre_summarise'])
                pre_summariser_args['api_key'] = api_key
            pre_summariser = gpt3_pre_summarise.GptPreSummariser(pre_summariser_args)
            try:
                pre_summarised_results = pre_summariser.pre_summarise(
                    query=query, search_result=search_result, searchable_attributes=searchable_attributes)
            except Exception as e:
                raise RerankerError(message="pre_summariser stage: " + str(e)) from e
    except TypeError:
        raise RerankerError(
            f"reRankerProperties not found. For GPT rerankers, reRankerProperties must be specified, with the"
            f" OpenAI API key as api_key"
        )
    if "pre_summarise" in reranker_properties:
        essay = reranker.rerank(
            query=query, search_result=pre_summarised_results, searchable_attributes=["pre_summary"])
        return essay
    else:
        essay = reranker.rerank(
            query=query, search_result=search_result, searchable_attributes=searchable_attributes)
        return essay
