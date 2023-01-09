from typing import List
import openai
from marqo.s2_inference.reranking.enums import ResultsFields


def construct_context(results: dict, searchable_attributes: List, content_separator: str) -> str:
    """Generates the context string to be consumed by the prompt templates"""
    context = ""
    for i, result in enumerate(results[ResultsFields.hits]):
        content = content_separator.join([result[attrib] for attrib in searchable_attributes if attrib in result])
        context += f"Source {i}): {content}\n"
    return context


def prompt_to_essay(prompt: str, openai_args: dict):
    """ Process GPT-3 prompt and clean string . """
    response = openai.Completion.create(
        **openai_args, prompt=prompt
    )
    return response['choices'][0]['text'].strip()
