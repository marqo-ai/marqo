from marqo.s2_inference.types import Dict, List, Optional
import openai
from marqo.s2_inference.reranking.enums import Columns, ResultsFields
from marqo.s2_inference.errors import RerankerError, RerankerNameError


def construct_context(results: dict, searchable_attributes: List, content_separator: str) -> str:
    """Generates the context string to be consumed by the prompt templates"""
    context = ""
    for i, result in enumerate(results[ResultsFields.hits]):
        content = content_separator.join([result[attrib] for attrib in searchable_attributes if attrib in result])
        context += f"Source {i}): {content}\n"
    return context


def prompt_to_essay(prompt: str, openai_key: str):
    """ Process GPT-3 prompt and clean string . """
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        temperature=0.0,
        max_tokens=256,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        api_key=openai_key
    )
    return response['choices'][0]['text'].strip().replace('\n', ' ')


class GptReranker:

    task_name = None

    def __init__(self, reranker_properties: dict):
        try:
            self.openai_key = reranker_properties['api_key']
            self.reranker_properties = reranker_properties
        except (KeyError, TypeError):
            raise RerankerError("OpenAI API Key not found in reranker properties")


class GptQuestionAnswering(GptReranker):

    task_name = "gpt3-qa"

    @staticmethod
    def prompt_template_question_answer(question: str, context: str):
        """ GPT3 prompt with text-based context. """
        return f'Background: \n{context}\n\nQuestion: {question}\n\nAnswer:'

    def rerank(self, query: str, search_result: Dict, searchable_attributes: List[str]):
        context = construct_context(
            results=search_result, searchable_attributes=searchable_attributes, content_separator=" | "
        )
        essay = prompt_to_essay(
            self.prompt_template_question_answer(question=query, context=context),
            openai_key=self.openai_key
        )
        search_result['reranker_output'] = essay


class GptFreeform(GptReranker):

    task_name = "gpt3-freeform"

    @staticmethod
    def prompt_template_freeform(context: str, your_text_here: str):
        """ GPT3 prompt that adds allows for freeform prompts"""
        return f'Background: \n{context}\n\n{your_text_here}'

    def rerank(
            self, query: str, search_result: Dict, searchable_attributes: List[str],
        ):
        context = construct_context(
            results=search_result, searchable_attributes=searchable_attributes, content_separator=" | "
        )

        essay = prompt_to_essay(
            self.prompt_template_freeform(context=context, your_text_here=self.reranker_properties['prompt']),
            openai_key=self.openai_key
        )
        search_result['reranker_output'] = essay


class GptSummariser(GptReranker):

    task_name = "gpt3-summarise"

    @staticmethod
    def prompt_template_summary(context: str):
        """ GPT3 prompt with text-based context. """
        return f'Background: \n{context}\n\nSummary:'

    def rerank(
            self, query: str, search_result: Dict, searchable_attributes: List[str],
    ):
        context = construct_context(
            results=search_result, searchable_attributes=searchable_attributes, content_separator=" | "
        )

        essay = prompt_to_essay(
            self.prompt_template_summary(context=context),
            openai_key=self.openai_key
        )
        search_result['reranker_output'] = essay


class GptReorder(GptReranker):

    task_name = "gpt3-reorder"

    @staticmethod
    def prompt_template_reorder(query: str, context: str):
        """ GPT3 prompt with text-based context. """
        return (f'Background: \n{context}\n\nQuery: {query}\n\n'
                f'Reorder the sources based on relevancy to the query:')

    def rerank(
            self, query: str, search_result: Dict, searchable_attributes: List[str],
    ):
        context = construct_context(
            results=search_result, searchable_attributes=searchable_attributes, content_separator=" | "
        )

        essay = prompt_to_essay(
            self.prompt_template_reorder(query=query, context=context),
            openai_key=self.openai_key
        )
        search_result['reranker_output'] = essay
