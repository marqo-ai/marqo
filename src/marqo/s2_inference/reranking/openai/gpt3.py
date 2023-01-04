from marqo.s2_inference.types import Dict, List, Optional
import openai
from marqo.s2_inference.reranking.enums import Columns, ResultsFields


def gpt_question_answer_template(question: str, context: str):
    """ GPT3 prompt with text-based context. """
    return f'Background: \n{context}\n\nQuestion: {question}\n\nAnswer:'


def prompt_to_essay(prompt):
    """ Process GPT-3 prompt and clean string . """
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        temperature=0.0,
        max_tokens=256,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    return response['choices'][0]['text'].strip().replace('\n', ' ')


class GptReranker:

    def __init__(self, api_key):
        openai.api_key = api_key


class GptQuestionAnswering(GptReranker):

    def rerank(self, query: str, search_result: Dict, searchable_attributes: List[str]):
        context = ""

        for i, result in enumerate(search_result[ResultsFields.hits]):
            content = " | ".join([result[attrib] for attrib in searchable_attributes if attrib in result])
            context += f"Source {i}): {content}\n"

        essay = prompt_to_essay(
            gpt_question_answer_template(question=query, context=context)
        )
        search_result['reranker_output'] = essay


class GptFreeform(GptReranker):
    """TODO"""

