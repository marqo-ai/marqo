import openai
import os
openai.api_key = os.environ.get('OPENAI_API_KEY', None)
from functools import partial
from marqo.s2_inference.reranking.enums import ResultsFields

class LLM:

    def __init__(self):
        pass

    def load(self):
        pass

    def generate(self):
        pass

class GPT3(LLM):

    def __init__(self, api_key=None):
        super().__init__()
        self.api_key = api_key

    def load(self, task):
        self.task = task

    def generate(self, context):
        return prompt_to_essay(context)

def format_results(results, promptable_attributes, task, top_k=5):
    hits = results[ResultsFields.hits]

    # get attributes from results


def get_context_prompt(question, context):
    """ GPT3 prompt without text-based context from marqo search. """
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