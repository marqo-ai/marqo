import marqo
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

TEMPLATE = """
You are a question answerer, given the CONTEXT provided you will answer the QUESTION (also provided).
If you are not sure of the answer then say 'I am sorry but I do not know the answer'
Your answers should two to five sentences in length and only contain information relevant to the question. You should match the tone of the CONTEXT.
The beginnings of the CONTEXT should be the most relevant so try and use that wherever possible, it is important that your answers a factual and don't make up information that is not in the CONTEXT.


CONTEXT:
=========
{context}
QUESTION:
=========
{question}
"""


def answer_question(
    query: str,
    limit: int,
    index: str,
    mq: marqo.Client,
) -> str:
    print("Searching...")
    results = mq.index(index).search(
        q=query,
        limit=limit,
    )
    print("Done!")

    context = ". ".join([r["transcription"] for r in results["hits"]])

    prompt = PromptTemplate(template=TEMPLATE, input_variables=["context", "question"])
    llm = OpenAI(temperature=0.9, model_name="text-davinci-003")
    chain_qa = LLMChain(llm=llm, prompt=prompt)
    llm_results = chain_qa(
        {"context": context, "question": query}, return_only_outputs=True
    )
    return llm_results["text"]
