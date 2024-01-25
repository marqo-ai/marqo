import pandas as pd
from utilities import (
    marqo_prompt,
    extract_text_from_highlights,
    marqo_template,
    get_extra_data,
    reformat_npcs
)
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain.docstore.document import Document
from langchain.chains import LLMChain

load_dotenv()

if __name__ == "__main__":

    #############################################################
    #       0. Install Marqo
    #############################################################

    # run the following docker commands from the terminal to start marqo
    # docker rm -f marqo
    # docker pull marqoai/marqo:latest
    # docker run --name marqo -it --privileged -p 8882:8882 --add-host host.docker.internal:host-gateway marqoai/marqo:latest

    #############################################################
    #       1. Create some data
    #############################################################

    NPCs = [
        {
            "name": "Sara Lee",
            "backstory": "Sara was born in a small village in the mountains. She was always fascinated with nature and as soon as she was old enough, she left her village to study environmental science at a university. She now works as a park ranger.",
            "location": "The mountains",
            "occupation": "Park ranger",
            "family_history": "Sara is an only child and her parents were both farmers. Growing up close to nature instilled in her a deep respect and appreciation for the environment.",
            "work_history": "Sara worked as a research assistant for a university before becoming a park ranger.",
            "favorite_color": "Yellow",
            "hobbies": "Hiking, bird watching, and photography",
            "favorite_food": "Fruits and vegetables",
            "dislikes": "Loud noises",
        },
        {
            "name": "Jack Smith",
            "backstory": "Jack was born and raised in the city. He has always had a love for cars and as soon as he could he began working on them. He now runs his own successful auto repair shop.",
            "location": "The city",
            "occupation": "Auto mechanic",
            "family_history": "Jack has a younger sister and his father was also a mechanic who ran his own shop.",
            "work_history": "Jack worked as a mechanic at several auto repair shops before opening his own business.",
            "favorite_color": "Blue",
            "hobbies": "Working on cars, fishing, and playing video games",
            "favorite_food": "Steak",
            "dislikes": "Celery",
        },
        {
            "name": "Evelyn Parker",
            "backstory": "Evelyn grew up in a small town in the countryside. She always had a passion for cooking and eventually moved to the city to attend culinary school. She now works as a chef at a popular restaurant.",
            "location": "The city",
            "occupation": "Chef",
            "family_history": "Evelyn is the youngest of three siblings. Her parents were farmers and instilled in her a love for cooking with fresh ingredients.",
            "work_history": "Evelyn worked as a line cook at several restaurants before attending culinary school and becoming a head chef.",
            "favorite_color": "Green",
            "hobbies": "Cooking, gardening, and reading",
            "favorite_food": "Seafood",
            "dislikes": "Cilantro",
        }
    ]

    df = pd.DataFrame(reformat_npcs(NPCs))
    print(df.head())

    # make the data python dicts
    documents = df.to_dict(orient='records')

    #############################################################
    #       2. Setup Marqo
    #############################################################

    import marqo
    from marqo import Client

    marqo.set_log_level('WARN')

    mq = Client()

    index_name = "npc-chat"

    try:
        mq.index(index_name).delete()
    except:
        pass

    index_settings = {
        "normalizeEmbeddings": True,
        "textPreprocessing": {
            "splitLength": 5,
            "splitOverlap": 1,
            "splitMethod": "sentence"
        },
    }

    # create the index - if no settings are present then sensible defaults are used
    mq.create_index(index_name, settings_dict=index_settings)
    res = mq.index(index_name).add_documents(documents, tensor_fields=["name", "backstory", "location", "occupation",
                                                                       "family_history", "work_history",
                                                                       "favorite_color",
                                                                       "hobbies", "favorite_food", "dislikes"])

    #############################################################
    #       3. Regular NPC superhero
    #############################################################

    # select a character
    persona = "Evelyn Parker"

    # we pre-opulate them here to complete a conversation but it can easily be made interactive
    human_questions = ["hi, what is your name?",
                       "wow, what are some of your favorite things to do?",
                       "are you scared of anything?",
                       "where did you grow up?",
                       "what do you dislike?"]

    history = []
    template = marqo_template()
    prompt = marqo_prompt(template)

    # how many pieces of context to use
    n_history = 2

    # setup the LLM API call
    llm = OpenAI(temperature=0.9)

    for question in human_questions:
        history.append(f"\nHUMAN:{question}")
        print(history[-1])

        # search for background related to the question
        results = mq.index(index_name).search(question, filter_string=f"name:({persona})", limit=20)

        # optionally crop the text to the highlighted region to fit within the context window
        highlights, texts = extract_text_from_highlights(results, token_limit=150)

        # add the truncated/cropped text to the data structure for langchain
        summaries = [Document(page_content=f"Source [{ind}]:" + t) for ind, t in enumerate(texts[:n_history])]

        # get the conversation history
        chain_qa = LLMChain(llm=llm, prompt=prompt)

        llm_results = chain_qa.invoke({"summaries": summaries, "conversation": "\n".join(history)},
                                      return_only_outputs=False)

        history.append(llm_results['text'])
        print(history[-1])

    #############################################################
    #       3. IRONMAN
    #############################################################

    persona = "Evelyn Parker"

    # add some more info
    extra_docs = [{"text": text, "name": persona} for text in get_extra_data()]
    res = mq.index(index_name).add_documents(extra_docs, tensor_fields=["name", "backstory", "location", "occupation",
                                                                        "family_history", "work_history",
                                                                        "favorite_color",
                                                                        "hobbies", "favorite_food", "dislikes"])

    # we pre-opulate them here to complete a conversation but it can easily be made interactive
    human_questions = ["hi, what is your name?",
                       "wow, what are some of your favorite things to do?",
                       "are you scared of anything?",
                       "where did you grow up?",
                       "what do you dislike?"]

    history = []
    template = marqo_template()
    prompt = marqo_prompt(template)

    # how many pieces of context to use
    n_history = 2

    for question in human_questions:
        history.append(f"\nHUMAN:{question}")
        print(history[-1])

        # search for background related to the question
        results = mq.index(index_name).search(question, filter_string=f"name:({persona})", limit=20)

        # optionally crop the text to the highlighted region to fit within the context window
        highlights, texts = extract_text_from_highlights(results, token_limit=150)

        # add the truncated/cropped text to the data structure for langchain
        summaries = [Document(page_content=f"Source [{ind}]:" + t) for ind, t in enumerate(texts[-n_history:])]

        # get the conversation history
        chain_qa = LLMChain(llm=llm, prompt=prompt)

        llm_results = chain_qa({"summaries": summaries, "conversation": "\n".join(history)}, return_only_outputs=False)

        history.append(llm_results['text'])
        print(history[-1])
