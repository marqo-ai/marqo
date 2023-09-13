# From "iron manual" to "ironman" - Augmenting GPT for fast editable memory to enable context aware question & answering

TL:DR We show how [Marqo](https://github.com/marqo-ai/marqo) can be used as an update-able and domain specific memory to [GPT](https://en.wikipedia.org/wiki/GPT-3) to perform question and answering for products and chat agents. We also show how reference and hallucination checks can be easily implemented. A walk-through with images and animations is provided along with the [code](https://github.com/marqo-ai/marqo/tree/mainline/examples/GPT-examples) to reproduce.

<p align="center">
  <img src="assets/1 VwVrPVOcdN7BeMUrT2bNAQ.png"/>
</p>

## 1. Product Q & A - "Iron Manual"

<p align="center">
  <img src="assets/1 FcXqgXdF3nQQi2gexNCrdw.gif"/>
</p>

## 2. NPC/chat agent with editable history - "Ironman"

<p align="center">
  <img src="assets/1 e1JdfEgAngObm_TxwKBYeQ.gif"/>
</p>

## Introduction

LLM’s can be used for many tasks with little (few-shot) to no (zero-shot) training data. A single LLM can be used for tasks like summarization, translation, question and answering, and classification. Despite LLM’s recent success there are still some limitations. For example, after the models are trained they are not easily updatable with new information. They also have a fixed input length. This places restrictions on the amount of context they can have inserted when being prompted. To overcome these limitations, we show how an external knowledge base can be used as part of the LLM to provide a fast and editable memory (i.e. document store) for it to draw from.

<p align="center">
  <img src="assets/1 2lK1Q99fpKke6vPFwqPzEA.png"/>
</p>

## Use case 1 - Product Q&A

For the first use case, GPT is paired with Marqo to create a powerful search function for product documentation. This allows question and answering of its features. It is also able to provide a nice compact answer that is easy to read.

### 1.1 The product documents

To test the question answering capabilities, an “in the wild” use case was desired. A paper manual for a recently purchased clothes iron was selected. If already digitized text is available, this step can be skipped.

<p align="center">
  <img src="assets/1 cuLqghO8BeEs_wj7ZhDt3Q.png"/>
</p>

The manual is a particularly dry read. It consists of 7-pages of information related to the iron. Including information regarding its safe operation and maintenance.

### 1.2 Preparing the documents

Since the manual was on some paper, it needs to be digitized. AWS Textract was used to perform optical character recognition (OCR). The pages were two-columned which provided a challenge as the OCR output is left-to-right, causing the text to be intermingled. Bounding boxes are provided from the OCR output which would allow conceptual grouping of the text, however this was going to take too long. Instead, the OCR was performed again but with half the text blocked off by another piece of paper.

<p align="center">
  <img src="assets/1 7sgXp44TwyV9PgEHlrdNNw.png"/>
</p>

After scanning, there were seven documents, each representing a column of text from the manual. Below is an example of the text after OCR.

```python
"""
Your iron has an Anti-Drip system, Anti-Scales system
and Auto-Off function.
Anti-Drip system: This is to prevent water from escaping
from the soleplate when the iron is cold. During use, the
anti-drip system may emit a loud 'clicking' sound,
particularly when heating up or cooling down. This is
normal and indicates that the system is functioning
correctly.
Anti-Scale system: The built-in anti-scale cartridge is
designed to reduce the build-up of lime scale which
occurs during steam ironing and will prolong the
working life of your iron. The anti-calc cartridge is an
integral part of the water tank and does not need to be
replaced.
Auto-Off function: This feature automatically switches
off the steam iron if it has not been moved for a while.
"""
```

## 1.3 Indexing the documents
After creating a digital copy of the product manual, the next step is to index them into Marqo. Marqo embeds the documents using an encoder and allows for fast and efficient retrieval of relevant documents. Marqo provides both embedding based and lexical based retrieval. These retrieved documents are then going to be passed into a prompt for GPT. GPT is then asked to answer the query with respect to the retrieved documents (the “sources”).

## 1.3.1 Installing Marqo
We first install Marqo and the Marqo python client,

```bash
docker pull marqoai/marqo:0.0.12;
docker rm -f marqo;
docker run --name marqo -it --privileged -p 8882:8882 --add-host host.docker.internal:host-gateway marqoai/marqo:0.0.12
pip install marqo
```

## 1.3.2 Indexing documents
We then need to index the pages we have for the product. They need to be formatted as a python dictionary for ingestion.

```python
document1 = {"text":"Auto-Off function: This feature automatically switches
                 off the steam iron if it has not been moved for a while.",
             "source":"page 1"}
# other document content left out for clarity
documents = [document1, document2, document3, document4, document5]
```

Once the documents are prepared, we can start indexing them using the python client. If no index settings are present, the default encoder is used.

```python
from marqo import Client
mq = Client()
index_name = "iron-docs"
mq.create_index(index_name)
```

## 1.3.3 Searching documents
At this point, Marqo can be used to search over the document embeddings using an approximate nearest neighbor algorithm (HNSW).
```python
results = mq.index(index_name).search("what is the loud clicking sound?")
```
or using lexical search which uses BM25.
```python
results = mq.index(index_name).search("what is the loud clicking sound?",  
                                                  search_method="LEXICAL")
```
## 1.4 Connecting Marqo to GPT
The documents (product manual) can now be searched. Searching and retrieving relevant documents will provide the context for GPT to create a final answer. GPT requires an API key and can be obtained from the OpenAI website. The key then needs to be set as an environment variable.

```bash
export OPENAI_API_KEY="..."
```

## 1.4.1 Prompt creation
The first thing that needs to be done is to create a prompt. There are a plethora of examples to draw from here but something like the following will get good results.

```python
template = """
Given the following extracted parts of a long document ("SOURCES") and a question ("QUESTION"), create a final answer one paragraph long.
Don't try to make up an answer and use the text in the SOURCES only for the answer. If you don't know the answer, just say that you don't know.
QUESTION: {question}
=========
SOURCES:
{summaries}
=========
ANSWER:
"""
```
Here we instruct GPT to answer based on the context and not make anything up (this may not be perfect though). The question and answer is then inserted along with the context (“summaries”).

To save time, we will use Langchain to help with the communication with GPT. Langchain can make it easy to setup interactions with LLM’s and removes a lot of the boiler plate code that would otherwise be required.
```python
pip install langchain
from langchain.prompts import PromptTemplate
prompt = PromptTemplate(template=template, input_variables=["summaries", "question"])
```

## 1.4.2 Preparing the context
In order to connect GPT to Marqo, we need to format the results so they can be easily inserted into the prompt that was just created.

```python
from langchain.docstore.document import Document
results = client.index(index_name).search(question)
text = [res['_highlights']['text'] for res in results['hits']]
docs = [Document(page_content=f"Source [{ind}]:"+t) for ind,t in enumerate(texts)]
```
To start with, we take just the highlights from Marqo. This is convenient because they are small pieces of text. They fit within the prompt and do not occupy too many tokens. Token limits matter because GPT (and LLM’s in general) will have a context length and often charge by token. This is the maximum number of tokens that can be used for the input. The more context, the larger the text (and background) that GPT can use. The drawback here is that the highlights might be too short to accurately answer the question.

## 1.4.3 Token aware context truncating (optional)
<p align="center">
  <img src="assets/1 e9aMy1npv-O8Gy87RMivWw.png"/>
</p>

To help with token limits while also having control over the input context length - a "dilation" procedure around the highlight can be performed to allow for more context. This means that some text before and after the highlight is included to allow for greater context for GPT. This is also very helpful as the pricing models for these models can be per token.

```python
highlights, texts = extract_text_from_highlights(results, token_limit=150)
docs = [Document(page_content=f"Source [{ind}]:"+t) for ind,t in enumerate(texts)]
```
The next step is to provide the prompt with formatted context.

## 1.4.4 GPT inference
Now we have the prepared documents and prompt, we can call GPT using Langchain. We initiate an OpenAI class which communicates with the GPT API.

```python
from langchain.chains import LLMChain
llm = OpenAI(temperature=0.9,  model_name = "text-davinci-003")
chain_qa = LLMChain(llm=llm, prompt=prompt)
llm_results = chain_qa({"summaries": docs, "question": results['query']}, return_only_outputs=True)
```

The result is a dictionary with the text output from GPT. This essentially completes the required steps to augment GPT with an external knowledge base. However, we can add another feature to include and score sources of information that were used in the answer. This can be useful since LLM’s are known to hallucinate details. Providing the original sources can be used to check the output of the LLM’s results. In the next section we show how a re-ranker from two-stage retrieval can be repurposed to check which sources were used and provide a score.

## 1.4.5 Rating sources
After we have received a response in from the LLM, we can score the sources with respect to the LLM's response. This is in contrast to other methods that get the LLM themselves to cite the sources. From the experience here, that method was sometimes unreliable.

<p align="center">
  <img src="assets/1 hyVjvO2E_zADn1829W4oXw.png"/>
</p>

For the proposed method, we take a re-ranker (sentence-transformers cross-encoder) which would normally score each (query, document) pair to re-rank search results. Instead here, we score each (llm_result, document) pair. This provides a score for the "relevency" of the LLM's response with the provided sources. The idea being, the ones that were used will be most similar to the response.

```python
scores = predict_ce(llm_results['text'], texts)
```
and we can see scores for each piece of context with the LLM response.

<p align="center">
  <img src="assets/1 BTmMOgmzwuYhFw-Uzn65SQ.png"/>
</p>

The model is a classification model. A score of 1 is exact match and 0 is not a match. This method is not perfect though so some care should be taken. For example, GPT can be quite verbose when it does not know an answer. This response can include the original question and will cause false positives when scoring. Fine-tuning the re-ranker would probably reduce the false positives considerably.

# Use case 2 - conversational agents with a story
The second use case deals with a conversational agent that can draw on history (or background)  as context to answer questions. This could be used for creating NPC's with a backstory or other chat agents that may need past context.

<p align="center">
  <img src="assets/1 cntJoKdSHk0zGzrlLW9tYw.gif"/>
</p>

## 2.1 Indexing NPC data
Now we have indexed the documents, we can search over them. In this case the documents are the backstories and the search is used context for the conversation. This search and retrieve step will provide the context for GPT to create a final answer.

## 2.1.1 NPC data
Here is an example of what the documents look like for the NPC's:
```python
document1 = {"name":"Sara Lee", "text":"my name is Sara Lee"}
document2 = {"name":"Jack Smith", "text":"my name is Jack Smith"}
document3 = {"name":"Sara Lee", "text":"Sara worked as a research assistant for a university before becoming a park ranger."}
documents = [document1, document2, document3]
```

## 2.1.2 Indexing the data 
Now we can index the data. We need to get the Marqo client and create an index name.
```python
from marqo import Client
mq = Client()
index_name = "iron-docs"
mq.create_index(index_name)
```
Now we index the documents
```python
results = mq.index(index_name).add_documents(documents, tensor_fields = ["name", "text"], auto_refresh=True)
```

We can search and see what comes back.
```python
results = mq.index(index_name).search("sara lee")
```

Which gives the desired output
```python
In [32]: res['hits'][0]['_highlights']
Out[32]: {'name': 'Sara Lee'}
```

Different characters can be easily selected (filtered). This means only their background can be searched.
```python
persona = "Jack Smith"
results = mq.index(index_name).search('what is your hobby', filter_string=f'name:({persona})')
```

## 2.2 Connecting Marqo to GPT
Now we have indexed the documents, we can search over them. In this case the documents are the backstories and the search is used context for the conversation. This search and retrieve step will provide the context for GPT to create a final answer.

## 2.2.1 Prompt creation
We use a prompt that contains some context and sets the stage for the LLM conversationalist.
```python
template = """
The following is a conversation with a fictional superhero in a movie. 
BACKGROUND is provided which describes some of the history and powers of the superhero. 
The conversation should always be consistent with this BACKGROUND. 
Continue the conversation as the superhero in the movie. 
You are very funny and talkative and **always** talk about your superhero skills in relation to your BACKGROUND.
BACKGROUND:
=========
{summaries}
=========
Conversation:
{conversation}
"""
```
Here we instruct GPT to answer for the character based on the background and to reference it where possible. Langchain is then used to create the prompt,

```python
pip install langchain
from langchain.prompts import PromptTemplate
prompt = PromptTemplate(template=template, input_variables=["summaries", "conversation"])
```

## 2.2.2 Preparing the context
Here we truncate the context around the highlight from the previous search using Marqo. We use the token aware truncation which adds context from before and after the highlight.

```python
highlights, texts = extract_text_from_highlights(results, token_limit=150)
docs = [Document(page_content=f"Source [{ind}]:"+t) for ind,t in enumerate(texts)]
```
The next step is to provide the prompt with formatted context.

## 2.2.3 GPT inference
Now we have the prepared documents and prompt, we can call GPT using Langchain. We initiate an OpenAI class which communicates with the GPT API,

```python
from langchain.chains import LLMChain
lm = OpenAI(temperature=0.9,  model_name = "text-davinci-003")
chain_qa = LLMChain(llm=llm, prompt=prompt)
llm_results = chain_qa({"summaries": docs, "conversation": "wow, what are some of your favorite things to do?", return_only_outputs=True)
```
The result is a dictionary with the text output from GPT. For example this is the response after the first text from the human,

```python
 {'conversation': 'HUMAN:wow, what are some of your favorite things to do?',
 'text': "SUPERHERO:I really enjoy working on cars, fishing, and playing video games. Those are some of the things that I like to do in my free time. I'm also really into maintaining and fixing stuff - I guess you could say it's one of my superhero powers! I have a lot of experience as an auto mechanic, so I'm really good at diagnosing and fixing problems with cars."}
 ```

which aligns well with the background which was,

```python
['my hobbies is Working on cars, fishing, and playing video games',
 'my favorite food is Steak',
 'my favorite color is Blue']
```

## 2.3 Making it conversational
The next step is to do some iterative prompting and inference to create a chat. We do this by iteratively updating the prompt with the question, searching across the background, formatting the context, calling the LLM and appending the result to the chat sequence (full code [here](https://github.com/marqo-ai/marqo/tree/mainline/examples/GPT-examples)).

```python
# how many background pieces of information to use
n_background = 2
# we keep track of the human and superhero responses
history.append(f"\nHUMAN:{question}")
# search for background related to the question
results = mq.index(index_name).search(question, filter_string=f"name:({persona})", searchable_attributes=['text'], limit=20) 
# optionally crop the text to the highlighted region to fit within the context window
highlights, texts = extract_text_from_highlights(results, token_limit=150)
# add the truncated/cropped text to the data structure for langchain
summaries = [Document(page_content=f"Source [{ind}]:"+t) for ind,t in enumerate(texts[:n_background])]
# inference with the LLM
chain_qa = LLMChain(llm=llm, prompt=prompt)
llm_results = chain_qa({"summaries": summaries, "conversation": "\n".join(history)}, return_only_outputs=False)
# add to the conversation history
history.append(llm_results['text'])
```
We can now see the conversation and the agent drawing on its retrieved background.

<p align="center">
  <img src="assets/1 WAa1eHpMekmkOG04ZxdAKA.png"/>
</p>

This fits nicely with the background we gave the character,
```python
{'name': 'Evelyn Parker', 'text': 'my name is Evelyn Parker'},
{'name': 'Evelyn Parker', 'text': 'my location is The city'},
{'name': 'Evelyn Parker', 'text': 'my work_history is Evelyn worked as a line cook at several restaurants before attending culinary school and becoming a head chef.'},
{'name': 'Evelyn Parker',
  'text': 'my hobbies is Cooking, gardening, and reading'},
{'name': 'Evelyn Parker', 'text': 'my favorite_food is Seafood'},
{'name': 'Evelyn Parker', 'text': 'my dislikes is Cilantro'}
```

## 2.4 Editing the characters background
We can patch, delete or add documents for the agents background with Marqo. Lets add something from the previous example,
```python
from iron_data import get_extra_data
extra_docs = [{"text":text, "name":persona} for text in get_extra_data()]
res = mq.index(index_name).add_documents(extra_docs, tensor_fields = ["name", "text"], auto_refresh=True)
```
This adds some of the safety information from the iron manual. We will also take the bottom ranked results (i.e least relevant) to make it interesting. The following is the conversation - we can see it weaving its new background into the story nicely!

<p align="center">
  <img src="assets/1 e1JdfEgAngObm_TxwKBYeQ.gif"/>
</p>


## Conclusion
We have shown how it is easy to make product question and answering and chat agents with an editable background using LLM’s like GPT and [Marqo](https://github.com/marqo-ai/marqo). We also showed how the limits of context length can be overcome by judicious truncation of the text. Reference scoring can be used to help verify the output. Although the results are really encouraging some care should still be made. GPT still has a habit of hallucinating results even with strong instructions and reference checks. If you are interested in combining GPT (or LLM’s in general) with Marqo — check out the [github](https://github.com/marqo-ai/marqo). Finally, if you are interested in running this in production, sign up for our [cloud](https://q78175g1wwa.typeform.com/to/d0PEuRPC).
