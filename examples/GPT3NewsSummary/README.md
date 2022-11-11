# Using Marqo to power topical news summarisation

## Who am I?

My name is Iain Mackie and I lead NLP investments at <a href="https://thecreatorfund.com/">Creator Fund</a> and previously worked in Quant trading. I am currently finishing my PhD in neural search systems at the <a href="https://grilllab.ai/">University of Glasgow</a> and recently won the $500k grand prize at the <a href="https://www.digit.fyi/next-gen-ai-assistant-from-glasgow-uni-wins-amazon-taskbot-challenge/">Alexa TaskBot Challenge</a>.🤖 

Given my passion for startups and NLP...I could not be happier to announce Creator Fund's £650,000 Pre-Seed investment into Marqo. Creator Fund (European Deep Tech) is joined by Blackbird (top VC in Australia/NZ), Activation Fund (US SaaS VC), and many high-profile angels. A global start to a global journey! 🚀    

## What is neural search?

Search, or information retrieval, is the study of retrieving relevant information given a user query. The most obvious example is Google, where a user inputs a query and Google will return a ranked list of web documents. Search is complex because:
<ol>
  <li>Indexes can have millions or billions of documents.</li>
  <li>Judgements mapping user requests can be incredibly sparse.</li>
  <li>Documents can be long with many different modalities (text, structured information, images, and videos).</li>
</ol>

Historically due to scale and computations, search systems were based on inverted indexes, where each word is stored in a database, making it easy to run simple algorithms over billions of documents. For example, for the query "good startup", search algorithms such as BM25 would return results with "good" and "startup" words. However, with the explosion of large language models (LLMs) with billions of parameters (SBERT, BERT, T5, GPT3, etc.), neural search systems have vastly improved search effectiveness. Specifically,  LLMs can draw semantic connections between similar terms, i.e. between "good startup" and words like "scalable", "strong founders", and "high growth". In addition, there has also been breakthroughs in multimodal neural models (CLIP, DALLE, etc.) that allow for aligned multimodal indexong and search. So, for example, you could search for images or videos that align with our "good startup" query (maybe the picture of Marqo below!). 

The problem is that multimodal search can be extremely beneficial in many multimodal applications (search engines, image recommendation, product search, etc.). Still, effective and robust search systems require technical experts to implement and keep up-to-date with this fast-moving search environment. Now that's where Marqo comes in...

## What is Marqo?

<a href="https://www.marqo.ai">Marqo</a> is developing "tensor search for humans" that improves search relevance for multimodal search applications (text 💬, image 🖼️, video 📽️), while being simple to set up and scale. Whereas many current neural databases require specialised search engineers, Marqo is a simple Python API where developers can index and query within seconds. Specifically, Marqo effortlessly combines the robustness and expressiveness of OpenSearch, allowing for complex filters and lexical search, with your favourite multimodal neural search models such as S-Bert and CLIP. Furthermore, Marqo's cloud platform enables customers to pay-per-query and reduces costs due to pooled resources ☁️. They are already powering child-friendly search engines, NFT recommendation systems, and much much more! 

More than just the technology, we are backing two incredible founders. Jesse has a PhD in Physics from La Trogue, Postdocs at UCL and Stanford, and was a Lead Machine Learning Scientist at Alexa and Amazon Robotics. Tom Hamer has software degrees from Australian National University and Cambridge, and was a software engineer within AWS' ElasticSearch and Database teams. Together, they have built a team of search enthusiasts with the passion and skillset to enable developers worldwide effortless access to neural search. 

<p align="center">
    <img src="https://github.com/iain-mackie/marqo-gpt3/blob/main/assets/marqo_photo.jpeg" alt="Jesse & Tom" width="420" height="600" >

Marqo's open-source Github <a href="https://github.com/marqo-ai/marqo">Github</a> has reached 1.1k+ 🌟 in 6 weeks (top 10 trending libraries!). They have also launched the cloud beta that allows customers to pay-per-query and reduces costs due to pooled resources (<a href="https://q78175g1wwa.typeform.com/to/d0PEuRPC">join waiting list</a>). Lastly, they are building a community of search enthusiasts tackling different problems (<a href="https://join.slack.com/t/marqo-community/shared_invite/zt-1d737l76e-u~b3Rvey2IN2nGM4wyr44w">Slack</a>).

## Topical news summarisation 

Now for the fun bit...

I wanted to build a fun search application within minutes to show the ease and power of Marqo. I decided to build a news summarisation application, i.e. answer questions like "What is happening in business today?" that synthesises example news corpus (<a href="https://github.com/iain-mackie/marqo-gpt3/blob/main/assets/news.py">link</a>).

The plan is to use Marqo's search to provide useful context for a generation algorithm; we use OpenAI's GPT3 API (<a href="https://openai.com/api/">link</a>). This is more formally called "retrieval-augmented generation" and helps with generation tasks that require specific knowledge that the model has not seen during training. For example, company-specific documents and news data that's "in the future". Overview of what we're planning: 

<p align="center">
    <img src="https://github.com/iain-mackie/marqo-gpt3/blob/main/assets/overview.png" alt="Overview" width="500" height="350" >


Thus, we can see the problem when we solely ask GPT3, "What is happening in business today?" It does not know and thus generates a generic response:
```
Question: What is happening in business today?

Answer:
There is a lot happening in business today. The economy is slowly recovering from the recession, and businesses are starting to invest again...
```
In fact, anyone following the financial markets knows 'the "economy is slowly recovering" and "businesses are starting to invest again" is completely wrong!!

To solve this, we need to start our Marqo docker container, which creates a Python API we'll interact with during this demo:
```
docker pull marqoai/marqo:0.0.6;
docker rm -f marqo;
docker run --name marqo -it --privileged -p 8882:8882 --add-host host.docker.internal:host-gateway marqoai/marqo:0.0.6
```

Next, let's look at our example news documents corpus, which contains BBC and Reuters news content from 8th and 9th of November. We use "_id" as Marqo document identifier, the "date" the article was written, "website" indicating the web domain, "Title" for the headline, and "Description" for the article body:
```
[
{
    '_id': '2',
    'date': '2022-11-09',
    'website': 'www.bbc.co.uk',
    'Title': 'COP27: Time to pay the climate bill - vulnerable nations',
    'Description': 'Leaders of countries flooded or parched due to climate change are pleading at the COP27 summit...'
},...
]
```  

We then index our news documents that manage both the lexical and neural embeddings. By default, Marqo uses SBERT from neural text embeding and has complete OpenSearch lexical and metadata functionality natively. 
```  
from news import MARQO_DOCUMENTS

DOC_INDEX_NAME = ''news-index'

print('Establishing connection to marqo client.')
mq = marqo.Client(url='http://localhost:8882')

print('Indexing documents')
mq.index(DOC_INDEX_NAME).add_documents(MARQO_DOCUMENTS)
```  

Now we have indexed our news documents, we can simply use Marqo Python search API to return relevant context for our GPT3 generation.  For query "q", we use the question and want to match news context based on the "Title" and "Description" text. We also want to filter our documents for "today", which was '2022-11-09'.   
```  
question = 'What is happening in business today?'
date = '2022-11-09'
results = mq.index(DOC_INDEX_NAME).search(
					q=question,
                    searchable_attributes=['Title', 'Description'],
					filter_string=f"date:{date}"
                    limit=5)
```  

Next, we insert Marqo's search results into GPT3 prompt as context, and we try generating an answer again::
```
Background: 
Source 0) M&S warns of 'gathering storm' as shoppers squeezed || Marks and Spencer has warned of a "gathering storm" of higher costs for retailers and pressure on household budgets as it reported a fall in profits for the first half of the year. The High Street giant said trading would become "more challenging" after it revealed its profits dropped by 24%. It said "all parts" of retail would be affected... 
Source 1) Facebook-owner Meta to cut 11,000 staff || Meta, which owns Facebook, Instagram and WhatsApp, has announced that it will cut 13% of its workforce. The first mass lay-offs in the firm's history will result in 11,000 employees, from a worldwide headcount of 87,000, losing their jobs. Meta chief executive Mark Zuckerberg said the cuts were "the most difficult changes we've made in Meta's history". The news follows... 
Source 2) Allianz beats quarterly profit expectations, posts rosier 2022 outlook || German insurer Allianz on Wednesday posted a better-than-expected 17% rise in third-quarter net profit, helped by strength at its property and casualty division, and gave a more optimistic full-year outlook.... 
Source 3) Georgia race goes to run-off as fight for US Senate neck-and-neck || Results are being declared in the US midterm elections, with control of Congress hanging in the balance. Republicans are likely to take control of the House of Representatives but the Senate fight is on a knife-edge. The race for the Senate seat in Georgia, which could determine the outcome, will not be decided until a runoff election on 6 December.... 
Source 4) Tesla stock hits 2-year low after Musk sells $4 bln worth of shares || Tesla Inc (TSLA.O) shares slid to their lowest level in nearly two years on Wednesday after Chief Executive Elon Musk sold $3.95 billion worth of shares in the electric-vehicle maker. The shares were down 6.1% at $179.66 in afternoon trading. Musk's latest share sale fueled jitters about the fallout of his Twitter buy on the world's most valuable automaker, analysts... 


Question: What is happening in business today?

Answer:
There are a few major stories happening in business today. Firstly, Marks and Spencer has warned of a "gathering storm" of higher costs for retailers and pressure on household budgets. Secondly, Facebook-owner Meta is cutting 11,000 staff in its first mass lay-offs. And finally, Tesla stock has hit a 2-year low after CEO Elon Musk sold $4 billion worth of shares.
```

Sucess! You'll notice that using Marqo to add relevant and temporally correct context means we can build a news summarisation application with ease. So instead of wrong and vague answers, we get factually-grounded summaries based on retrieved facts such as:
<ol>
  <li>Marks and Spencer has warned of a "gathering storm" of higher costs for retailers</li>
  <li>Facebook-owner Meta is cutting 11,000 staff</li>
  <li>Tesla stock has hit a 2-year low after CEO Elon Musk sold $4 billion worth of shares</li>
</ol>

Full code: <a href="https://github.com/iain-mackie/marqo-gpt3/blob/main/main.py">here</a> (you'll need GPT3 API token)