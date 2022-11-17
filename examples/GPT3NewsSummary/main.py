
import marqo
import json
import openai

from news import MARQO_DOCUMENTS

# init GPT3 API
openai.organization = None
openai.api_key = None

DOC_INDEX_NAME = 'news-index'
output = './news_summaries.txt'
queries = [
    # question                                              # date filter
    ('How is the US Midterm Election going?',               None),
    ('How is COP27 progressing?',                           None),
    ('What is happening in business today?',                '2022-11-09'),

]


if __name__ == '__main__':

    print('Establishing connection to marqo client.')
    mq = marqo.Client(url='http://localhost:8882')

    #########################################################################
    ######################### MARQO INDEXING ################################
    #########################################################################
    # mq.index(DOC_INDEX_NAME).delete()
    try:
        print(f'document index build: {mq.index(DOC_INDEX_NAME).get_stats()}')
    except KeyboardInterrupt:
        raise
    except:
        print('Indexing documents')
        mq.index(DOC_INDEX_NAME).add_documents(MARQO_DOCUMENTS)
        print('Done')


    #########################################################################
    ######################### GPT3 GENERATION ###############################
    #########################################################################

    def get_no_context_prompt(question):
        """ GPT3 prompt without any context. """
        return f'Question: {question}\n\nAnswer:'

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


    #########################################################################
    ########################### EXPERIMENTS ################################
    #########################################################################

    # Write to news_summaries.txt for analysis.
    with open(output, 'w') as f_out:
        for question, date in queries:
            f_out.write('////////////////////////////////////////////////////////\n')
            f_out.write('////////////////////////////////////////////////////////\n')

            f_out.write(f'question: {question}, date filter: {date}\n')

            f_out.write('================= GPT3 NO CONTEXT ======================\n')
            # Build prompt without context.
            prompt = get_no_context_prompt(question)
            f_out.write(f'Prompt: \n{prompt}\n')
            summary = prompt_to_essay(prompt)
            f_out.write(f'{summary}\n\n')

            f_out.write('================= GPT3 + Marqo  =======================\n')
            # Query Marqo and set filters based on user query
            if isinstance(date, str):
                results = mq.index(DOC_INDEX_NAME).search(q=question,
                                                          searchable_attributes=['Title', 'Description'],
                                                          filter_string=f"date:{date}",
                                                          limit=5)
            else:
                results = mq.index(DOC_INDEX_NAME).search(q=question,
                                                          searchable_attributes=['Title', 'Description'],
                                                          limit=5)

            # Build context using Marqo's highlighting functionality.
            context = ''
            for i, hit in enumerate(results['hits']):
                title =  hit['Title']
                text = hit['Description']
                # for section, text in hit['_highlights'].items():
                #     context += text + '\n'
                context += f'Source {i}) {title} || {" ".join(text.split()[:60])}... \n'
            # Build prompt with Marqo context.
            prompt = get_context_prompt(question=question, context=context)
            f_out.write(f'Prompt: \n{prompt}\n')
            summary = prompt_to_essay(prompt)
            f_out.write(f'{summary}\n\n')

