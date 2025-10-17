import glob
import pandas as pd
from langchain.prompts import PromptTemplate
import re
from text import split_text
from typing import List, Literal
import numpy as np
import torch

from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    GPT2TokenizerFast
)

def reformat_npcs(npcs):
    """ 
    helper function to make the data tidy
    """
    new_npcs = []
    for npc in npcs:

        # include a name field in each for filtering
        name = npc['name']

        for key,value in npc.items():
            doc = {"name":name, "text":f"my {key} is {value}"}
            new_npcs.append(doc)
    return new_npcs

def marqo_template():
    """
    holds the prompt template
    """
    template = """The following is a conversation with a fictional superhero in a movie. 
    BACKGROUND is provided which describes some the history and powers of the superhero. 
    The conversation should always be consistent with this BACKGROUND. 
    Continue the conversation as the superhero in the movie and **always** use something from the BACKGROUND. 
    You are very funny and talkative and **always** talk about your superhero skills in relation to your BACKGROUND.
    BACKGROUND:
    =========
    {summaries}
    =========
    Conversation:
    {conversation}
    """
    return template

def marqo_prompt(template = marqo_template()):
    """ 
    thin wrapper for prompt creation
    """
    PROMPT = PromptTemplate(template=template, input_variables=["summaries", "conversation"])
    return PROMPT

def read_md_file(filename):
    """ 
    generic md/txt file reader
    """
    with open(filename, 'r') as f:
        return f.read()

def clean_md_text(text):
    # Remove code blocks
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    
    # Remove inline code
    text = re.sub(r'`.*?`', '', text)
    
    # Remove headings
    text = re.sub(r'#+.*?\n', '', text)
    
    # Remove horizontal lines
    text = re.sub(r'---*', '', text)
    
    # Remove links
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)
    
    # Remove emphasis
    text = re.sub(r'\*\*.*?\*\*', '', text)
    text = re.sub(r'\*.*?\*', '', text)
    
    # Remove images
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    
    return text


def load_all_files(files):
    """ 
    wrapper to load and clean text files
    """

    results = []
    for f in files:
        text = read_md_file(f)
        splitted_text = split_text(text, split_length=10, split_overlap=3)
        cleaned_text = [clean_md_text(_text) for _text in splitted_text]   
        _files = [f]*(len(cleaned_text))

        results += list(zip(_files, splitted_text, cleaned_text))

    return pd.DataFrame(results, columns=['filename', 'text', 'cleaned_text'])

def load_data():
    """  
    wrapper to load all the data files
    """
    marqo_docs_directory = 'data/'
    files = glob.glob(marqo_docs_directory + 'p*.txt', recursive=True)
    files = [f for f in files if not f.startswith('_')]
    return load_all_files(files)

def qna_prompt():
    """ 
    prompt template for q and a type answering
    """

    template = """Given the following extracted parts of a long document ("SOURCES") and a question ("QUESTION"), create a final answer one paragraph long. 
    Don't try to make up an answer and use the text in the SOURCES only for the answer. If you don't know the answer, just say that you don't know. 
    QUESTION: {question}
    =========
    SOURCES:
    {summaries}
    =========
    ANSWER:"""
    PROMPT = PromptTemplate(template=template, input_variables=["summaries", "question"])
    return PROMPT

model_cache = dict()
def load_ce(model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
    """ 
    loads the sbert cross-encoder model
    """

    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return model, tokenizer

def predict_ce(query, texts, model=None, key='ce'):
    """ 
    score all the queries with respect to the texts
    """

    if model is None:
        if key not in model_cache:
            model, tokenizer = load_ce()
            model_cache[key] = (model, tokenizer) 
        else:
            (model, tokenizer) = model_cache[key]

    # create pairs
    softmax = torch.nn.Softmax(dim=0)
    N = len(texts)
    queries = [query]*N
    pairs = list(zip(queries, texts))
    features = tokenizer(pairs,  padding=True, truncation=True, return_tensors="pt")
    
    model.eval()
    with torch.no_grad():
        scores = model(**features).logits

    return softmax(scores)

def get_sorted_inds(scores):
    """ 
    return indexes based on sorted scores
    """
    return scores.argsort(0, descending=True)

def extract_text_from_highlights(res, token_limit=256, truncate=True):

    highlights = []
    texts = []
    for ind,hit in enumerate(res[ResultsFields.hits]):
        highlight_list = hit[ResultsFields.highlights]
        highlight_key = list(highlight_list[0].keys())[0]
        highlight_text = list(highlight_list[0].values())[0]
        text = hit[highlight_key]
    
        if truncate:
            text = " ".join(text.split())
            highlight_text = " ".join(highlight_text.split())
            text = truncate_text(text, token_limit, highlight_text)
            
        texts.append(text)
        highlights.append(highlight_text)

    return highlights, texts


class ResultsFields:
    hits = 'hits'
    highlights = '_highlights'

cached_tokenizers = dict()

def _lies_between(offset_tuple, offset):
    """ 
    given a tuple of ints, determine if offset lies between them
    """
    return offset >= offset_tuple[0] and offset < offset_tuple[1]


def _find_end_character_mapping(offset_mapping, offset):
    """assumes sorted offset_mapping. unless this was modified 
       this will be the default from the tokenizer
    """
    # if the max length is bigger we just return the last index
    if offset >= max(offset_mapping[-1]):
        return [offset_mapping[-1]]
    return [ind for ind in offset_mapping if _lies_between(ind, offset)]

def find_highlight_index_in_text(text, highlight):
    """ 
    return start and end character indices for the sub-string (highlight)
    """
    if highlight not in text:
        return (None, None)

    # returns left right
    left_ind = text.index(highlight)
    right_ind = left_ind + len(highlight)

    return (left_ind, right_ind)


def get_token_indices(text: str, token_limit: int,  
                method: Literal['start', 'end', 'center','offset'] = 'start',
                tokenizer = None,
                offset: int = None):

    # leave it here instead of a paramter
    default_tokenizer = 'gpt2'

    if tokenizer is None:
        if default_tokenizer not in cached_tokenizers:
            tokenizer = GPT2TokenizerFast.from_pretrained(default_tokenizer)
            cached_tokenizers[default_tokenizer] = tokenizer
        else:
            tokenizer = cached_tokenizers[default_tokenizer]

    tokenized_text = tokenizer(text, return_offsets_mapping=True)
    token_ids = tokenized_text['input_ids']
    character_offsets = tokenized_text['offset_mapping']
    text_token_len = len(token_ids)

    # need to get the offset from the start to hit the full size
    delta = text_token_len - token_limit

    # nothing to do if it fits already
    if delta <= 0:
        return [character_offsets[0], character_offsets[-1]]

    # convert offset into token space
    character_offset_tuple = _find_end_character_mapping(character_offsets, offset)
    token_offset = character_offsets.index(character_offset_tuple[0])

    is_odd_offset = 1
    if token_limit % 2 == 1: is_odd_offset = 0

    if method == 'start':
        ind_start = character_offsets[0]
        ind_end = character_offsets[token_limit-1]

    elif method == 'end':
        ind_start = character_offsets[delta]
        ind_end = character_offsets[-1]

    elif method == 'center':       
        center_token = text_token_len//2
        left_ind = max(center_token - token_limit//2, 0)
        right_ind = min(center_token + token_limit//2, text_token_len)
        ind_start = character_offsets[left_ind]
        ind_end = character_offsets[right_ind-is_odd_offset]
    
    elif method == 'offset':
        center_token = token_offset 
        left_ind = max(center_token - token_limit//2, 0)
        right_ind = min(center_token + token_limit//2, text_token_len)
        ind_start = character_offsets[left_ind]
        ind_end = character_offsets[right_ind-is_odd_offset]
    
    else: 
        raise RuntimeError("incorrect method specified")

    return ind_start, ind_end

def truncate_text(text, token_limit, highlight=None):
    """ 
    truncates text to a token limit centered on the highlight text
    """

    if highlight is None:
        method = 'start'
        center_ind = 0 # this will not be used for this start method
    else:
        # TODO the full context may ot get used if the highlight is not centered    
        # we would need to add the excess to the end/start

        method = 'offset'
        # get indices of highlight
        inds = find_highlight_index_in_text(text, highlight)
        # get the center of the highlight in chars
        center_ind = (max(inds) - min(inds))//2 + min(inds)
        # now map this to tokens and get the left/right char indices to achieve token limit

    ind_left, ind_right = get_token_indices(text, token_limit, method=method, offset=center_ind)
    trunc_text = text[min(ind_left):max(ind_right)]

    return trunc_text

def check_highlights_field(hit, highlight=ResultsFields.highlights):
    """
    check the validity of the highlights in the hit
    """

    if highlight in hit:
        if len(hit[highlight]) == 0:
            return False
        elif isinstance(hit[highlight], dict):
            if hit[highlight].values() == 0:
                return False
            else:
                return True
        else:
            raise RuntimeError("invalid hits and highlights")
    else:
        return False


def test_truncate():
    import random
    import string
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    text_en = ['hello this is a test sentence. this is another one? i think so. maybe throw in some more letteryz....xo']
    text_rm = [''.join(random.choices(string.ascii_uppercase + string.digits, k=10000)) for _ in range(10)]
    ks = [1, 32, 128, 1024, 2048]
    texts = text_en + text_rm
    methods = ['offset', 'start', 'end', 'center']
    for text in texts:
        k_gt = len(tokenizer(text)['input_ids'])
        for k in ks:
            for method in methods:
                ind_left, ind_right = get_token_indices(text, k, method=method, offset=2)
                trunc_text = text[min(ind_left):max(ind_right)]
                k_fn = len(tokenizer(trunc_text)['input_ids'])
                
                assert k_fn <= min(k,k_gt)


def test_find_highlight_in_text():

    n_highlights = 5
    
    texts = [
        'hello how are you', 
        "I assume you only want to find the first occurrence of word in phrase. If that's the case, just use str.index to get the position of the first character. Then, add len(word) - 1 to it to get the position of the last character.",
        ]

    for text in texts:
        for _ in range(n_highlights):
            highlight_ind = sorted(np.random.choice(list(range(len(text))),2, replace=False))
            highlight = text[highlight_ind[0]:highlight_ind[1]]
            inds = find_highlight_index_in_text(text, highlight)
            assert text[inds[0]:inds[1]] == highlight


def test_check_highlights_field():
    results_lexical = {
    'hits': [
        {   
            'Title': 'Extravehicular Mobility Unit (EMU)',
            'Description': 'The EMU is a spacesuit that provides environmental protection, mobility, life support, and' 
                           'communications for astronauts',
            '_highlights': [],
            '_id': 'article_591',
            '_score': 0.61938936
        }, 
        {   
            'Title': 'The Travels of Marco Polo',
            'Description': "A 13th-century travelogue describing Polo's travels",
            '_highlights': [],
            '_id': 'e00d1a8d-894c-41a1-8e3b-d8b2a8fce12a',
            '_score': 0.60237324
        }
    ],
    'limit': 10,
    'processingTimeMs': 49,
    'query': 'What is the best outfit to wear on the moon?'
    }

    assert all(not check_highlights_field(hit, highlight=ResultsFields.highlights) for hit in results_lexical[ResultsFields.hits])

    results_tensor = {
        'hits': [
            {   
                'Title': 'Extravehicular Mobility Unit (EMU)',
                'Description': 'The EMU is a spacesuit that provides environmental protection, mobility, life support, and' 
                            'communications for astronauts',
                '_highlights': [{
                    'Description': 'The EMU is a spacesuit that provides environmental protection, '
                                'mobility, life support, and communications for astronauts'
                }],
                '_id': 'article_591',
                '_score': 0.61938936
            }, 
            {   
                'Title': 'The Travels of Marco Polo',
                'Description': "A 13th-century travelogue describing Polo's travels",
                '_highlights': [{'Title': 'The Travels of Marco Polo'}],
                '_id': 'e00d1a8d-894c-41a1-8e3b-d8b2a8fce12a',
                '_score': 0.60237324
            }
        ],
        'limit': 10,
        'processingTimeMs': 49,
        'query': 'What is the best outfit to wear on the moon?'
    }

    assert all(check_highlights_field(hit, highlight=ResultsFields.highlights) for hit in results_tensor[ResultsFields.hits])


def get_extra_data():
    """
    the extra data for the NPC's
    """
    text1 = """

    IMPORTANT SAFETY INSTRUCTIONS
    Read this user manual carefully before first use and
    save it for future reference.
    This appliance is not intended for use by persons
    (including children) with reduced physical, sensory
    or mental capabilities or lack of experience and
    knowledge, unless they have been given supervision
    or instructions concerning use of the appliance by a
    person responsible for their safety.
    Children should be supervised to ensure that they do
    not play with the appliance.
    Cleaning and user maintenance shall not be made by
    children without supervision.
    This product has been designed for domestic use
    only. In case of any commercial use, inappropriate
    use or failure to comply with the instructions, the
    will not apply.
    manufacturer is not responsible, and the guarantee
    Before connecting your appliance, check if the mains
    voltage is the same as the voltage indicated on your
    appliance and that the power outlet has an earth
    connection. If unsure check with an electrician.
    Ensure before each use that the supply cord or any
    other parts are not damaged.
    Keep the iron and its cord out of reach of children
    when connected to mains power or cooling down.
    Never direct the steam towards persons or animals.
    Never direct the steam jet towards any other
    electrical or/and electronic appliances.
    DO NOT use the iron if it has been dropped, if there
    are visible signs of damage, malfunction or if it is
    leaking water.
    """

    text2 = """

    unplugged and cooled any
    Always down before doing
    make sure the appliance is switched off,
    maintenance work.
    IMPORTANT! This iron must not be left unattended
    while it is connected to mains power and before it
    has cooled down. The soleplate of the iron can
    become extremely hot and may cause burns if
    touched.
    Do not unplug the appliance by pulling on the cord
    or on the appliance.
    Never immerse the iron, the stand, the power cord
    or the plug in water. Never hold them under a water
    tap.
    Do not allow the supply cord to come in contact with
    the soleplate when the iron is hot.
    If the supply cord is damaged, it must be replaced by
    qualified electrical person only or the product must
    be disposed.
    Disconnect from mains power when filling the
    reservoir in the iron with water.
    The iron must be used and rested on a flat, stable
    surface.
    Always place and operate the iron on a flat, solid,
    clean and dry surface. When placing the iron on its
    stand, ensure that the surface on which the stand is
    placed is stable.
    For additional protection, this appliance should be
    connected to a household residual current device
    (safety switch) with a rating of no more than 30mA.
    If unsure consult an electrician for advice.

    """

    text3 = """
    BEFORE FIRST USE
    Unpack the appliance and check if all parts are there
    and undamaged. Should this not be the case, return the
    product to Kmart for replacement.
    The packaging contains:
    One steam iron
    One filler cup
    Clean all parts before first use. See Cleaning and
    Maintenance
    Heat up the iron to its maximum temperature and iron
    over a piece of damp cloth for several minutes. This will
    burn-off any manufacturing residue from the iron.
    Smoke and odour may occur for a short period during
    this process, which is normal.
    FEATURES
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

    text4 = """

    PARTS
    D
    C
    B
    A
    A
    Soleplate
    E
    Spray button
    B
    Spray nozzle
    F
    Steam shot button
    C
    Water filling inlet
    G
    Cord bushing
    D
    Variable steam
    H
    Indicator lamp
    button
    I
    Temperature dial

    OPERATION
    Filling the water tank
    Ensure that the iron is cold and not connected to a
    power supply.
    Open the water filling cover (C). Hold the iron in a
    tilted position and fill the tank with plain water,
    using the filler cup.
    DO NOT exceed the max level on the tank.
    Close the water filling cover (C) by pressing down the
    cap until you hear a 'click'. To avoid water spillage,
    ensure that the water tank is properly closed.
    NOTES:
    To avoid calcium deposits, we recommend you use
    distilled water for your steam iron.
    To avoid damage or blockage of your steam iron, do not
    use any scented water or additives.
    Steam Ironing
    Your steam iron is equipped with the Eco Steam System.
    This function offers 3 steam levels for optimal ironing:
    ECO: Reduced amount of steam for energy saving,
    suitable for most fabric types
    Normal steam :suitable for most fabric types
    Power steam :increased steam output for remove
    stubborn creases
    Fill the water tank with plain water.
    Position the iron vertically on its stand on a suitable
    flat stable surface.

    """

    text5 = """

    CLEANING AND MAINTENANCE
    Unplug the iron from mains power and allow it to cool
    completely before cleaning and storing.
    down
    Cleaning
    Clean the upper part of iron with a damp cloth. If
    necessary, use a mild detergent.
    Do not use alcohol, acetone, benzene, scourging
    cleaning agents, etc., to clean the iron. Do not use
    hard brushes or metallic objects.
    Remove any deposits on the soleplate with a damp
    cloth. To avoid scratching the finishing, never use
    metallic pad to clean the soleplate, and never place
    the iron on a rough surface.
    To remove synthetic residue from the soleplate, iron
    over an old cotton rag in high setting.
    Always store the emptied iron horizontally on a
    stable surface with a cloth protecting the soleplate.
    Self-clean function
    The self-cleaning function should be used on a regular
    basis, depending on usage and the hardness of water
    used.
    WARNING: Never pour white vinegar or any liquid
    cleaners into the water tank!
    Fill the water tank. Refer to OPERATION: Filling the
    water tank.
    Position the iron vertically on its stand on a suitable
    surface.
    Ensure that the steam control (D) is in OFF position.
    Connect the plug to a suitable mains power outlet.
    Set to maximum ironing temperature.
    Once the soleplate temperature is reached, unplug
    the iron from mains power.
    Hold the iron over a sink (away from your body),
    pressand hold the steam control (D) at "CALC CLEAN"
    position. Boiling water and steam will now emit from
    the holes in soleplate.
    Carefully shake the iron forwards and backwards to
    allow any scale and other deposits being expelled
    with the boiling water and the steam.
    Release the CALC CLEAN button when the water tank
    is empty.
    Plug the appliance into a mains power outlet socket,
    set a temperature and leave in operation for at least
    two minutes to dry the soleplate.
    Unplug the iron and let the iron cool down fully.
    """

    text6 = """


    IRONING TIPS
    IMPORTANT! always check whether a label with ironing instructions is attached to the garment. Always
    follow the ironing instructions attached to the garment.
    The iron heats up quicker than it cools down, therefore, you should start ironing the articles requiring the
    lowest temperature such as those made of synthetic fibre.
    If the fabric consists of various kinds of fibres, you must always select the lowest ironing temperature to
    iron the composition of those fibres.
    Silk and other fabrics that are likely to become shiny should always be ironed on the inner side. To prevent
    staining, do not spray water straight on silk or other delicate fabrics.
    To prevent staining, do not spray water straight on silk or other delicate fabrics.
    Velvet and other textures that rapidly become shiny should be ironed in one direction with light pressure
    applied. Always keep the iron moving at any moment.
    Pure wool fabrics (100% wool) may be ironed with the iron set to steam position. Preferably set steam
    button to the maximum position and use a dry cloth between the garment and the iron for protection.
    Do not touch plastic buttons with a hot iron because they may melt.
    Be careful around zippers and similar items to prevent the soleplate from scratching.
    Symbol
    Temperature setting
    Variable steam
    Steam shot
    Spray
    Fabric
    MIN
    x
    x
    Synthetic fiber
    x
    x
    Wool, Silk
    Cotton, Linen
    MAX
    possible
    x
    Attention! This symbol indicates: DO NOT IRON!
    not possible
    TECHNICAL DATA
    Rated voltage: 220-240V~ 50-60Hz
    Rate power: 2000-2400W
    """

    text7 = """

    12 Month Warranty
    Thank you for your purchase from Kmart.
    Kmart Australia Ltd warrants your new product to be free from defects in materials and workmanship for the
    period stated above, from the date of purchase, provided that the product is used in accordance with
    accompanying recommendations or instructions where provided. This warranty is in addition to your rights
    under the Australian Consumer Law.
    Kmart will provide you with your choice of a refund, repair or exchange (where possible) for this product if it
    becomes defective within the warranty period. Kmart will bear the reasonable expense of claiming the
    warranty. This warranty will no longer apply where the defect is a result of alteration, accident, misuse, abuse
    or neglect.
    Please retain your receipt as proof of purchase and contact our Customer Service Centre on 1800 124 125
    (Australia) or 0800 945 995 (New Zealand) or alternatively, via Customer Help at Kmart.com.au for any
    difficulties with your product. Warranty claims and claims for expense incurred in returning this product can
    be addressed to our Customer Service Centre at 690 Springvale Rd, Mulgrave Vic 3170.
    Our goods come with guarantees that cannot be excluded under the Australian Consumer Law. You are
    entitled to a replacement or refund for a major failure and compensation for any other reasonably foreseeable
    loss or damage. You are also entitled to have the goods repaired or replaced if the goods fail to be of
    acceptable quality and the failure does not amount to a major failure.
    For New Zealand customers, this warranty is in addition to statutory rights observed under New Zealand
    legislation.
    """

    return [text1, text2, text3, text4, text5, text6, text7]