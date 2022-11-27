from tempfile import NamedTemporaryFile
import urllib3
import os
from pathlib import Path
import subprocess

from PIL import Image
import streamlit as st
import pandas as pd

import marqo

####################################################
# some conveniance functions
#  
# Streamlit configuration settings
st.set_page_config(
    page_title="Marqo Demo App",
    page_icon="favicon.png",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={}
)

urllib3.disable_warnings()

def format_to_field_from_attributes(attributes):
    return [f'is_{a}' for a in attributes]

@st.experimental_singleton
def load_marqo_logo():
    cwd = os.getcwd() 
    return Image.open(f"{cwd}/marqo-logo.jpg")

PORT = 8221

@st.experimental_singleton
def start_image_server(image_dir, port=PORT):
    pid = subprocess.Popen(['python3', '-m', 'http.server', f'{port}', '--directory', image_dir], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


# setup the marqo client
mq = marqo.Client("http://localhost:8882")
docker_path = f'http://host.docker.internal:{PORT}/'
temp_filename = 'temp.jpg'
image_key = "Image Location"

def main():

    image_dir = os.getcwd() + '/temp/'
    Path(image_dir).mkdir(parents=True, exist_ok=True)
    start_image_server(image_dir)

    logo = load_marqo_logo()
    st.image(logo)

    device = 'cpu'

    uploaded_file = st.file_uploader("", type=['jpg','png','jpeg'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns( [0.5, 0.5])

        with col1:
            st.markdown('<p style="text-align: center;">Selected image</p>',unsafe_allow_html=True)
            st.image(image, width=512)  

        # col11, col22 = st.columns(2)

        # form1 = col11.form(key='my-form1')
        # submit1 = form1.form_submit_button('Get attributes...')
        
        temp_file = NamedTemporaryFile(delete=False)
        if uploaded_file:
            temp_file.write(uploaded_file.getvalue())
        
        save_name = f'{image_dir}{temp_filename}'
        image = Image.open(temp_file.name)
        image.save(save_name)

        text_in = st.empty()
        sentence = text_in.text_input("Enter custom attributes (e.g. house, bedroom ) or a question (what color is the wall?)")

        if sentence:

            is_qa = False
            if ',' in sentence or not sentence.endswith('?'):
                columns = sentence.split(',')
                enrichment = {
                            "task": "attribute-extraction",
                            "to": format_to_field_from_attributes(columns),
                            "kwargs": {
                                "attributes": [{"string": sentence}],
                                "image_field": {"document_field": image_key}
                            },
                        }

            else:
                columns = [sentence]
                is_qa = True
                enrichment = {
                        "task": "question-answer",
                        "to": ["question"], 
                        "kwargs": {
                            "query": [
                                {"string":sentence}],
                            "image_field": {"document_field":image_key}
                        },
                    }

            documents = [{image_key:docker_path + temp_filename}]

            print(documents)
            print(enrichment)
            # now we can call the enrich endpoint on our documents
            enriched = mq.enrich(
                documents=documents,
            enrichment=enrichment,
            device=device)

            df = pd.DataFrame(enriched['documents'])
            st.dataframe(data=df[[c for c in df.columns if c != image_key]])

            
main()