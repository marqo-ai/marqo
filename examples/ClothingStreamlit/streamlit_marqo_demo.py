import os
import requests
import streamlit as st
import pandas as pd
from PIL import Image
import pprint
import marqo

# Streamlit configuration settings
st.set_page_config(
    page_title="Marqo Demo App",
    page_icon="favicon.png",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={}
)

mq = marqo.Client(url='http://localhost:8882') # Connection to Marqo Docker Container
cwd = os.getcwd() # Get current working directory

def load_index(number_data):
    try:
        shirt_data = pd.read_csv('clothing-dataset/images.csv').head(number_data)[['image','label','kids']].to_dict('records')

        for data in shirt_data:
            path = "http://host.docker.internal:8222/clothing-dataset/images/" + data['image'] + ".jpg"
            data['image'] = path
            
        settings = {
            "treat_urls_and_pointers_as_images":True,   # allows us to find an image file and index it
            "model":"ViT-B/16"
        }
         
        mq.create_index("demo-search-index", **settings)

        with st.spinner("Creating Index..."):
            mq.index("demo-search-index").add_documents(shirt_data)

        st.success("Index successfully created.")
    except:
        st.error("Index already exists.")

def delete_index():
    try:
        mq.index("demo-search-index").delete()
        st.success("Index successfully deleted.")
    except:
        st.error("Index does not exist.")

def save_uploadedfile(uploadedfile):
     with open(os.path.join(cwd, uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
     return uploadedfile.name

def reset_state():
    st.session_state['results'] = {}
    st.session_state['page'] = -1

def create_filter_str(filter_list):
    filter_string = ""

    if 'Kids' in filter_list:
        filter_string += 'kids:true'
        filter_list.remove('Kids')
    else:
        filter_string += 'kids:false'
    
    for field in filter_list:
        filter_string += f" AND label:({field})"

    print(filter_string)
    return filter_string 
    
def main():
    # Streamlit state variables (this is to save the state of the session for pagination of Marqo query results)
    if 'results' not in st.session_state:
        st.session_state['results'] = {}

    if 'page' not in st.session_state:
        st.session_state['page'] = -1

    # Index Settings Frontend
    with st.sidebar:
        st.write("Index Settings:")
        values = st.slider(
            label='Select a range of values',
            min_value=10.0, 
            max_value=2000.0,
            value=1000.0,
            step=10.0)

        create_col, _, delete_col = st.columns([1,1,1])

        with create_col:
            create_btn = st.button('Create Index')
        if create_btn:
            load_index(int(values))
        with delete_col:
            delete_btn = st.button('Delete Index')
        if delete_btn:
            delete_index()

    # Main application frontend
    logo = Image.open("{}\marqo-logo.jpg".format(cwd))
    st.image(logo)

    search_text, search_image_url, search_image = None, None, None
    search_mode = st.radio("",("Text", "Image"), horizontal=True, on_change=reset_state)
    if search_mode == "Text":
        box_col, search_mode_col = st.columns([6,1])
        with box_col:
            search_text = st.text_input("Text Search")

        with search_mode_col:
            search_text_mode = st.radio("Search mode", ("Tensor", "Lexical"))
    else:
        image_input_col, image_type_col = st.columns([6,1])

        with image_type_col:
            image_type = st.radio("Image type", ("Web", "Local"))

        with image_input_col:
            if image_type=="Web":
                search_image_url = st.text_input("Provide an Image URL")

            else:
                search_image = st.file_uploader('Upload an Image', type=['jpg'])

    with st.expander("Search Settings"):
        attr_col, filter_col = st.columns(2)
        with attr_col:    
            searchable_attr = st.multiselect('Searchable Attributes', ['Image', 'Label'], default=['Label'])

        with filter_col:
            filtering = st.multiselect('Pre-filtering Options', ['Dress', 'Hat', 'Longsleeve', 'Outwear', 'Pants', 'Shirt', 'Shoes', 'Shorts', 'Skirt', 'T-Shirt', 'Kids'], default=None)


    search_btn = st.button('Search')

    # Marqo Results logic
    if ((search_image is not None) or (search_image_url) or (search_text)) and search_btn:
        if search_text != "" and search_text != None:
            results = mq.index("demo-search-index").search(
                search_text,
                filter_string=create_filter_str(filtering), 
                search_method=search_text_mode.upper(), 
                searchable_attributes=[i.lower() for i in searchable_attr],
                limit=30
                )
        
        elif search_image_url != "" and search_image_url != None:
            results = mq.index("demo-search-index").search(
                search_image_url,
                filter_string=create_filter_str(filtering), 
                searchable_attributes=[i.lower() for i in searchable_attr],
                limit=30
                )
        
        else:
            uploaded_img_name = save_uploadedfile(search_image)
            
            uploaded_img_path = f"http://host.docker.internal:8222/{uploaded_img_name}"
            print(uploaded_img_path)

            results = mq.index("demo-search-index").search(
                uploaded_img_path,
                filter_string=create_filter_str(filtering), 
                searchable_attributes=[i.lower() for i in searchable_attr],
                limit=30
                )

        pprint.pprint(results)

        st.session_state['results'] = results

        if st.session_state['results']['hits']:
            st.session_state['page'] = 0
        else:
            st.session_state['page'] = -1



    # Results Pagination Logic
    if st.session_state['page'] > -1:
        prev_col, page_col, next_col = st.columns([1,9,1])
        with prev_col:
            prev_btn = st.button("Prev")
            if prev_btn and (st.session_state['page'] > 0):
                st.session_state['page']-=1

        with next_col:
            next_btn = st.button("Next")
            if next_btn and (st.session_state['page'] < 2):
                st.session_state['page'] += 1

        with page_col:
            st.markdown('<div style="text-align: center"> {}</div>'.format("Page " + str(st.session_state['page']+1)), unsafe_allow_html=True)

    if st.session_state['results'] != {}:
        if st.session_state['results']['hits']:
            st.write("Results (Top 30):")
            col = st.columns(5)
            for img in enumerate(st.session_state['results']['hits']):
                pic = img[1]['image']

                if pic is not None:
                    if (img[0] >= st.session_state['page']*10) and (img[0] < (st.session_state['page']*10 + 10)):
                        with col[(img[0]%10)//2]:
                            if pic.startswith("http://host.docker.internal:8222/"):
                                pic_url = 'http://localhost:8222/' + pic.split("http://host.docker.internal:8222/")[1]
                                image = Image.open(requests.get(pic_url, stream=True).raw)
                                st.image(image, caption=img[1]['label'])
        else:
            st.write("No results")


main()