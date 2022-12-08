from PIL import Image, ImageOps, ImageDraw
import streamlit as st
from tempfile import NamedTemporaryFile
import validators
import requests
import os
import urllib3
urllib3.disable_warnings()

from marqo import Client

client = Client()

def load_image_from_path(image):
    """loads an image into PIL from a string path that is
    either local or a url
    Args:
        image (str): _description_
    Raises:
        ValueError: _description_
    Returns:
        ImageType: _description_
    """
    
    if os.path.isfile(image):
        img = Image.open(image)
    elif validators.url(image):
        img = Image.open(requests.get(image, stream=True).raw)
    else:
        raise ValueError(f"input str of {image} is not a local file or a valid url")

    return img    

def render_images(images, captions=None, boxes=None):
    """renders a list of image pointers

    Args:
        images (_type_): _description_
        captions (_type_, optional): _description_. Defaults to None.
    """
    
    if boxes is None:
        images = [ImageOps.expand(load_image_from_path(image), border=100, fill=(255,255,255) ) for image in images]
    else:
        images = [ImageOps.expand(draw_box(load_image_from_path(image), box), border=100, fill=(255,255,255) ) for box,image in zip(boxes,images)]
        

    if captions is None:
        st.image(images, use_column_width=False, width=230)
    else:
        st.image(images, caption=captions, use_column_width=False)


def load_image(image_file):
    """loads an image using PIL

    Args:
        image_file (_type_): _description_

    Returns:
        _type_: _description_
    """
    img = Image.open(image_file)
    return img

def draw_box(img, box):
    """draws a rectangle box on a PIL image

    Args:
        img (_type_): _description_
        box (_type_): _description_

    Returns:
        _type_: _description_
    """
    img1 = ImageDraw.Draw(img)  
    img1.rectangle(box, outline  = 'orange', width = 8)
    return img

def main():

    #########################################
    # SETUP SOME PARAMTERS
    # this is for temporary storage of an image
    temp_image_dir = os.getcwd() + '/'
    # the field name for the image
    image_location = 'image_location'
    # some enums
    highlights = '_highlights'
    hits = 'hits'

    # specify the device
    device = 'cuda'
    
    docker_image_server_prefix = 'http://host.docker.internal:8222/'
    local_image_location = os.getcwd() + '/images/'

    #########################################

    st.title("image search with localisation")

    sentence = st.text_input("Please enter some text to start searching...")

    s1,s2,s3,s4,s5 = st.columns(5)

    options = ['None', 'yolox', 'dino/v2']

    option = s4.radio('Select the indexing method', options)

    if option == options[0]:
        index_name = 'visual-search'
    elif option == options[1]:
        index_name = 'visual-search-yolox'
    elif option == options[2]:
        index_name = 'visual-search-dino-v2'
    else:
        raise ValueError(f"unexpected option for {option}")

    option_reranker = s5.radio('Select the reranker',
                  ['None', "google/owlvit-base-patch32"])

    if option_reranker == 'None':
        reranker = None
    else:
        reranker = option_reranker

    form1 = s1.form(key='my-form1')
    submit1 = form1.form_submit_button('Search with tensor...')

    form2 = s2.form(key='my-form2')
    submit2 = form2.form_submit_button('Search with tensor localisation...')

    image_file = s3.file_uploader("Upload Images", type=["png", "jpg", "jpeg"])

    # search with images and text?
    if submit1:
        st.text("searching using '{}'...".format(sentence))
        res = client.index(index_name).search("{}".format(sentence), searchable_attributes=[image_location], reranker=reranker, device=device)

        # get the image 
        if len(res[hits]) == 0:
            st.text(f"No results found for {sentence}")
        else:
            # get the image 
            images = [i[image_location].replace(docker_image_server_prefix, local_image_location) for i in res['hits']]
            # render text
            render_images(images) #, boxes=boxes)
 
    if submit2:       
        st.text("searching using '{}'...".format(sentence))        
        res = client.index(index_name).search("{}".format(sentence), searchable_attributes=[image_location], reranker=reranker, device=device)
        # get the image 
        if len(res[hits]) == 0:
            st.text(f"No results found for {sentence}")
        else:
            # get the image 
            images = [i[image_location].replace(docker_image_server_prefix, local_image_location) for i in res['hits']]
            boxes = [i[highlights][image_location] for i in res[hits]]

            # render text
            render_images(images, boxes=boxes)

    
    if image_file is not None:

        temp_file = NamedTemporaryFile(delete=False)
        if image_file:
            temp_file.write(image_file.getvalue())
           
        save_name = f'{temp_image_dir}temp.png'
        image = load_image(temp_file.name)
        image.save(save_name)
        
        query = save_name#.replace(data_dir, "http://localhost:8223/")
        # To View Uploaded Image
        st.image(image, width=250)

        res = client.index(index_name).search("{}".format(query), searchable_attributes=[image_location], 
                                         limit=9, device=device)

        # get the image 
        if len(res['hits']) == 0:
            st.text(f"No results found for {sentence}")
        else:
            images = [i[image_location].replace(docker_image_server_prefix, local_image_location) for i in res['hits']]

            boxes = [i[highlights][image_location] for i in res[hits]]

            # render text
            render_images(images, boxes=boxes)

    
           
  
main()
