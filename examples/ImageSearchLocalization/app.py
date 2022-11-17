from PIL import Image, ImageOps
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
        st.image(images, caption=captions, use_column_width=False)#, width=230)


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
    import math
    from PIL import Image, ImageDraw
        
    # create rectangle image
    img1 = ImageDraw.Draw(img)  
    img1.rectangle(box, outline  = 'orange', width = 8)
    return img

def main():

    data_dir = "/home/jesse/code/stable-diffusion/scripts/coco/"
    image_location = 'image_docker'
    highlights = '_highlights'
    device = 'cuda'
    
    docker_prefix = 'http://host.docker.internal:8222/'
    locale_location = '/home/jesse/code/stable-diffusion/scripts/coco/'

    st.title("image search with localisation")

    sentence = st.text_input("Please enter some text to start searching...")

    s1,s2,s3,s4,s5 = st.columns(5)

    options = ['None',
                   'yolox',
                   'dino/v2',
                   'dino/v1',
                   'frcnn'
                   ]
    option = s4.radio('Select three known variables:',
                  options)

    print(option)

    if option == options[0]:

        index_name = 'visual-search'

    elif option == options[1]:

        index_name = 'visual-search-yolox'
    
    elif option == options[2]:

        index_name = 'visual-search-dino-v2'

    elif option == options[3]:

        index_name = 'visual-search-dino-v1'

    elif option == options[4]:

        index_name = 'visual-search-frcnn'


    option_reranker = s5.radio('Select three known variables:',
                  ['None',
                   "google/owlvit-base-patch32",
                   "google/owlvit-base-patch16",
                   "google/owlvit-large-patch14"
                   ])

    if option_reranker == 'None':
        reranker = None
    else:
        reranker = option_reranker
    print(option_reranker)
    form1 = s1.form(key='my-form1')
    submit1 = form1.form_submit_button('Search with tensor...')

    form2 = s2.form(key='my-form2')
    submit2 = form2.form_submit_button('Search with tensor localisation...')

    image_file = s3.file_uploader("Upload Images", type=["png","jpg","jpeg"])

    # search with images and text?
    if submit1:
        st.text("searching using '{}'...".format(sentence))
        
        res = client.index(index_name).search("{}".format(sentence), searchable_attributes=[image_location], reranker=reranker, device=device)
        # print(res)
        print(res, len(res['hits'][0]))
        # get the image 
        if len(res['hits']) == 0:
            st.text(f"No results found for {sentence}")
        else:
            # get the image 
            images = [i[image_location].replace(docker_prefix, locale_location) for i in res['hits']]
            #boxes = [i[highlights][image_location] for i in res['hits']]
            # render text
            render_images(images) #, boxes=boxes)
 
    if submit2:
       
        st.text("searching using '{}'...".format(sentence))
        
        res = client.index(index_name).search("{}".format(sentence), searchable_attributes=[image_location], reranker=reranker, device=device)
        # print(res)
        print(res, len(res['hits'][0]))
        # get the image 
        if len(res['hits']) == 0:
            st.text(f"No results found for {sentence}")
        else:
            # get the image 
            images = [i[image_location].replace(docker_prefix, locale_location) for i in res['hits']]
 
            boxes = [i['_highlights'][image_location] for i in res['hits']]

            # render text
            render_images(images, boxes=boxes)

    
    if image_file is not None:

        temp_file = NamedTemporaryFile(delete=False)
        if image_file:
            temp_file.write(image_file.getvalue())
            #st.write(load_image(temp_file.name))
        save_name = f'{data_dir}temp.png'
        image = load_image(temp_file.name)
        image.save(save_name)
        
        query = save_name#.replace(data_dir, "http://localhost:8223/")
        # To View Uploaded Image
        st.image(image, width=250)

        res = client.index(index_name).search("{}".format(query), searchable_attributes=[image_location], 
                                         limit=9, device=device)
        print(res, len(res['hits']))
        # get the image 
        if len(res['hits']) == 0:
            st.text(f"No results found for {sentence}")
        else:
            images = [i[image_location].replace(docker_prefix, locale_location) for i in res['hits']]

            boxes = [i['_highlights'][image_location] for i in res['hits']]

            # render text
            render_images(images, boxes=boxes)

    
           
  
main()
