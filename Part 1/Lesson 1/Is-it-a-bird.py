from duckduckgo_search import DDGS
from fastcore.all import *
from fastdownload import download_url
from fastai.vision.all import *
import os

# Updated DDG image search method
def search_images(term, max_images=200): 
    urls = []
    with DDGS() as ddgs:
        for r in ddgs.images(term, max_results=max_images): # Somehow returns 7 images total when max_images = 1 ..? 
            urls.append(r['image']) # Append key='image' value from each dict in the list
    return urls # Return the list of image urls


urls = search_images('bird photos', max_images=1)

# Place in same dir as this script
dest = os.path.dirname(__file__) + '/'
image_dest = dest + 'bird.jpg'
thumb_dest = dest + 'bird_thumb.jpg'

# Download the images from the urls using the fastdownload library
download_url(urls[0], image_dest, show_progress=False)

# Convert to thumb
im = Image.open(image_dest)
im.to_thumb(256,256).save(thumb_dest)