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


def download_dataset(search_terms:list, images_per_term:int):
    '''Download the "search_terms" list of search terms.
    Get "images_per_term" for each term. 
    Put in folders named by each search term. 
    Make thumbnails named with _thumb for each image.'''
    # Prepare a dict with search terms
    urls = {}

    # Create each of the search term key in the url dict and add the urls as a list in the corresponding values
    for term in search_terms:
        urls[term] = search_images(term, max_images=images_per_term)[:images_per_term] # Subindex to make sure no more than the defined amount is in the list

    # Place under same dir as this script
    base_dir = os.path.dirname(__file__) + '/'

    # Download the images for each search term. Then make a copy converted to thumb
    for term in urls:
        count = 0
        for url in urls[term]:
            # Name directories and img file
            image_dest = f'{base_dir}/{term}/{count}.jpg'
            thumb_dest = f'{base_dir}/{term}/{count}_thumb.jpg'

            # Download the images from the urls using the fastdownload library and place it in the defined destination
            download_url(url, image_dest, timeout=5, show_progress=False)

            # Convert to thumb
            im = Image.open(image_dest)
            im.to_thumb(256,256).save(thumb_dest)

            # Append the img conter
            count+=1

# Main function
def main():
    # Search terms
    search_terms = ['bird photo', 
                    'forest photo']

    # Images per search term
    images_per_term = 200

    # Run the function to download dataset. If already downloaded, this can be disabled
    download_dataset(search_terms, images_per_term)

# Run the main function when this script is executed directly
if __name__ == "__main__":
    main()