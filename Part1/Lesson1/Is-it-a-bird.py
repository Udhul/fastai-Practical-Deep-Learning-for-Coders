from duckduckgo_search import DDGS
from fastcore.all import *
from fastdownload import download_url
from fastai.vision.all import *
import os

# Updated DDG image search method
def search_images(term, max_images=60, term_suffix:list=None):
    '''term can be "bird"
    max_images gives approx that amount total. Each variation will have max_images/len(combined_terms) amount of images. 
    Therefore, max_images is subject to an int rounding error
    term_suffix can be ["photo","light photo","shade photo"]
    ''' 
    variation_urls = []
    urls = []

    # Make seach term combinations. I.e. ["bird photo", "bird light photo", "bird shade photo"]
    if term_suffix:
        combined_terms = [term+' '+variation for variation in term_suffix]
    else:
        combined_terms = [term]

    # Amount of images for each combination
    variation_image_count = int(max_images/len(combined_terms))

    # For each of the combined terms, get the urls
    for term_variant in combined_terms:
        # Using new DDG search method
        with DDGS() as ddgs:
            for r in ddgs.images(term_variant, max_results=variation_image_count):
                 # Append key='image' value from each dict in the list
                variation_urls.append(r['image'])
        # Subindex to make sure no more than the defined amount is in the list
        variation_urls = variation_urls[:variation_image_count]
        # Add the variation urls to the parent list of urls for the search term
        urls += variation_urls

    # Remove any duplicate urls from the list. Could also use list(set()) but note it will mess up the order
    unique_urls = []
    for item in urls:
        if item not in unique_urls:
            unique_urls.append(item)

    # Return the list of image urls
    return urls


def download_dataset(search_terms:list, images_per_term:int, term_suffix:list=None):
    '''Download the "search_terms" list of search terms.
    Get "images_per_term" for each term. 
    Put in folders named by each search term. 
    Make thumbnails named with _thumb for each image.'''
    # Prepare a dict with search terms
    urls = {}

    # Create each of the search term key in the url dict and add the urls as a list in the corresponding values
    for term in search_terms:
        urls[term] = search_images(term, max_images=images_per_term, term_suffix=term_suffix)

    # Place under same dir as this script and then in dataset
    base_dir = os.path.dirname(__file__) + '/dataset/'

    # Download the images for each search term. Then make a copy converted to thumb
    for term in urls:
        count = 0
        for url in urls[term]:
            # Name directories and img file
            image_dest = f'{base_dir}/{term}/{count}.jpg'
            thumb_dest = f'{base_dir}/{term}/{count}_thumb.jpg'

            # If the file is already there, just tick the counter and continue to next loop iteration
            if os.path.isfile(thumb_dest): 
                count+=1
                continue

            try:
                # Download the images from the urls using the fastdownload library and place it in the defined destination
                print(f'Getting "{term}" image {count}')
                download_url(url, image_dest, show_progress=False)

                # Convert to thumb
                im = Image.open(image_dest)
                im.to_thumb(256,256).save(thumb_dest)
            # Handle errors so the program doesn't crash if some images fail to download
            except:
                print(f'The "{term}" url failed: {url}')

            # Tick the img conter
            count+=1

# Main function
def main():
    # Search terms
    search_terms = ['bird', 'forest']

    term_suffix = ['photo', 'bright photo', 'dark and shaded photo']

    # Images per search term
    images_per_term = 200

    # Run the function to download dataset. If already downloaded, this can be disabled
    download_dataset(search_terms, images_per_term, term_suffix)

# Run the main function when this script is executed directly
if __name__ == "__main__":
    main()