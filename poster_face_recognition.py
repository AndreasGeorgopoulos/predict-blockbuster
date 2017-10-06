"""
Created on Wed Jun 21 11:56:28 2017

@author: Andreas Georgopoulos
"""


from unidecode import unidecode
from skimage import io
from tqdm import tqdm
import requests
import dlib
import json
import time
import csv
import os


# Working directory
os.getcwd()

# Create folder 'posters' in wd
os.mkdir("posters")


# Import Movie Data -------------------------------------------------------
# produced at scrap_movie_data.py
with open('final_data/movies_data_final.json') as data_file:    
    movies_data = json.load(data_file)


# Extract Poster Links as a tuple (movie_name, poster_link) ---------------
poster_links = []
for i in  tqdm(range(len(movies_data))):
    poster_links.append((list(movies_data.items())[i][0], list(movies_data.items())[i][1]['Poster_Link']))
    time.sleep(0.01)
# Save list or urls:
with open('posters/poster_links_list', 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(poster_links)
# Read list of postcode urls
#with open('posters/poster_links_list', 'r') as f:
#    reader = csv.reader(f)
#    poster_links = list(reader)


# Download Movie Poster image ---------------------------------------------
for i in tqdm(range(len(movies_data))):
    if poster_links[i][1] != "N/A":
        img_data = requests.get(poster_links[i][1]).content
        # check for presence of path-used character '/' (it will break the code)
        image_name = poster_links[i][0]
        if('/' in image_name):
            image_name = image_name.replace("/","-")
        # unidecode for foreign titles with accents  
        with open('posters/'+unidecode(image_name)+'.jpg', 'wb') as handler:
            handler.write(img_data)
            
    time.sleep(0.01)


# Extract downloaded poster images' paths ----------------------------------
image_dir = "/Users/macuser/Desktop/Capstone/posters"    # change to yr current wd
# Image path
files = os.listdir(image_dir)
image_paths = [os.path.join(image_dir, f) for f in files if f.endswith("jpg")]


# Face Detector ------------------------------------------------------------
detector = dlib.get_frontal_face_detector()

image_and_facenumber_dic = {}

for i in tqdm(range(len(image_paths))):
    path =  image_paths[i]       
    image_id = path.split("/")[-1].split(".")[0]
    try:
        img = io.imread(path)
        # Run the face detector
        dets = detector(img, 1) 
        # Run the face detector, upsampling the image 3 time to find smaller faces.  (tradeoff cannot go to very small since it ll skip the whole pic but need to detect some smaller faces if exist. In any case even if for example in a poster exist 5 faces from which 2 are very small then we take as correct the answer of 3 main faces in the poster)
        dets_small = detector(img, 3) 
        # Find number of faces
        image_and_facenumber_dic[image_id] = max(len(dets), len(dets_small))
    except OSError:
        pass
    time.sleep(0.01)

# Save Number of faces data per movie
with open("image_and_facenumber_dic.json", "w") as output:
    json.dump(image_and_facenumber_dic, output, sort_keys=True, indent=4)



# Add feature to initial dataset -------------------------------------------

# Decode renamed movie names
renamed_movies = []
for i in tqdm(range(len(movies_data))):
    if poster_links[i][1] != "N/A":
        # check for presence of path-used character '/' (it wi
        image_name = poster_links[i][0]
        if('/' in image_name):
            renamed_movies.append(image_name)

# Add number of faces feature to initial dictionary for each movie (if no poster link then None)
for i in range(1,len(movies_data)):
    movie = poster_links[i][0]
    if(movie in renamed_movies):
        # Take renamed movie name saved in image_and_facenumber_dic
        movie_coded = movie.replace("/","-")
    else:
        movie_coded = movie
    # Add feature Poster_Num_Faces for each movie at initial mdic movie_data    
    if poster_links[i][1] == "N/A":
        movies_data[movie]["Poster_Num_Faces"] = None
    else:
        try:
            movies_data[movie]["Poster_Num_Faces"] = image_and_facenumber_dic[unidecode(movie_coded)]
        except:
            # either empty page or incorrect poster link
            movies_data[movie]["Poster_Num_Faces"] = None

# Save updated movie_data
with open("final_data/movies_data_final_numfaces.json", "w") as output:
    json.dump(movies_data, output, sort_keys=True, indent=4)






