#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 19:18:55 2017

@author: Andreas Georgopoulos
"""
from unidecode import unidecode
from bs4 import BeautifulSoup 
from tqdm import tqdm
import pandas as pd
import numpy  as np
import itertools
import httplib2
import html5lib
import string
import json
import omdb
import time
import csv
import re



def imdb_search_urls(movies):
    """
        Get urls of each provided movie on IMDb website    
    
        Input: Dataframe of movies including name and corresponding released year
        Output: List of urls in imdb of corresponding movie to scrap data from 
    """
       
    http = httplib2.Http()
    urls = []
    for m in range(len(movies)):
        year = int(movies.Year.iloc[m])
        name = movies.Movie.iloc[m]
        title_for_url = (name.replace(" ","+"))
        # search in imdb for the title_url
        imdb_search_link = "http://www.imdb.com/find?ref_=nv_sr_fn&q={}&s=tt".format(title_for_url)
        
        # Scrap data
        status, api_response = http.request(imdb_search_link)
        soup = BeautifulSoup(api_response, "html5lib")

        try:
            # get all searches and start parsing from top list to find the correct one based on title and year
            searches = soup.find("table","findList").findAll("td","result_text")
            
            correct_result = False
            i = 0                    
            while (correct_result == False & i < len(searches)):                
                # take result from top to bottom until u find the correct movie
                result = searches[i]
                # year of movie
                result_text = result.get_text()
                returned_movie_year = int(result_text[result_text.find("(")+1:result_text.find(")")])
            
                if(returned_movie_year == year):
                    correct_result = True
                    # get the href and title of the correct returned movie from the search result page
                    first_returned_movie_href = result.findAll("a")[0]['href']
                    # create href url for the movie
                    full_imdb_url = "http://www.imdb.com" + first_returned_movie_href
                    
                # if no exact match of movie and year in all results   
                if(i+1 == len(searches)):
                    # i.e. if movie foreign and release year in USA different from its original country, take first result
                    correct_result = True # terminate
                    full_imdb_url = "http://www.imdb.com" + searches[0].findAll("a")[0]['href']
                
                i = i + 1            
        except:
            # no search result
            full_imdb_url = None
        
        urls.append(full_imdb_url)
        
    return urls



def imdb_search_url_simple(movies):
    """
        Get urls of each movie on IMDb website (first search result)
    
        Input: Dataframe of movies including name 
        Output: List of urls in imdb of corresponding movie
    """
       
    http = httplib2.Http()
    urls = []
    for m in tqdm(range(len(movies))):
        name = movies.Movie.iloc[m]
        title_for_url = (name.replace(" ","+"))
        # search in imdb for the title_url
        imdb_search_link = "http://www.imdb.com/find?ref_=nv_sr_fn&q={}&s=tt".format(title_for_url)        
        # Scrap data
        status, api_response = http.request(imdb_search_link)
        soup = BeautifulSoup(api_response, "html5lib")
        try:
            # get all searches and start parsing from top list to find the correct one based on title and year
            searches = soup.find("table","findList").findAll("td","result_text")            
            # get the href and title of the first returned movie from the search result page
            first_returned_movie_href = searches[0].findAll("a")[0]['href']
            # create href url for the movie
            full_imdb_url = "http://www.imdb.com" + first_returned_movie_href                               
        except:
            # no search result
            full_imdb_url = None        
        
        urls.append(full_imdb_url)
        time.sleep(0.01)
        
    return urls



def scrap_movie_keywords(movie_url):
    """
        Input: str URL of movie in IMDb website
        Output: List of all movie's keywords
    """
    http = httplib2.Http()
    # Get movie's data
    status, api_response = http.request(movie_url)
    soup_movie = BeautifulSoup(api_response, "html5lib")
    
    
    # Find movie's keywords' URL -------------------------------------
    # check if keywords exist in the page
    movie_keywords = []
    if (soup_movie.find("div",{"class": "article","id": "titleStoryLine"}).find("div",{"itemprop": "keywords"})) is not None:
        keyword_url = "http://www.imdb.com" + soup_movie.find("div",{"class": "article","id": "titleStoryLine"}).find("div",{"itemprop": "keywords"}).findAll("a")[-1]['href']
        # Get keyowrds
        status_key, response_keyw = http.request(keyword_url)
        soup_key = BeautifulSoup(response_keyw, "html5lib")
        try:
            keywords_tags = soup_key.find("table",{"class": "dataTable evenWidthTable2Col"}).findAll("a", href = re.compile("keyword"))
            if(len(keywords_tags) > 0):
                # Return list of all keywords
                for k in range(len(keywords_tags)):
                    movie_keywords.append(keywords_tags[k].get_text())
            else:
                movie_keywords.append(None)
        except:
            movie_keywords.append(None)
    else:
        movie_keywords.append(None)
        
    return movie_keywords



def screen_scrap_number_of_trailers(movie_video_url):
    """
        Input: URL of movie's video gallery webpage
        Output: Number of trailers released
    """
    http = httplib2.Http()
    # Get video page data
    status_vid, api_response_vid = http.request(movie_video_url)
    soup_movie_vid = BeautifulSoup(api_response_vid, "html5lib")
    # Find Number of trailers
    num_trailers = None
    try:
        for line in soup_movie_vid.findAll("div","aux-content-widget-3"):
            try:
                for row in line.findAll("li"):
                    if(row.find("a").get_text() == "Trailer"):
                        num_trailers = re.split(r"[()]+",row.find("span").get_text())[1]
                        break
            except:
                pass
        
            if(num_trailers is not None):
                break
    except:
        pass
    
    return num_trailers



def screen_scrap_movie_data(movie_url):
    """
        Input: str URL of movie in IMDb website
        Output: List of all movie's keywords,List of all Leading Actors IDs and Names, List of all Directors IDs, List of all Production Studios IDs and Names
                        Production Budget and Domestic USA Gross
    """
    http = httplib2.Http()
    # Get movie's data
    status, api_response = http.request(movie_url)
    soup_movie = BeautifulSoup(api_response, "html5lib")
    
    prod_ids = []
    prod_names = []
    dir_ids = []
    act_ids = []
    act_names = []
    movie_keywords = []
    budget = None
    gross = None
    age_restr = None
    num_trailers = None
    
    
    if(soup_movie):
        # production studios Names & IDs ----------------------------------------------
        if(soup_movie.find("div",{"class": "article","id": "titleDetails"}) is not None):
            lines  = soup_movie.find("div",{"class": "article","id": "titleDetails"}).findAll("div",{"class": "txt-block"})
            for line in lines:
                for row in line.findAll("a"):
                    try:
                        if(row["href"].split("/")[1] == 'company'):
                            prod_ids.append(re.split(r"[/?]+",row["href"])[-2])
                            prod_names.append(row.get_text())
                    except:
                        pass
        else:
            prod_ids.append(None)
            prod_names.append(None)

        # Directors IDs  -------------------------------------------------------
        
        if(len(soup_movie.findAll("div",{"class": "credit_summary_item"})) > 0):
            if(soup_movie.findAll("div",{"class": "credit_summary_item"})[0] is not None):
                num_dir = len(soup_movie.findAll("div",{"class": "credit_summary_item"})[0].findAll("a"))
                for i in range(num_dir):
                    dir_ids.append(re.split(r"[/?]+",soup_movie.findAll("div",{"class": "credit_summary_item"})[0].findAll("a")[i]["href"])[-2])
            else:
                dir_ids.append(None)
        else:
            dir_ids.append(None)
        
        # Star Actors Names & IDs -------------------------------------------------------
        
        if(len(soup_movie.findAll("div",{"class": "credit_summary_item"})) >= 3):
            if(soup_movie.findAll("div",{"class": "credit_summary_item"})[2] is not None):
                num_star_actors = len(soup_movie.findAll("div",{"class": "credit_summary_item"})[2].findAll("a"))
                for i in range(num_star_actors-1):
                    act_ids.append(re.split(r"[/?]+",soup_movie.findAll("div",{"class": "credit_summary_item"})[2].findAll("a")[i]["href"])[-2])
                    act_names.append(soup_movie.findAll("div",{"class": "credit_summary_item"})[2].findAll("a")[i].get_text())
            elif(len(soup_movie.findAll("div",{"class": "credit_summary_item"})) >= 2):
                try:
                    if(soup_movie.findAll("div",{"class": "credit_summary_item"})[1] is not None):
                        num_star_actors = len(soup_movie.findAll("div",{"class": "credit_summary_item"})[1].findAll("a"))
                        for i in range(num_star_actors-1):
                            act_ids.append(re.split(r"[/?]+",soup_movie.findAll("div",{"class": "credit_summary_item"})[1].findAll("a")[i]["href"])[-2])
                            act_names.append(soup_movie.findAll("div",{"class": "credit_summary_item"})[1].findAll("a")[i].get_text())
                except:
                    act_ids.append(None)
                    act_names.append(None)
            else:
                act_ids.append(None)
                act_names.append(None)
        
            # Keywords --------------------------------------------------------------
    
            # Find movie's keywords' URL 
            # check if keywords exist in the page
            if (soup_movie.find("div",{"class": "article","id": "titleStoryLine"}) is not None):
                if (soup_movie.find("div",{"class": "article","id": "titleStoryLine"}).find("div",{"itemprop": "keywords"})) is not None:
                    keyword_url = "http://www.imdb.com" + soup_movie.find("div",{"class": "article","id": "titleStoryLine"}).find("div",{"itemprop": "keywords"}).findAll("a")[-1]['href']
                    # Get keyowrds
                    status_key, response_keyw = http.request(keyword_url)
                    soup_key = BeautifulSoup(response_keyw, "html5lib")
                    try:
                        keywords_tags = soup_key.find("table",{"class": "dataTable evenWidthTable2Col"}).findAll("a", href = re.compile("keyword"))
                        if(len(keywords_tags) > 0):
                            # Return list of all keywords
                            for k in range(len(keywords_tags)):
                                movie_keywords.append(keywords_tags[k].get_text())
                        else:
                            movie_keywords.append(None)
                    except:
                        movie_keywords.append(None)
                else:
                    movie_keywords.append(None)
            else:
                movie_keywords.append(None)
        
            # Production Budget & Gross ---------------------------------------------
            if(soup_movie.find("div",{"class": "article","id": "titleDetails"}) is not None):
                for line in soup_movie.find("div",{"class": "article","id": "titleDetails"}).findAll("div",{"class": "txt-block"}):
                    try:
                        if(line.find("h4", "inline").get_text() == 'Budget:'):
                            budget = line.get_text().split("\n")[1].split(":")[-1].replace(" ", "")
                        else:
                            pass
                        if(line.find("h4", "inline").get_text() == 'Gross:'):
                            gross = line.get_text().split("\n")[1].split(":")[-1].replace(" ", "")
                        else:
                            pass
                    except:
                        pass
                
            # Age Restrictions ------------------------------------------------------
            
            try:
                age_restr = soup_movie.find("div","title_wrapper").find("div","subtext").find("meta",{"itemprop":"contentRating"}).get('content')
            except:
                pass
     
            # Number of trailers ---------------------------------------------------------
            # Find url of videos' page
            url_videos = None
            if(soup_movie.find("div","caption") is not None):
                try:
                    url_videos = "http://www.imdb.com" + soup_movie.find("div","caption").findAll("a")[0]["href"]
                except:
                    pass
            elif(soup_movie.find("div","combined-see-more see-more") is not None):
                try:
                    for line in soup_movie.find("div","combined-see-more see-more").findAll("a"):
                        if(line.get_text().split(" ")[-1] in ["videos", "VIDEOS", "video", "VIDEO"]):
                            url_videos = "http://www.imdb.com" + line["href"]
                            break
                except:
                    pass
            # Scrap data from video gallery URL
            if(url_videos is not None):
                num_trailers = screen_scrap_number_of_trailers(url_videos)
            else:
                num_trailers = None
            
    return (movie_keywords, act_ids, act_names, dir_ids, prod_ids, prod_names, budget, gross, age_restr, num_trailers)





def scrap_movie_data(movie_url):
    """
        Input: str of Movie URL in IMDb website
        Output: Dictionary of each Movie and its characteristics scraped through IMDb & OMDb API 
    """
    movie_data_dic = {}
    
    # Extract movie ID from URL
    movie_id = movie_url.split("/")[-2]    
    # Get data from OMDb Database
    movie_data = omdb.get(imdbid = movie_id, fullplot = True, tomatoes = True)
    if(movie_data):
        # Movie ID
        movie_data_dic['Movie_id'] = movie_id
        # Release Date
        movie_data_dic['Released_Date'] = movie_data["released"]
        # Country
        movie_data_dic['Country'] = movie_data["country"].split(", ")
        # Genre
        movie_data_dic['Genre'] = movie_data["genre"].split(", ")
        # Duration
        movie_data_dic['Duration'] = movie_data["runtime"]
        # Directors
        movie_data_dic['Directors'] = movie_data["director"].split(", ")
        # Poster Link
        movie_data_dic['Poster_Link'] = movie_data["poster"]
        # Plot 
        movie_data_dic['Plot_Summary'] = movie_data["plot"]
        # Box_Office (domestic)
        movie_data_dic['Box_Office_USD'] = movie_data["box_office"]    
        # Keywords, Actors IDs, Actors Names, Directors IDs, Production Studios IDs, Production Studios Names
        movie_data_dic['Keywords'], movie_data_dic['Leading_Actors_ids'], movie_data_dic['Leading_Actors'], \
                  movie_data_dic['Directors_ids'], movie_data_dic['Production_studios_ids'], movie_data_dic['Production_studios'], \
                  movie_data_dic['Budget'], movie_data_dic['Box_Office_USD_2'], movie_data_dic['Age_Restrictions'], \
                                movie_data_dic['Num_Trailers'] = screen_scrap_movie_data(movie_url)
    
    return movie_data_dic
  



"""
    Scrap Data from "the-numbers.com" -------------------------------------------
    scrap list of movies in all times from its database
"""
http = httplib2.Http()
data_movies = {'Movie' : [], 'Release_Date' : [], 'Genre' : [], 'Production_Budget' : [], 'Domestic_Box_Office' : []}

# Parse each page (open url: http://www.the-numbers.com/movies/letter to check)
for i in (list(range(0, 9)) + list(string.ascii_uppercase)):
    # Create url
    url = 'http://www.the-numbers.com/movies/letter/{}'.format(i)

    # Get HTML
    status, response = http.request(url)
    soup = BeautifulSoup(response, "html5lib")
    # extract data and save in dictionary format
    for row in soup.find("table").findAll("tr"):
        if len(row) == 13:
            data_movies["Release_Date"].append(row.contents[1].get_text().replace(u'\xa0', u' '))
            data_movies["Movie"].append(row.contents[3].get_text().replace(u'\xa0', u' '))
            data_movies["Genre"].append(row.contents[5].get_text().replace(u'\xa0', u'0'))
            data_movies["Production_Budget"].append(row.contents[7].get_text().replace(u'\xa0', u'0'))
            data_movies["Domestic_Box_Office"].append(row.contents[9].get_text().replace(u'\xa0', u'0'))

# Convert to DataFrame
data_movies_numbers = pd.DataFrame(data_movies)
#data_movies_numbers = pd.read_csv('data_movies_df_with_nan.csv')
#data_movies_numbers =  data_movies_numbers.drop(['Unnamed: 0','Genre'], axis=1)

# Extract year
data_movies_numbers["Year"] = data_movies_numbers.Release_Date.str.split(", ", expand = True)[1].astype(int)

# Drop NA's in domestic_box_office only (from 28,664 to 15,227)
data_movies_numbers = data_movies_numbers[data_movies_numbers.Domestic_Box_Office != '$0']



"""
    ##################--- Get 15k movies' data ---#############################
"""
omdb.api._client.params_map['apikey'] = 'apikey'
omdb.set_default('apikey', '5cdc6e92')


# Get movies' URLs
data_movies_numbers['urls'] = imdb_search_url_simple(data_movies_numbers)  # check names of columns movie names and year to be: 'Movie' & 'Year'
data_movies_numbers.to_csv("data_movies_df_urls.csv", index = False)
#data_movies_df = pd.read_csv('data_movies_df_urls.csv')
# Convert nan to None
data_movies_df = data_movies_numbers.where((pd.notnull(data_movies_numbers)), None)

# Get movies' data    
movies_dic = {}

for m in tqdm(range(len(data_movies_df))):
    try:
        # If movie exist in imdb
        if(data_movies_df.urls.iloc[m] is not None):
            # Scrap data
            movies_dic[data_movies_df.Movie.iloc[m]] = scrap_movie_data(data_movies_df.urls.iloc[m])
    except:
        omdb.api._client.params_map['apikey'] = 'apikey'
        omdb.set_default('apikey', '5cdc6e92')
        http = httplib2.Http()
        if(data_movies_df.urls.iloc[m] is not None):
            # Scrap data
            movies_dic[data_movies_df.Movie.iloc[m]] = scrap_movie_data(data_movies_df.urls.iloc[m])

    time.sleep(0.01)

# Save movie data to json file
with open('final_data/movies_data_all.json', 'w') as output:
    json.dump(movies_dic, output, sort_keys=True, indent=4)


"""
    Correct scrapped movie's data for title mismatches & incorrect scraping
"""
# Import json file
with open('final_data/movies_data_5000_1.json') as data_file:    
    movies_dic = json.load(data_file)

# Create DataFrame of imported nested dic movies_data
movies_data_df = pd.DataFrame(list(movies_dic.values()), columns = list(list(movies_dic.values())[0].keys()))
movies_data_df['Movie'] = list(movies_dic.keys())
# Convert nan to None
movies_data_df = movies_data_df.where((pd.notnull(movies_data_df)), None)

# Drop rows with nan in more than four columns
movies_data_df = movies_data_df.dropna(thresh=4)
# reindex df
movies_data_df = movies_data_df.reset_index(drop=True)

# Unique Genres in DF
list(set(itertools.chain(*list(movies_data_df.Genre))))

# Remove movies with NA's in following features (result of a wrong mismatch)
index_list_to_drop = list(set().union(movies_data_df[movies_data_df.Leading_Actors_ids.map(set) & {None}].index.tolist(),\
                            movies_data_df[movies_data_df.Genre.map(set) & {'N/A','Reality-TV','TV-series','Talk-Show', 'News'}].index.tolist(),\
                                           movies_data_df[movies_data_df.Directors.map(set) & {'N/A'}].index.tolist() , \
                                                         movies_data_df[movies_data_df.Duration.map(set) & {'N/A'}].index.tolist()))

movies_data_df_corrrected = movies_data_df.drop(movies_data_df.index[index_list_to_drop])

# Turn df back to nested Dictionary and save as json
movies_data_dic_corrrected = movies_data_df_corrrected.set_index('Movie')[list(movies_data_df_corrrected.columns)[:-1]].T.to_dict()
# Save corrected movie data to json file
with open('final_data/movies_data_final.json', 'w') as output:
    json.dump(movies_data_dic_corrrected, output, sort_keys=True, indent=4)

"""
    ############################--- END ---####################################
"""


"""
    Run Poster Face Recognition ------------------------------------------------
"""



"""
    Merge Scrapped Movie data (all features) with "the-numbers.com" box office data
"""
# Import movies & box office scrapped from the-numbers  
#(15,227 movies with recorded gross) -------------------------------------------
data_movies_numbers = pd.read_csv('data_movies_df_with_nan.csv')
data_movies_numbers =  data_movies_numbers.drop(['Unnamed: 0','Genre'], axis=1)
# Extract year
data_movies_numbers["Year"] = data_movies_numbers.Release_Date.str.split(", ", expand = True)[1].astype(int)
# Drop NA's in domestic_box_office
data_movies_numbers = data_movies_numbers[data_movies_numbers.Domestic_Box_Office != '$0']


# Scrapped Movie data (14,036 movies) ------------------------------------------
with open('final_data/movies_data_final_numfaces.json') as data_file:    
    movies_data_final = json.load(data_file)
# Convert to df
movies_data_final_df = pd.DataFrame(list(movies_data_final.values()), columns = list(list(movies_data_final.values())[0].keys()))
movies_data_final_df['Movie'] = list(movies_data_final.keys())
# Convert nan to None
movies_data_final_df = movies_data_final_df.where((pd.notnull(movies_data_final_df)), None)

# Convert movie date from string to Date object
movies_data_final_df.loc[movies_data_final_df.Released_Date != 'N/A', "Released_Date"] = pd.to_datetime(movies_data_final_df.loc[movies_data_final_df.Released_Date != 'N/A', "Released_Date"], format = '%d %b %Y').dt.date
# Extract year of movie
movies_data_final_df["Year"] = ['N/A' if x == 'N/A' else x.year for x in movies_data_final_df['Released_Date']]


"""
 Merge dataframes for correct matched movies-----------------------------------
"""
# First by Year and Movie (avoid duplicates or movies different years in the number.com and incorrect matches when searching in imdb search by title)
movies_data_final_df_2 = pd.merge(movies_data_final_df, data_movies_numbers, on = ['Movie','Year'], how = 'inner')

# Find unmerged movies based on title and Year (analyse for different released dates appeared on the two daraframes)
df_unmerged = movies_data_final_df[~movies_data_final_df.Movie.isin(movies_data_final_df_2.Movie)]
# Find unmerged movies' data
df_unmerged = pd.merge(df_unmerged, data_movies_numbers, on = 'Movie', how = 'left')
# Extract correct ones: Movies with release date in imdb in country of origin whereas in numbers release date in US (threshold difference of one year)
movies_data_final_df_3 = df_unmerged[df_unmerged['Year_x'] != 'N/A'][abs(df_unmerged[df_unmerged['Year_x'] != 'N/A']['Year_y'] - df_unmerged[df_unmerged['Year_x'] != 'N/A']['Year_x']) < 2]
# Keep year from USA released date (year_y) n drop the other
movies_data_final_df_3 = movies_data_final_df_3.rename(index=str, columns={"Year_y": "Year"}).drop('Year_x', axis = 1)


# Then for those with 'N/A' in released date in imdb, merge by movie only
subset_na_date = movies_data_final_df[movies_data_final_df.Released_Date == 'N/A']
movies_data_final_df_4 = pd.merge(subset_na_date, data_movies_numbers, on = 'Movie', how = 'left')
# Keep year from USA released date (year_y) n drop the other
movies_data_final_df_4 = movies_data_final_df_4.rename(index=str, columns={"Year_y": "Year"}).drop('Year_x', axis = 1)

# Concatenate dfs
movie_data_final_df_cor = pd.concat([movies_data_final_df_2, movies_data_final_df_3, movies_data_final_df_4])

# Check for duplicates (keep last by index)
np.where(movie_data_final_df_cor.duplicated('Movie'))
movie_data_final_df_cor.loc[movie_data_final_df_cor.duplicated('Movie'),['Movie','Year','Domestic_Box_Office']]
movie_data_final_df_cor = movie_data_final_df_cor.drop_duplicates(subset='Movie', keep='last')
# By movie id
np.where(movie_data_final_df_cor.duplicated('Movie_id'))
movie_data_final_df_cor.loc[movie_data_final_df_cor.duplicated('Movie_id'),['Movie_id','Year','Domestic_Box_Office']]
movie_data_final_df_cor = movie_data_final_df_cor.drop_duplicates(subset='Movie_id', keep='last')


"""
    Include additional cast features 
    (actors, directors: experience, popularity, success) 
    Run corresponding files ---------------------------------------------------
"""

# Success Actors/Directors (awards scores) + Experience of Production Studios
cast_features = pd.read_csv('final_data/ActorsDirectorsStudios_Final.csv')

# Experience Actors/Directors + Incremental popularity (mean momentum)
exper_cast = pd.read_csv('final_data/experience_momentum_entropy.csv', encoding = "utf-8")

# Merge with Movie Data file by Movie
movie_data_final_df_cor = pd.merge(movie_data_final_df_cor, cast_features, on = 'Movie', how = 'inner')
movie_data_final_df_cor = pd.merge(movie_data_final_df_cor, exper_cast, on = 'Movie_id', how = 'inner')

# Check for duplicates (keep last by index)
np.where(movie_data_final_df_cor.duplicated('Movie_id'))
movie_data_final_df_cor.loc[movie_data_final_df_cor.duplicated('Movie'),['Movie','Year','Domestic_Box_Office']]
movie_data_final_df_cor = movie_data_final_df_cor.drop_duplicates(subset='Movie', keep='last')
# By movie id
np.where(movie_data_final_df_cor.duplicated('Movie_id'))
movie_data_final_df_cor.loc[movie_data_final_df_cor.duplicated('Movie_id'),['Movie_id','Year','Domestic_Box_Office']]
movie_data_final_df_cor = movie_data_final_df_cor.drop_duplicates(subset='Movie_id', keep='last')



# Turn df back to nested Dictionary and save as json
col_names = list(movie_data_final_df_cor.columns)
col_names.remove('Movie')
# Convert datetime: json doesn t support datetime object, transform to string
movie_data_final_df_cor.Released_Date = ['N/A' if x == 'N/A' else x.strftime('%Y-%m-%d') for x in movie_data_final_df_cor['Released_Date']] 
# Convert to a nested dictionary with primary key movie name
movies_data_final_dic_cor = movie_data_final_df_cor.set_index('Movie')[col_names].T.to_dict()
# Save corrected movie data to json file
with open('final_data/movies_data_final_all_preproc.json', 'w') as output:
    json.dump(movies_data_final_dic_cor, output, sort_keys=True, indent=4)