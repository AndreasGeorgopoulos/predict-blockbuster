#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 12:33:23 2017

@author: Andreas Georgopoulos
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import Imputer, MinMaxScaler
from gensim.models import word2vec
import matplotlib.pyplot as plt
from bisect import bisect_left
from bs4 import BeautifulSoup 
from datetime import date
from tqdm import tqdm
import pandas as pd
import numpy  as np
import itertools
import httplib2
import html5lib
import calendar
import logging
import pylab
import math
import json
import time
import re



# Import Movie Data (produced at scrap_movie_data.py)
with open('final_data/movies_data_final_all_preproc.json') as data_file:    
    movies_data_final = json.load(data_file)
    
# Create DataFrame of imported nested dic movies_data
movies_data_df_final = pd.DataFrame(list(movies_data_final.values()), columns = list(list(movies_data_final.values())[0].keys()))
movies_data_df_final['Movie_Name'] = list(movies_data_final.keys())

col_names_disagr_score = ['Movie_Name','Domestic_Box_Office','Country','Release_Date','Year', 'Genre','Duration','Production_Budget','Budget','Keywords','Poster_Num_Faces','Num_Trailers','Age_Restrictions','AvgStudiosBoxOffice','actors_avg_experience','actors_avg_momentum','directors_avg_experience','directors_avg_momentum','actors_experience_entropy','actors_momentum_entropy','NOscarWins', 'NOscarNominations', 'NGoldenWins',	'NGoldenNominations',	'NWins', 'NPlaces', 'NNominations','DirectorNOscarWins', 'DirectorNOscarNominations', 'DirectorNGoldenWins', 'DirectorNGoldenNominations', 'DirectorNWins', 'DirectorNPlaces', 'DirectorNNominations']

movies_data_df_final = movies_data_df_final[col_names_disagr_score]



"""
   ############################################################################
   ############################ Box Office USD ################################
   ############################################################################
"""

"""

# Adjust based on year's ticket price as: adj_box_office_2017 = (box_office_movie_year / avg_ricket_price_movie_year) * avg_ricket_price_2017
movies_data_df_final['Adj_Domestic_Box_Office_USD'] = movies_data_df_final.apply(lambda x: (x['Domestic_Box_Office'] / adj_prices_df.loc[x['Year']][0]) * adj_prices_df.loc[2017][0], axis = 1).astype(int)
"""

# Convert Domestic_Box_Office into integer (need to convert the commas in the numbers)
movies_data_df_final.Domestic_Box_Office = movies_data_df_final.Domestic_Box_Office.str.split("$").str[-1].str.replace(',','').astype(int)





"""
   ############################################################################
   ############################## Release Date ################################
   ############################################################################
"""

# Convert string Release Date to Date
movies_data_df_final.Release_Date = pd.to_datetime(movies_data_df_final.Release_Date, format = '%b %d, %Y').dt.date
# Extract day of week and create binary Feature of Friday_Weekend_Premiere date
#movies_data_df_final["Weekend_Premiere"] = [1 if calendar.day_name[x.weekday()] in ['Saturday','Sunday'] else 0 for x in movies_data_df_final['Release_Date']]
movies_data_df_final["Fri_Wknd_Premiere"] = [1 if calendar.day_name[x.weekday()] in ['Friday','Saturday','Sunday'] else 0 for x in movies_data_df_final['Release_Date']]



"""
   ############################################################################
   ############################# Ticket Prices ################################
   ############################################################################
"""
# Scrap Estimated ticket prices per year from boxofficemojo.com  ----------------------

url = 'http://www.boxofficemojo.com/about/adjuster.htm'
# Get HTML
http = httplib2.Http()
status, response = http.request(url)
soup = BeautifulSoup(response, "html5lib")

adj_prices = {'Year' : [], 'Avg_Ticket_Price_USD' : []}
for row in soup.find("table").findAll("tr"):
    if len(row) == 4:
        adj_prices["Year"].append(row.contents[0].get_text())
        adj_prices["Avg_Ticket_Price_USD"].append(row.contents[2].get_text().split("$")[-1])
# Convert dict to df
adj_prices_df = pd.DataFrame(adj_prices)
# Convert column types to float and int 
adj_prices_df.Avg_Ticket_Price_USD = adj_prices_df.Avg_Ticket_Price_USD.astype(float)
adj_prices_df.Year = adj_prices_df.Year.astype(int)
# Save file
adj_prices_df.to_csv('adj_prices_df.csv', index = False)
# Read
#adj_prices_df = pd.read_csv('adj_prices_df.csv')
# Reindex by year
adj_prices_df.set_index('Year', drop=False, inplace=True)

# Fill missing year values ---------------------------------------------------------------

# Add missing years between 1920 - 2017 (min,max movie years in our dataset)
adj_prices_df2 = pd.DataFrame({'Year':range(1910,2018), 'Avg_Ticket_Price_USD':0})
adj_prices_df2.set_index('Year', drop=True, inplace=True)
adj_prices_df2.Avg_Ticket_Price_USD = adj_prices_df.Avg_Ticket_Price_USD

# Fill missing avg ticket prices between 1920 and 1959 with avg price from previous and next recorded years
adj_prices_df.reset_index(drop=True, inplace = True)
adj_prices_df = adj_prices_df.iloc[::-1] # reverse
years = list(adj_prices_df.Year)         # take sequence of all years with recorded avg ticket price with reversed order

for year in adj_prices_df2[np.isnan(adj_prices_df2.Avg_Ticket_Price_USD)].index.tolist():
    # find index of previous and next year with records using bisection:
    next_year = bisect_left(years, year)
    prev_year = bisect_left(years, year) - 1
    adj_prices_df2.loc[year][0] = np.mean([adj_prices_df.Avg_Ticket_Price_USD.iloc[next_year],adj_prices_df.Avg_Ticket_Price_USD.iloc[prev_year]])
adj_prices_df = adj_prices_df2
adj_prices_df.reset_index(drop=False, inplace = True)

# Merge with initial dataframe by Year ---------------------------------------------------
movies_data_df_final = pd.merge(movies_data_df_final, adj_prices_df, on = 'Year', how = 'left')



"""
   ############################################################################
   ############################ Production Budget #############################
   ############################################################################
"""
# Fill NA's in Production_Budget(the-numbers) by corresponding values in Budget(imdb)
movies_data_df_final.Budget.fillna(movies_data_df_final.Production_Budget, inplace=True)
movies_data_df_final.drop('Production_Budget', axis=1, inplace=True)

# Get currencies
movies_data_df_final['Curerncy_Budget'] = movies_data_df_final.apply(lambda x: re.split(r'(\d)',  x['Budget'], maxsplit = 1)[0].replace("\xa0",""), axis = 1)
# Replace symbols with currency code
movies_data_df_final['Curerncy_Budget'] = ['EUR' if x=='€' else x for x in movies_data_df_final['Curerncy_Budget']]
movies_data_df_final['Curerncy_Budget'] = ['GBP' if x=='£' else x for x in movies_data_df_final['Curerncy_Budget']]
# Get list of all currencies needed transformations to USD
currencies = list(set(list(movies_data_df_final.loc[(movies_data_df_final.Curerncy_Budget !='$') & (movies_data_df_final.Curerncy_Budget != ''), 'Curerncy_Budget'])))

# Remove commas from numbers
movies_data_df_final.Budget = movies_data_df_final.Budget.str.replace(',','')
# Get int Budget
movies_data_df_final.Budget = movies_data_df_final.apply(lambda x: re.findall(r'\d+',  x['Budget'])[0], axis = 1).astype(int)


# Scrap yearly Historic Exchange Rates for all currencies in data
#         from fxtop.com  -------------------------------------------------------- 

hist_exchange_rates = {}
for cur in currencies:
    
    url = 'http://fxtop.com/en/historical-exchange-rates.php?A=1&C1=USD&C2={}&YA=1&DD1=&MM1=&YYYY1=1953&B=1&P=&I=1&DD2=30&MM2=06&YYYY2=2017&btnOK=Go%21'.format(cur)
    # Get HTML
    http = httplib2.Http()
    status, response = http.request(url)
    soup_rates = BeautifulSoup(response, "html5lib")
    
    exchange_rates = {}

    rows = soup_rates.find("table").findAll("td")[140].findAll("td")
    for y in range(5, len(rows),5):
        exchange_rates[rows[y].get_text()] = float(rows[y+1].get_text())
    
    hist_exchange_rates[cur] = exchange_rates
# Save movie data to json file
with open('final_data/hist_exchange_rates.json', 'w') as output:
    json.dump(hist_exchange_rates, output, sort_keys=True, indent=4)
# Read
#with open('final_data/hist_exchange_rates.json') as data_file:    
#    hist_exchange_rates = json.load(data_file)



# Transform  Budget to USD ----------------------------------------------------
# Get index of movies that need budget currency transformation (year after 1960)
movies_to_transform = movies_data_df_final.loc[(movies_data_df_final.Curerncy_Budget !='$') & (movies_data_df_final.Curerncy_Budget != '')  & (movies_data_df_final.Year > 1959)].index.tolist()
# Transform to USD based on exchange rate of currency and year of movie
for index in tqdm(movies_to_transform):
    movies_data_df_final.Budget.iloc[index] =  int(movies_data_df_final.Budget.iloc[index] / hist_exchange_rates[movies_data_df_final.Curerncy_Budget.iloc[index]][str(movies_data_df_final.Year.iloc[index])]) 
    time.sleep(0.01)
    
# Set 0 production budget for movies where no histoc exchange rates are available (year <1960)
for index in movies_data_df_final.loc[(movies_data_df_final.Curerncy_Budget !='$') & (movies_data_df_final.Curerncy_Budget != '')  & (movies_data_df_final.Year <= 1959)].index.tolist():
    movies_data_df_final.Budget.iloc[index] =  0

"""
# Adjust for inflation (same with box office) ---------------------------------
movies_data_df_final['Adj_Budget_USD'] = movies_data_df_final.apply(lambda x: (x['Budget'] / adj_prices_df.loc[x['Year']][0]) * adj_prices_df.loc[2017][0], axis = 1).astype(int)
"""
# Transform missing values from 0 to None 
#movies_data_df_final.loc[movies_data_df_final.Adj_Budget_USD == 0, 'Adj_Budget_USD'] = None
movies_data_df_final.loc[movies_data_df_final.Budget == 0, 'Budget'] = None
                    
# Drop Currency column
movies_data_df_final.drop('Curerncy_Budget', axis=1, inplace=True)



"""
   ############################################################################
   ######################### ---- Countries ---- ##############################
   ############################################################################
"""


"""
    Remove NA's in Countries and Keywords (662 movies) --------------------------------------
"""
# Convert all types of missing values to None for each faeture
movies_data_df_final.loc[movies_data_df_final.Country.map(set) & {'N/A',None,''}, 'Country'] = None
movies_data_df_final.loc[movies_data_df_final.Keywords.map(set) & {'N/A',None,''}, 'Keywords'] = None
movies_data_df_final.Duration = [None if x == 'N/A' else int(x.split(" ")[0]) for x in movies_data_df_final['Duration']]
movies_data_df_final_na = movies_data_df_final              # Keep this dataframe for final plot of initial NA's values

# Remove NA's 
movies_data_df_final.dropna(subset=['Keywords','Country'], inplace = True)


"""
 Country - categorical to dummies -----------------------------------------------------------
"""

# Create a set of dummy variables for each country at the Country feature
countries_dummies = movies_data_df_final.loc[movies_data_df_final.Country.str.len() > 0, 'Country'].apply(lambda x: pd.Series([1] * len(x), index=x)).fillna(0, downcast='infer')

# Count occurences per country and sort
countries_dummies_sort = countries_dummies.sum(axis=0).reset_index(name="Occurence").sort_values(by = 'Occurence', ascending = False)
# Most popular countries (occurence > 40 movies)
countries_pop = list(countries_dummies_sort.loc[countries_dummies_sort.Occurence > 40, 'index'])

# Group by countries with less than 40 movies as "Other" ----------------------
# Exclude non popular countries
countries_names_to_transform = list(set(list(countries_dummies.columns)) - set(countries_pop))
# Sum by row the binary columns of countries to tranform and if >0 then put 1 in 'Other' column
countries_dummies['Other'] = countries_dummies[countries_names_to_transform].apply (lambda row: sum(row),axis=1)
countries_dummies['Other'] = [1 if x>0 else 0 for x in countries_dummies['Other']]
# Drop columns of transformed countries
countries_dummies.drop(countries_names_to_transform, axis = 1, inplace = True)

# Join with initial dataset
movies_data_df_final = movies_data_df_final.join(countries_dummies)


# Create Binaary Column indicating if a movie included produced in mroe than one country
movies_data_df_final['Multiple_Countries']  = [1 if len(x)>1 else 0 for x in movies_data_df_final['Country']]
                       
                       
"""
    Country combinations --------------------------------------------------------------------
"""   
# Based on Country's Analysis (see initial_findings) 
#     add dummies of 30 most successful of the most popular genre's combinations

# Sort each Country List alhabetically------------------------------------------------
#   (so that 'USA, Japan' and 'Japan,USA' be considered same combination)
movies_data_df_final['Country'] = movies_data_df_final['Country'].apply(lambda x: sorted(x))

# Find most popular combinations of Countries (gross) --------------------------------
country_dummies_combined = movies_data_df_final['Country'].str.get_dummies()
countries_comb_df = pd.DataFrame({'Country':list(country_dummies_combined.columns), 'Number_of_Movies':-1})
for col in countries_comb_df.Country:
    countries_comb_df.loc[countries_comb_df.Country == col, 'Number_of_Movies'] = country_dummies_combined[col].sum()
# Sort df
country_comb_df = countries_comb_df.sort_values(by = 'Number_of_Movies', ascending = False)
# Transform Country names
country_comb_df.Country = country_comb_df.Country.str.replace("[","").str.replace("]","").str.replace(","," |").str.replace("'","")
# Extract top 30 most popular combinations
country_comb_df.head(100)


# Find MOST successful combinations of countries (gross) of 
#          the 100 most popular ones -----------------------------------------------
movies_data_df_final['Country_comb'] = movies_data_df_final.Country.str.join(' | ')
# Compute average gross per genre combination
countries_comb_gross = movies_data_df_final.groupby('Country_comb')['Domestic_Box_Office'].mean().astype(int).reset_index(name="Average_Domestic_Box_Office_USD")
# Sort df
countries_comb_gross = countries_comb_gross.sort_values(by = 'Average_Domestic_Box_Office_USD', ascending = False)

# Plot distribution of 30 most successful (average gross) combinations of countries
countries_comb_gross_subset = movies_data_df_final.loc[movies_data_df_final.Country_comb.isin(list(countries_comb_gross.Country_comb.head(30))), ['Domestic_Box_Office','Country_comb']]                                          
countries_comb_gross_subset = countries_comb_gross_subset.sort_values(by = 'Country_comb', ascending = True)

# Compute average gross per genre combination of the 100 most popular 
countries_comb_gross_top30 = movies_data_df_final.loc[movies_data_df_final.Country_comb.isin(list(country_comb_df.Country.head(100))), ['Domestic_Box_Office','Country_comb']].groupby('Country_comb')['Domestic_Box_Office'].mean().astype(int).reset_index(name="Average_Domestic_Box_Office_USD")
# Sort df
countries_comb_gross_top30 = countries_comb_gross_top30.sort_values(by = 'Average_Domestic_Box_Office_USD', ascending = False)

# Exclude one-country entries
countries_comb_gross_top30 = countries_comb_gross_top30[countries_comb_gross_top30.Country_comb.apply(lambda x: '|' in x)]

# Get list of top30 most successful (of the most popular) countries combinations
countries_comb_top30 = list(countries_comb_gross_top30.head(30).Country_comb)

# Plot -------------------------------------
import seaborn as sns

# Plot distribution of 30 most successful (average gross) of most popular combinations of genres
countr_comb_gross_subset_top30 = movies_data_df_final.loc[movies_data_df_final.Country_comb.isin(list(countries_comb_gross_top30.Country_comb.head(30))), ['Domestic_Box_Office','Country_comb']]                                          
countr_comb_gross_subset_top30 = countr_comb_gross_subset_top30.sort_values(by = 'Domestic_Box_Office', ascending = False)

sns.set(style="white", color_codes=True, font_scale=1.2)
g = sns.boxplot(x="Domestic_Box_Office", y="Country_comb", data=countr_comb_gross_subset_top30)
# show also observations as point to show density
sns.stripplot(x="Domestic_Box_Office", y="Country_comb", data=countr_comb_gross_subset_top30, jitter=True, size=1.5, color="maroon", linewidth=0)
# Log scale
g.set_xscale("log")
sns.despine(trim=True)
sns.despine(offset=10, trim=True)
plt.xticks(g.get_xticks(), ["$1k","$10k","$100k", "$1m", "$10m", "$100m","$1b",'$10b'])
plt.xlabel('Domestic Box Office (USD)')
plt.ylabel('')
plt.title("Most Successful of the most Frequent Countries' Combinations", fontweight='bold')
plt.savefig('Plots_Final/plt_boxplot_suc_country_top30', bbox_inches='tight') 



# Create top30 countries combinations dummies -------------------------------------
countries_comb_all_dummies = pd.get_dummies(movies_data_df_final['Country_comb'])
countries_comb_top30_dummies = countries_comb_all_dummies[countries_comb_top30]
# Join with initial dataset
movies_data_df_final = movies_data_df_final.join(countries_comb_top30_dummies)

# Delete Country_comb column
movies_data_df_final.drop('Country_comb', axis=1, inplace=True)




# Find LEAST successful combinations of countries (gross) of 
#          the 100 most popular ones -----------------------------------------------
movies_data_df_final['Country_comb'] = movies_data_df_final.Country.str.join(' | ')
# Compute average gross per genre combination
countries_comb_gross = movies_data_df_final.groupby('Country_comb')['Domestic_Box_Office'].mean().astype(int).reset_index(name="Average_Domestic_Box_Office_USD")
# Sort df
countries_comb_gross = countries_comb_gross.sort_values(by = 'Average_Domestic_Box_Office_USD', ascending = False)

# Plot distribution of 30 least successful (average gross) combinations of countries
countries_comb_gross_subset = movies_data_df_final.loc[movies_data_df_final.Country_comb.isin(list(countries_comb_gross.Country_comb.tail(30))), ['Domestic_Box_Office','Country_comb']]                                          
countries_comb_gross_subset = countries_comb_gross_subset.sort_values(by = 'Country_comb', ascending = True)

# Compute average gross per genre combination of the 100 most popular 
countries_comb_gross_bottom30 = movies_data_df_final.loc[movies_data_df_final.Country_comb.isin(list(country_comb_df.Country.head(100))), ['Domestic_Box_Office','Country_comb']].groupby('Country_comb')['Domestic_Box_Office'].mean().astype(int).reset_index(name="Average_Domestic_Box_Office_USD")
# Sort df
countries_comb_gross_bottom30 = countries_comb_gross_bottom30.sort_values(by = 'Average_Domestic_Box_Office_USD', ascending = False)

# Exclude one-country entries
countries_comb_gross_bottom30 = countries_comb_gross_bottom30[countries_comb_gross_bottom30.Country_comb.apply(lambda x: '|' in x)]

# Get list of bottom30 most successful (of the most popular) countries combinations
countries_comb_bottom30 = list(countries_comb_gross_bottom30.tail(30).Country_comb)


# Plot -------------------------------------
import seaborn as sns

# Plot distribution of 30 most successful (average gross) of most popular combinations of genres
countr_comb_gross_subset_least30 = movies_data_df_final.loc[movies_data_df_final.Country_comb.isin(countries_comb_bottom30), ['Domestic_Box_Office','Country_comb']]                                          
countr_comb_gross_subset_least30 = countr_comb_gross_subset_least30.sort_values(by = 'Domestic_Box_Office', ascending = False)

sns.set(style="white", color_codes=True, font_scale=1.2)
g = sns.boxplot(x="Domestic_Box_Office", y="Country_comb", data=countr_comb_gross_subset_least30)
# show also observations as point to show density
sns.stripplot(x="Domestic_Box_Office", y="Country_comb", data=countr_comb_gross_subset_least30, jitter=True, size=1.5, color="maroon", linewidth=0)
# Log scale
g.set_xscale("log")
sns.despine(trim=True)
sns.despine(offset=10, trim=True)
plt.xticks(g.get_xticks(), ["$1k","$10k","$100k", "$1m", "$10m"])
plt.xlabel('Domestic Box Office (USD)')
plt.ylabel('')
plt.title("Least Successful of the most Frequent Countries' Combinations", fontweight='bold')
plt.savefig('Plots_Final/plt_boxplot_suc_country_least30', bbox_inches='tight') 


# Create top30 countries combinations dummies -------------------------------------
countries_comb_all_dummies = pd.get_dummies(movies_data_df_final['Country_comb'])
countries_comb_bottom30_dummies = countries_comb_all_dummies[countries_comb_bottom30]

# Join with initial dataset
movies_data_df_final = movies_data_df_final.join(countries_comb_bottom30_dummies)

# Delete Country_comb column
movies_data_df_final.drop('Country_comb', axis=1, inplace=True)
                     
  
                                              
  
"""
   ############################################################################
   ############################ Age Restrictions ##############################
   ############################################################################
"""

# Group by AgE Restriction Labels as following: {G: A, AA, E, U, NOT RATED, PG: GP, PG-13: 12,12A, R:15, NC-17: 18, X, (BANNED)}
movies_data_df_final.loc[movies_data_df_final.Age_Restrictions.isin(['A','AA','E','U','NOT RATED']),'Age_Restrictions'] = 'G'
movies_data_df_final.loc[movies_data_df_final.Age_Restrictions.isin(['GP']),'Age_Restrictions'] = 'PG'
movies_data_df_final.loc[movies_data_df_final.Age_Restrictions.isin(['12','12A']),'Age_Restrictions'] = 'PG-13'
movies_data_df_final.loc[movies_data_df_final.Age_Restrictions.isin(['15']),'Age_Restrictions'] = 'R'
movies_data_df_final.loc[movies_data_df_final.Age_Restrictions.isin(['18','X','REJECTED','(BANNED)']),'Age_Restrictions'] = 'NC-17'

# Create a set of dummy variables for each age restriction label (n-1, one for reference)
age_rest_dummies = pd.get_dummies(movies_data_df_final.Age_Restrictions, drop_first = True)
# Count occurences per age label and sort
age_dummies_sort = age_rest_dummies.sum(axis=0).reset_index(name="Occurence").sort_values(by = 'Occurence', ascending = False)
# Join with initial dataset
movies_data_df_final = movies_data_df_final.join(age_rest_dummies)



"""
   ############################################################################
   ################################ Duration ##################################
   ############################################################################
"""

# Transform missing values from 0 to None
movies_data_df_final.loc[movies_data_df_final.Duration == 0, 'Duration'] = None
                        
# Create bucket effects - dummies: short: 1-80min, medium: 81- 130min, long: >131 min (one hot encoding n-1)
movies_data_df_final['duration_short'] = [1 if x <=80 else 0 for x in movies_data_df_final['Duration']]
movies_data_df_final['duration_medium'] = [1 if((x >80) & (x<=130)) else 0 for x in movies_data_df_final['Duration']]
#movies_data_df_final['duration_long'] = [1 if x >130 else 0 for x in movies_data_df_final['Duration']]
                    
"""
   ############################################################################
   ################################ Trailers ##################################
   ############################################################################


"""
# Convert to int
movies_data_df_final.Num_Trailers = pd.to_numeric(movies_data_df_final.Num_Trailers, errors='coerce')
                           
                        
"""
   ############################################################################
   ########################### ---- Genres ---- ###############################
   ############################################################################
"""

"""
 Genre - categorical to dummies -----------------------------------------------------------
"""
# Create a set of dummy variables for each genre at the Genre feature (fill NaN with 0 in corresponding dummy)
genres_dummies = movies_data_df_final.loc[movies_data_df_final.Genre.str.len() > 0, 'Genre'].apply(lambda x: pd.Series([1] * len(x), index=x)).fillna(0, downcast='infer')
# Join with initial dataset
movies_data_df_final = movies_data_df_final.join(genres_dummies)
# Transform missing values to None
movies_data_df_final.loc[movies_data_df_final.Genre.map(set) & {'N/A',None,''}, 'Genre'] = None

# Create Binaary Column indicating if a movie included in more than one Genres
movies_data_df_final['Multiple_Genres']  = [1 if len(x)>1 else 0 for x in movies_data_df_final['Genre']]
                         
# Remove Short films (64)
movies_data_df_final = movies_data_df_final[movies_data_df_final.Short == 0]
movies_data_df_final.drop('Short', axis=1, inplace=True)
genres_dummies.drop('Short', axis=1, inplace=True)


"""
    Genre combinations --------------------------------------------------------------------
"""   
# Based on Genre's Analysis (see initial_findings) add dummies of 30 most successful of the most popular genre's combinations
# Find most popular combinations of genres (gross) ----------------------------------
genres_dummies_combined = movies_data_df_final['Genre'].str.get_dummies()
genres_comb_df = pd.DataFrame({'Genre':list(genres_dummies_combined.columns), 'Number_of_Movies':-1})
for col in genres_comb_df.Genre:
    genres_comb_df.loc[genres_comb_df.Genre == col, 'Number_of_Movies'] = genres_dummies_combined[col].sum()
# Sort df
genres_comb_df = genres_comb_df.sort_values(by = 'Number_of_Movies', ascending = False)
# Transform Genre names
genres_comb_df.Genre = genres_comb_df.Genre.str.replace("[","").str.replace("]","").str.replace(","," |").str.replace("'","")
# Extract top 30 most popular combinations
genres_comb_df.head(30)


# Find MOST successful combinations of genres (gross) of 
#          the 100 most popular ones -----------------------------------------------
movies_data_df_final['Genre_comb'] = movies_data_df_final.Genre.str.join(' | ')
# Compute average gross per genre combination
genres_comb_gross = movies_data_df_final.groupby('Genre_comb')['Domestic_Box_Office'].mean().astype(int).reset_index(name="Average_Domestic_Box_Office_USD")
# Sort df
genres_comb_gross = genres_comb_gross.sort_values(by = 'Average_Domestic_Box_Office_USD', ascending = False)

# Plot distribution of 30 most successful (average gross) combinations of genres
genres_comb_gross_subset = movies_data_df_final.loc[movies_data_df_final.Genre_comb.isin(list(genres_comb_gross.Genre_comb.head(30))), ['Domestic_Box_Office','Genre_comb']]                                          
genres_comb_gross_subset = genres_comb_gross_subset.sort_values(by = 'Genre_comb', ascending = True)

# Compute average gross per genre combination of the 100 most popular 
genres_comb_gross_top30 = movies_data_df_final.loc[movies_data_df_final.Genre_comb.isin(list(genres_comb_df.Genre.head(100))), ['Domestic_Box_Office','Genre_comb']].groupby('Genre_comb')['Domestic_Box_Office'].mean().astype(int).reset_index(name="Average_Domestic_Box_Office_USD")
# Sort df
genres_comb_gross_top30 = genres_comb_gross_top30.sort_values(by = 'Average_Domestic_Box_Office_USD', ascending = False)
# Exclude one-genre entries
genres_comb_gross_top30 = genres_comb_gross_top30[genres_comb_gross_top30.Genre_comb.apply(lambda x: '|' in x)]

# Get list of top30 most successful (ot the most popular) genres combinations
genres_comb_top30 = list(genres_comb_gross_top30.head(30).Genre_comb)

# Create top30 genres combinations dummies -------------------------------------
genres_comb_all_dummies = pd.get_dummies(movies_data_df_final['Genre_comb'])
genres_comb_top30_dummies = genres_comb_all_dummies[genres_comb_top30]

# Join with initial dataset
movies_data_df_final = movies_data_df_final.join(genres_comb_top30_dummies)

# Delete Genre_comb column
movies_data_df_final.drop('Genre_comb', axis=1, inplace=True)



# Find LEAST successful combinations of genres (gross) of 
#          the 100 most popular ones -----------------------------------------------
movies_data_df_final['Genre_comb'] = movies_data_df_final.Genre.str.join(' | ')
# Compute average gross per genre combination
genres_comb_gross = movies_data_df_final.groupby('Genre_comb')['Domestic_Box_Office'].mean().astype(int).reset_index(name="Average_Domestic_Box_Office_USD")
# Sort df
genres_comb_gross = genres_comb_gross.sort_values(by = 'Average_Domestic_Box_Office_USD', ascending = False)

# Plot distribution of 30 most successful (average gross) combinations of genres
genres_comb_gross_subset = movies_data_df_final.loc[movies_data_df_final.Genre_comb.isin(list(genres_comb_gross.Genre_comb.tail(30))), ['Domestic_Box_Office','Genre_comb']]                                          
genres_comb_gross_subset = genres_comb_gross_subset.sort_values(by = 'Genre_comb', ascending = True)

# Compute average gross per genre combination of the 100 most popular 
genres_comb_gross_bottom30 = movies_data_df_final.loc[movies_data_df_final.Genre_comb.isin(list(genres_comb_df.Genre.head(100))), ['Domestic_Box_Office','Genre_comb']].groupby('Genre_comb')['Domestic_Box_Office'].mean().astype(int).reset_index(name="Average_Domestic_Box_Office_USD")
# Sort df
genres_comb_gross_bottom30 = genres_comb_gross_bottom30.sort_values(by = 'Average_Domestic_Box_Office_USD', ascending = False)
# Exclude one-genre entries
genres_comb_gross_bottom30 = genres_comb_gross_bottom30[genres_comb_gross_bottom30.Genre_comb.apply(lambda x: '|' in x)]

# Get list of top30 most successful (ot the most popular) genres combinations
genres_comb_bottom30 = list(genres_comb_gross_bottom30.tail(30).Genre_comb)

# Create top30 genres combinations dummies -------------------------------------
genres_comb_all_dummies = pd.get_dummies(movies_data_df_final['Genre_comb'])
genres_comb_bottom30_dummies = genres_comb_all_dummies[genres_comb_bottom30]

# Join with initial dataset
movies_data_df_final = movies_data_df_final.join(genres_comb_bottom30_dummies)

# Delete Genre_comb column
movies_data_df_final.drop('Genre_comb', axis=1, inplace=True)


"""
   ############################################################################
   ######################### ---- Keywords ---- ###############################
   ############################################################################
   
   Identify most important keywords to consider as dummies
"""

"""
    TF-IDF Computations  ------------------------------------------------------
"""
"""
# List of keywords (by word analysis and not phrase) ---------------------------
# Get for each genre (document) a list of keywords in all movies 
# and add to collection of keywords 
collection_keywords_all = []
# Convert list of keywords in each movie into a single spaced string
movies_data_df_final.Keywords = movies_data_df_final.Keywords.apply(lambda x: ' '.join(x))


genres_list = list(genres_dummies.columns)
for genre in genres_list:
    collection_keywords_all = list(movies_data_df_final.loc[movies_data_df_final[genre] == 1,'Keywords'])


# Initialise vectoriser
tfidf_vectorizer = TfidfVectorizer(norm='l2',min_df = 1,use_idf=True, stop_words = 'english')
# Fit vectoriser and compute tf-idf values
tfidf_matrix = tfidf_vectorizer.fit_transform(collection_keywords_all)
# Print tf-idf matrix
print(tfidf_matrix.todense())
# Get feature names 
features_names = tfidf_vectorizer.get_feature_names()
"""

# List of lists of keywords ---------------------------------------------------


# Compute collection of genres (documents)'keywords 
num_keyw = 0        # total number of keywords
num_keyw_unique = 0 # total number of unique keywords
collection_keywords_all_2 = []  # Collection of documents(genres) keywords
genres_list = list(genres_dummies.columns)
for genre in genres_list:
    genre_keywords = []
    # find movies in genre and exctract keywords lists and merge into one
    genre_keywords = list(itertools.chain.from_iterable(list(movies_data_df_final.loc[movies_data_df_final[genre] == 1,'Keywords'])))
    num_keyw = num_keyw + len(genre_keywords)
    num_keyw_unique = num_keyw_unique + len(list(set(genre_keywords)))
    collection_keywords_all_2.append(genre_keywords)


# Initialise vectoriser (analyse by keuword phrase and not by word)
tfidf_vectorizer2 = TfidfVectorizer(norm='l2',min_df = 1,use_idf=True, stop_words = 'english',tokenizer=lambda doc: doc, lowercase=False)
tfidf_matrix_2 = tfidf_vectorizer2.fit_transform(collection_keywords_all_2)
features_names_2 = tfidf_vectorizer2.get_feature_names()



def top_tfidf_keywords_genre(row_id, features, top_n=25):
    """
        Input: the row of the tfidf matrix (genre id), the keywords(as feature names), number of keywords with top tf-idf values to extract
        Output: Dataframe of top n keywords and corresponding td-idf value
    """
    
    
    if(type(row_id) == int):
        # Convert row number into dense format
        row = np.squeeze(tfidf_matrix_2[row_id].toarray())
    else:
        row = row_id
    # Indecies of sorted faetures by tf-idf value in descending order
    topn_ids = np.argsort(row)[::-1][:top_n]
    # Extract top n features and values
    top_feats = [(features[i], row[i]) for i in topn_ids]
    # Return as dataframe
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df
       

def top_avg_tfidf_all_genres(matrix, features, ids= None,min_tfidf=0.1, top_n=25):
    """ 
        Input: tf-idf matrix, features names, specific index of rows(documents), 
                threshold tf-idf value to consider, top_n features to consider
        Output: Return top n keywords based on id-idf values in all documents(genres)
    """
    # If specific documents to find avg df-idf
    if ids:
        D = matrix[ids].toarray()
    else:
        # else take whole collection(all genres)
        D = matrix.toarray()
    # Exclude keywords with di-idf < threshold
    D[D < min_tfidf] = 0
    tfidf_mean = np.mean(D, axis=0)
    
    return top_tfidf_keywords_genre(tfidf_mean, features, top_n)


# Find 100 keywords with highest avg tf-idf in all documents (genres)
keywords_top100_tfidf = top_avg_tfidf_all_genres(tfidf_matrix_2, features_names_2, None, 0.1, 100)


# Plot top n keywords with highest tfidf
def plot_kwrds(col_nan_df):
    """
        Input: Dataframe of Features(i.e.Genres) and corresponding Count variable
        Output: Barplot of Count variable per feature (sorted)
    """
    # Sort features by number of NA
    col_nan_df = col_nan_df.sort_values(by = 'tfidf', ascending = True)
    
    # Barplot
    fig, ax = plt.subplots()

    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    ax.set_facecolor('white')
    
    # Add labels in each bar
    for i, v in enumerate(list(col_nan_df.tfidf)):
        ax.text(v + 0.0008, i + .25, str(round(v,4)), size = 12)
    
    pos = pylab.arange(len(col_nan_df))+.5     
    plt.barh(pos,list(col_nan_df.tfidf), align='center')
    plt.yticks(pos, list(col_nan_df.feature), fontweight='bold')
    plt.xlabel('TF-IDF value in all genres (documents)')
    plt.title("25 Most Important Keywords in Collection of Movies' Keyword Phrases", fontweight='bold')    
    plt.savefig('Plots_Final/plot_kwrds_tfidf', bbox_inches='tight')    
    plt.show()

plot_kwrds(keywords_top100_tfidf.head(25))



"""
    Word2Vec Model  -----------------------------------------------------------
"""

# Train word2vec model on all genres(documents) movies' keywords ----------------
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

num_features = 500    # Word vector dimensionality                      
min_word_count = 30    # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model
model = word2vec.Word2Vec(collection_keywords_all_2, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)

# Call init_sims for  memory-efficiency
model.init_sims(replace=True)
# Save model
model.save('keywords_model_word2vec')
# Number of words in vocabulary
len(model.wv.vocab)

# Extract word vectors
word_vectors = model.wv
# Save wv
word_vectors.save('word_vectors_keywords')
#Read wv
#word_vectors = KeyedVectors.load(fname)


"""
    Identify most important keywords (and their similar ones) -----------------
"""
# For each of the top 100 keywords based on tf-idf find their top2 similar ones
keywords_top100_tfidf['similar_features'] = keywords_top100_tfidf.apply(lambda x: [], axis=1)
for i in range(len(keywords_top100_tfidf)):
    try:
        # For each keyword find top2 similar keywords (fist item returned same)
        sim_kwds_tfidf = model.similar_by_vector(model[keywords_top100_tfidf.feature.iloc[i]], topn=3)
        # Extract names of similar keywords
        sim_kwds = [x[0] for x in sim_kwds_tfidf[1:]]
        # Add to column
        keywords_top100_tfidf.similar_features.iloc[i] = sim_kwds
    except:
        keywords_top100_tfidf.similar_features.iloc[i] = None

# Find similar features in top100 list and keep unique ones
list_top_100 = list(keywords_top100_tfidf.feature)
redundant_features = [] # if two keywords that appear in top100 list are in the similar_features of one another
for i in range(len(keywords_top100_tfidf)):
    if(keywords_top100_tfidf.similar_features.iloc[i] is not None):
        redunants = list(set(keywords_top100_tfidf.similar_features.iloc[i]).intersection(set(list_top_100)))
        if(keywords_top100_tfidf.feature.iloc[i] not in redundant_features):
            # If similar features are already in the top100 list:
            if(len(redunants)>0):
                for j in range(len(redunants)):
                    if(redunants[j] not in redundant_features):
                        if(keywords_top100_tfidf.feature.iloc[i] in keywords_top100_tfidf.loc[keywords_top100_tfidf.feature == redunants[j],'similar_features']):
                            redundant_features.append(redunants[j])
# If exist redundant features drop them from top100_tf-idf 
if redundant_features:
    keywords_top100_tfidf = keywords_top100_tfidf[~keywords_top100_tfidf['feature'].isin(redundant_features)]

# Join keyword and similar features into a list
for i in range(len(keywords_top100_tfidf)):
    if(keywords_top100_tfidf.similar_features.iloc[i] is not None):
        keywords_top100_tfidf.similar_features.iloc[i] = keywords_top100_tfidf.similar_features.iloc[i] + [keywords_top100_tfidf.feature.iloc[i]]
    else:
        keywords_top100_tfidf.similar_features.iloc[i] = [keywords_top100_tfidf.feature.iloc[i]]

# Check in movies keywords if each of the top100_tf-idf keywords (or their similar ones) exist and add to a list
movies_data_df_final['top100_TFIDF'] = movies_data_df_final.apply(lambda x: [], axis=1)
for i in tqdm(range(len(movies_data_df_final))):
    for j in range(len(keywords_top100_tfidf)):
        # Find union of lists of keywords (movie's and top100_tf-idf)
        keywords_union = list(set(movies_data_df_final.Keywords.iloc[i]).intersection(set(keywords_top100_tfidf.similar_features.iloc[j])))
        if(len(keywords_union) > 0):
            for k in range(len(keywords_union)):
                # append top100 feature (if this or one of its similar features exist)
                movies_data_df_final.top100_TFIDF.iloc[i].append(keywords_top100_tfidf.feature.iloc[j])
        else:
            movies_data_df_final.top100_TFIDF.iloc[i].append('N/A')
    
    time.sleep(0.01)      

# Get unique top100_tf-idf keywords in each movie
movies_data_df_final['top100_TFIDF'] = movies_data_df_final['top100_TFIDF'].apply(lambda x: list(set(x)))


# Create dummies --------------------------------------------------------------
top100_tfidf_dummies = movies_data_df_final.loc[movies_data_df_final.top100_TFIDF.str.len() > 0, 'top100_TFIDF'].apply(lambda x: pd.Series([1] * len(x), index=x)).fillna(0, downcast='infer')
# Delete column 'N/A'
top100_tfidf_dummies.drop('N/A', axis=1, inplace=True)
# Join with initial dataset
movies_data_df_final = movies_data_df_final.join(top100_tfidf_dummies)

# Delete column top100_TFIDF
movies_data_df_final.drop('top100_TFIDF', axis=1, inplace=True)

"""
   ############################################################################
   ############################# END - Keywords ###############################
   ############################################################################
"""





"""
   ############################################################################
   ############################# Missing Values ###############################
   ############################################################################
"""


"""
 Plot NA's --------------------------------------------------------------------------
"""

# Find number of missing values in each feature
col_nan_df = pd.DataFrame({'Feature':list(movies_data_df_final_na.columns), 'Number_of_NaN':-1})
for col in col_nan_df.Feature:
    col_nan_df.loc[col_nan_df.Feature == col, 'Number_of_NaN'] = movies_data_df_final_na[col].isnull().sum()

print(col_nan_df)

# Plot 
def plot_NA(col_nan_df):
    """
        Input: Dataframe of Features and corresponding number of NA's
        Output: Barplot of Num of NA's per feature (sorted)
    """
    # Sort features by number of NA
    col_nan_df = col_nan_df.sort_values(by = 'Number_of_NaN', ascending = True)
    
    # Barplot
    fig, ax = plt.subplots()

    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    ax.set_facecolor('white')
    
    # Add labels in each bar
    for i, v in enumerate(list(col_nan_df.Number_of_NaN)):
        ax.text(v + 0.4, i + .25, str(v))
    
    pos = pylab.arange(len(col_nan_df))+.5     
    plt.barh(pos,list(col_nan_df.Number_of_NaN), align='center')
    plt.yticks(pos, list(col_nan_df.Feature))
    plt.xlabel('Number of Missing Values')
    plt.title("Missing Values per Feature", fontweight='bold')    
    plt.savefig('plot_na', bbox_inches='tight')    
    plt.show()

plot_NA(col_nan_df)

# If drop all nan's 6287 movies
len(movies_data_df_final_na.dropna())



col_nan_df = pd.DataFrame({'Feature':['Budget','Keywords','Domestic_Box_Office','Country','Release_Date','Year','Genre','Duration','Poster_Num_Faces','Num_Trailers','Age_Restrictions','Avg_Studios_BoxOffice','Actors_Experience_Scores','Directors_Experience_Scores','Actors_Awards','Directors_Awards','Avg_Ticket_Price_USD'], 
                           'Number_of_NaN':[5493,648,0,18,0,0,0,113,413,4353,2386,3175,520,0,715,225,0]})
for col in col_nan_df.Feature:
    col_nan_df.loc[col_nan_df.Feature == col, 'Number_of_NaN'] = movies_data_df_final_na[col].isnull().sum()

print(col_nan_df)





"""
   ############################################################################
   ############################# Fill NA Values ###############################
   ############################################################################
"""


"""
    Fill Age Restrictions NA's -------------------------------------------------------
"""

movies_data_df_final_fillna = movies_data_df_final.drop(['NC-17', 'PG', 'PG-13', 'R'], axis = 1).reset_index(drop = True)


#drop the target variable (age restrictions) from original data
X = movies_data_df_final.drop(['Domestic_Box_Office', 'Budget','AvgStudiosBoxOffice', 'NC-17', 'PG', 'PG-13', 'R','Movie_Name','Country','Genre','Release_Date','Duration','Keywords','Age_Restrictions'], 1).reset_index(drop = True)
y = movies_data_df_final['Age_Restrictions'].reset_index(drop = True)


# Mean Imputation of Predictors
imp = Imputer() 
imputed_X = pd.DataFrame(imp.fit_transform(X))
imputed_X.columns = X.columns
imputed_X.index = X.index
X = imputed_X


# Train Set (non NA's in Age Restrictions)
X_train = X.iloc[np.where(y.notnull())[0].tolist(),:]
y_train = y[np.where(y.notnull())[0].tolist()]

# Test Set
index_na_age = np.where(y.isnull())[0].tolist()
X_test = X.iloc[index_na_age,:]


# Scale
min_max_scaler = MinMaxScaler(feature_range=(0, 1))
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.transform(X_test)   


# Train RF
forest = RandomForestClassifier(max_features='sqrt')

# Grid Search parameters of RF
parameter_grid3 = {
                 'max_depth' : [5,7,9,11,13,15,17, 20, 22, 24, 26],
                 'n_estimators': [50,80,110,170,200,230,260,290,320,350],
                 'criterion': ['gini','entropy'],
                 }
# Identify best parameters of RF with 10Fold CV And Train RF Classifier with those
cross_validation = StratifiedKFold(y_train, n_folds=10)

grid_search3 = GridSearchCV(forest,
                           param_grid=parameter_grid3,
                           cv=cross_validation)
grid_search3.fit(X_train, y_train)

print('Best score: {}' .format(grid_search3.best_score_))
print('Best parameters: {}' .format(grid_search3.best_params_))
#Best score: 0.6088059247235467
#Best parameters: {'criterion': 'gini', 'max_depth': 26, 'n_estimators': 290}

grid_search3 = GridSearchCV(forest,param_grid={
                 'max_depth' : [26],
                 'n_estimators': [290],
                 'criterion': ['gini'],
                 },
                 cv=cross_validation)
grid_search3.fit(X_train, y_train)


# Predict the NA's of Budget
predicted_age = grid_search3.predict(X_test)

# Match Predictions with NA's at initial dataset
movies_data_df_final_fillna.loc[index_na_age,'Age_Restrictions'] = predicted_age.tolist()

# Create new Dummies for Age Restrictions
age_rest_dummies_fillna = pd.get_dummies(movies_data_df_final_fillna.Age_Restrictions, drop_first = True)
# Join with initial dataset
movies_data_df_final_fillna = movies_data_df_final_fillna.join(age_rest_dummies_fillna)




"""
    Fill Budget NA's ------------------------------------------------------------
"""

data = movies_data_df_final_fillna.drop(['Domestic_Box_Office','AvgStudiosBoxOffice','Movie_Name','Country','Genre','Release_Date','Duration','Keywords','Age_Restrictions'], axis = 1)

## Remove outliers

data = data[data.Budget != 1700000000].reset_index(drop = True)


## Remove all rows that have missing values for budget
data_to_remove = []
for i in range(data.shape[0]):
    if math.isnan(data.loc[i, 'Budget']): 
        data_to_remove.append(i)
data2 = data.drop(data.index[data_to_remove])

## Create ranges of budgets
# Histogram for budget made by us
bins = [0 , 5000000, 15000000, 35000000, 80000000, 100000000, 500000000]
plt.hist(np.array(data2[['Budget']]), bins = bins)  
plt.title("Histogram")
plt.show()

## Fil missing values with averages (except for budget)
data2 = data2.fillna(data2.mean())

## Split into train and test set
min_max_scaler = MinMaxScaler(feature_range=(0, 1))

train = data2[0: int((len(data2)*80/100))]
y_train = train.pop('Budget')
train_norm = pd.DataFrame(min_max_scaler.fit_transform(train), index=train.index, columns=train.columns)
test = data2[int((len(data2)*80/100)): (len(data2))]
y_test = test.pop('Budget')
test_norm = pd.DataFrame(min_max_scaler.transform(test), index=test.index, columns=test.columns)
train = train.reset_index()
test = test.reset_index()
train_norm = train_norm.reset_index()
test_norm = test_norm.reset_index()
y_train = y_train.reset_index()
y_test = y_test.reset_index()

# Fill nan in test set with 0
test_norm = test_norm.fillna(0)




## Define and include in our training set the bins
for i in range(len(train_norm)):
    if y_train.loc[i, 'Budget'] <= 5000000:
        train_norm.loc[i, 'Bin'] = 1
    elif y_train.loc[i, 'Budget'] <= 15000000:
        train_norm.loc[i, 'Bin'] = 2
    elif y_train.loc[i, 'Budget'] <= 35000000:
        train_norm.loc[i, 'Bin'] = 3
    elif y_train.loc[i, 'Budget'] <= 80000000:
        train_norm.loc[i, 'Bin'] = 4
    elif y_train.loc[i, 'Budget'] <= 100000000:
        train_norm.loc[i, 'Bin'] = 5
    else:
        train_norm.loc[i, 'Bin'] = 6
## Define and include in our test set the bins
for i in range(len(test_norm)):
    if y_test.loc[i, 'Budget'] <= 5000000:
        test_norm.loc[i, 'Bin'] = 1
    elif y_test.loc[i, 'Budget'] <= 15000000:
        test_norm.loc[i, 'Bin'] = 2
    elif y_test.loc[i, 'Budget'] <= 35000000:
        test_norm.loc[i, 'Bin'] = 3
    elif y_test.loc[i, 'Budget'] <= 80000000:
        test_norm.loc[i, 'Bin'] = 4
    elif y_test.loc[i, 'Budget'] <= 100000000:
        test_norm.loc[i, 'Bin'] = 5
    else:
        test_norm.loc[i, 'Bin'] = 6
                     
            
## Random Forest 
train_norm_new = train_norm
del train_norm_new['index']
y_train_new = train_norm_new.pop('Bin')
test_norm_new = test_norm
del test_norm_new['index']
y_test_new = test_norm_new.pop('Bin')
# Create a random forest classifier
model = RandomForestClassifier(random_state = 0)
#tune for hyper parameters 'n_estimators' & 'min_samples_leaf'
param_grid = { 
    'n_estimators': [1, 5, 10, 20, 50, 60, 70, 80, 90, 100, 150, 200],
    'min_samples_leaf': [1, 2, 3, 5]
}
grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = 5) 
grid.fit(train_norm_new, y_train_new) # takes some time to run

optimal_trees = grid.best_estimator_.n_estimators #defines the optimal number of trees 
optimal_leaf = grid.best_estimator_.min_samples_leaf #defines the optimal number of leaves

clf = RandomForestClassifier(n_estimators = optimal_trees, min_samples_leaf = optimal_leaf, random_state = 0)
clf.fit(train_norm_new, y_train_new)           
predvalue = clf.predict(test_norm_new)
difference = predvalue - y_test_new 
count = 0
count1 = 0
for i in range(len(difference)):
    if difference[i] != 0:
        count += 1
        if y_test_new[i] == 1 or y_test_new[i] == 2 or y_test_new[i] == 3:
            count1 += 1 # the 480 out of 670 missmatches are found within the first 3 ranges
clas_error = count/len(difference) # approximately 49%



# Fill in the missing values of budget by the average values of their bin    
data_without_budget = data
budget = data.pop('Budget')
data_without_budget = data_without_budget.fillna(data_without_budget.mean()) 

# New train set (train and test previously)
train_norm_new = pd.concat([train_norm_new, test_norm_new],axis=0, join='outer')
y_train_new = pd.concat([y_train_new, y_test_new])
y_train = pd.concat([y_train, y_test])

# Train RF in all data
clf = RandomForestClassifier(n_estimators = 200, min_samples_leaf = 2, random_state = 0)
clf.fit(train_norm_new, y_train_new)           
# Predict Bins
predvalue2 = clf.predict(data_without_budget.iloc[data_to_remove,:])

# Mergw predicted Bins of each movie with Budget NA
final_table = train_norm_new.join(y_train_new, how = 'inner')
final_table = final_table.join(y_train, how = 'inner')
mean_values = (final_table.groupby(['Bin'], as_index=False).mean()).loc[ : ,'Budget']

# Fill NA's at initial dataset with avg Budget of the movie in the predicted bin from RF
j = 0
for i in range(movies_data_df_final_fillna.shape[0]):
    if math.isnan(movies_data_df_final_fillna.loc[i, 'Budget']):
        cluster = predvalue2[j] - 1
        movies_data_df_final_fillna.loc[i, 'Budget'] = mean_values[cluster]
        j += 1





"""
    Fill AvgProduction Studio Budget NA's -------------------------------------------------------
"""

data = movies_data_df_final_fillna.drop(['Domestic_Box_Office','Movie_Name','Country','Genre','Release_Date','Duration','Keywords','Age_Restrictions'], axis = 1)

## Remove all rows that have missing values for budget
data_to_remove = []
for i in range(data.shape[0]):
    if math.isnan(data.loc[i, 'AvgStudiosBoxOffice']): 
        data_to_remove.append(i)
data2 = data.drop(data.index[data_to_remove])
data2 = data2.reset_index(drop = True)

## Create ranges of budgets
# Histogram for budget made by us
bins = [0 , 5000000, 15000000, 35000000, 80000000, 100000000, 400000000]

## Fil missing values with averages (except for budget)
data2 = data2.fillna(data2.mean())

## Split into train and test set
min_max_scaler = MinMaxScaler(feature_range=(0, 1))
train = data2[0: int((len(data2)*80/100))]
y_train = train.pop('AvgStudiosBoxOffice')
train_norm = pd.DataFrame(min_max_scaler.fit_transform(train), index=train.index, columns=train.columns)
test = data2[int((len(data2)*80/100)): (len(data2))]
y_test = test.pop('AvgStudiosBoxOffice')
test_norm = pd.DataFrame(min_max_scaler.transform(test), index=test.index, columns=test.columns)
test_norm = test_norm.reset_index(drop = True)
y_test = y_test.reset_index(drop = True)
# Fill nan in test set with 0
test_norm = test_norm.fillna(0)




## Define and include in our training set the bins
for i in range(len(train_norm)):
    if y_train[i] <= 5000000:
        train_norm.loc[i, 'Bin'] = 1
    elif y_train[i] <= 15000000:
        train_norm.loc[i, 'Bin'] = 2
    elif y_train[i] <= 35000000:
        train_norm.loc[i, 'Bin'] = 3
    elif y_train[i] <= 80000000:
        train_norm.loc[i, 'Bin'] = 4
    elif y_train[i] <= 100000000:
        train_norm.loc[i, 'Bin'] = 5
    else:
        train_norm.loc[i, 'Bin'] = 6
## Define and include in our test set the bins
for i in range(len(test_norm)):
    if y_test[i] <= 5000000:
        test_norm.loc[i, 'Bin'] = 1
    elif y_test[i] <= 15000000:
        test_norm.loc[i, 'Bin'] = 2
    elif y_test[i] <= 35000000:
        test_norm.loc[i, 'Bin'] = 3
    elif y_test[i] <= 80000000:
        test_norm.loc[i, 'Bin'] = 4
    elif y_test[i] <= 100000000:
        test_norm.loc[i, 'Bin'] = 5
    else:
        test_norm.loc[i, 'Bin'] = 6
                     
            
## Random Forest 
train_norm_new = train_norm
y_train_new = train_norm_new.pop('Bin')
test_norm_new = test_norm
y_test_new = test_norm_new.pop('Bin')
# Create a random forest classifier
model = RandomForestClassifier(random_state = 0)
# Tune FR hyper parameters 'n_estimators' & 'min_samples_leaf'
param_grid = { 
    'n_estimators': [1, 5, 10, 20, 50, 60, 70, 80, 90, 100, 150, 200],
    'min_samples_leaf': [1, 2, 3, 5]
}
grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = 5) 
grid.fit(train_norm_new, y_train_new) 

optimal_trees = grid.best_estimator_.n_estimators    # optimal number of trees 
optimal_leaf = grid.best_estimator_.min_samples_leaf # optimal number of leaves
clf = RandomForestClassifier(n_estimators = optimal_trees, min_samples_leaf = optimal_leaf, random_state = 0)
clf.fit(train_norm_new, y_train_new)           
predvalue = clf.predict(test_norm_new)
difference = predvalue - y_test_new 
count = 0
count1 = 0
for i in range(len(difference)):
    if difference[i] != 0:
        count += 1
        if y_test_new[i] == 1 or y_test_new[i] == 2 or y_test_new[i] == 3:
            count1 += 1 
clas_error = count/len(difference) 

# Fill in the missing values of budget by the average values of their bin    
data_without_avgstudioboxoffice = data
avg_studio_box_office = data.pop('AvgStudiosBoxOffice')
data_without_avgstudioboxoffice = data_without_avgstudioboxoffice.fillna(data_without_avgstudioboxoffice.mean()) 

# New train set (train and test previously)
train_norm_new = pd.concat([train_norm_new, test_norm_new],axis=0, join='outer')
y_train_new = pd.concat([y_train_new, y_test_new])
y_train = pd.concat([y_train, y_test]) 


# Train RF in all data
clf = RandomForestClassifier(n_estimators = 200, min_samples_leaf = 2, random_state = 0)
clf.fit(train_norm_new, y_train_new)           
# Predict Bins
predvalue2 = clf.predict(data_without_avgstudioboxoffice.iloc[data_to_remove,:])


final_table = train_norm_new.join(y_train_new, how = 'inner')
final_table = final_table.join(y_train, how = 'inner')
mean_values = (final_table.groupby(['Bin'], as_index=False).mean()).loc[ : ,'AvgStudiosBoxOffice']

# Fill NA's at initial dataset with avg StudiosBoxOffice of the movies in the predicted bin from RF
j = 0
for i in range(movies_data_df_final_fillna.shape[0]):
    if math.isnan(movies_data_df_final_fillna.loc[i, 'AvgStudiosBoxOffice']):
        cluster = predvalue2[j] - 1
        movies_data_df_final_fillna.loc[i, 'AvgStudiosBoxOffice'] = mean_values[cluster]
        j += 1



"""
    Fill NA's of other features  with Mean Values --------------------------------
"""
# Fill NA's of Poster_Num_Faces (310 movies) with average value
movies_data_df_final_fillna.Poster_Num_Faces = movies_data_df_final_fillna.Poster_Num_Faces.fillna(round(np.mean(movies_data_df_final_fillna.Poster_Num_Faces),0))

# Fill NA's of Duration (46 movie with average value
movies_data_df_final_fillna.Duration = movies_data_df_final_fillna.Duration.fillna(round(np.mean(movies_data_df_final_fillna.Duration),0))
# Update dummies
movies_data_df_final_fillna['duration_short'] = [1 if x <=80 else 0 for x in movies_data_df_final_fillna['Duration']]
movies_data_df_final_fillna['duration_medium'] = [1 if((x >80) & (x<=130)) else 0 for x in movies_data_df_final_fillna['Duration']]

# Fill NA's of All other features (actors scores and directors) with Avg Values (582 movies)
features_na = ['ActorScore','DirectorScore']
movies_data_df_final_fillna.ActorScore = movies_data_df_final_fillna['ActorScore'].fillna(np.mean(movies_data_df_final_fillna['ActorScore']))
movies_data_df_final_fillna.DirectorScore = movies_data_df_final_fillna['DirectorScore'].fillna(np.mean(movies_data_df_final_fillna['DirectorScore']))

"""
movies_data_df_final_fillna.NOscarWins = movies_data_df_final_fillna['NOscarWins'].fillna(np.mean(movies_data_df_final_fillna['NOscarWins']))
movies_data_df_final_fillna.NOscarNominations = movies_data_df_final_fillna['NOscarNominations'].fillna(np.mean(movies_data_df_final_fillna['NOscarNominations']))
movies_data_df_final_fillna.NGoldenWins = movies_data_df_final_fillna['NGoldenWins'].fillna(np.mean(movies_data_df_final_fillna['NGoldenWins']))
movies_data_df_final_fillna.NWins = movies_data_df_final_fillna['NWins'].fillna(np.mean(movies_data_df_final_fillna['NWins']))
movies_data_df_final_fillna.NPlaces = movies_data_df_final_fillna['NPlaces'].fillna(np.mean(movies_data_df_final_fillna['NPlaces']))
movies_data_df_final_fillna.NNominations = movies_data_df_final_fillna['NNominations'].fillna(np.mean(movies_data_df_final_fillna['NNominations']))
movies_data_df_final_fillna.DirectorNOscarWins = movies_data_df_final_fillna['DirectorNOscarWins'].fillna(np.mean(movies_data_df_final_fillna['DirectorNOscarWins']))
movies_data_df_final_fillna.DirectorNOscarNominations = movies_data_df_final_fillna['DirectorNOscarNominations'].fillna(np.mean(movies_data_df_final_fillna['DirectorNOscarNominations']))
movies_data_df_final_fillna.DirectorNGoldenWins = movies_data_df_final_fillna['DirectorNGoldenWins'].fillna(np.mean(movies_data_df_final_fillna['DirectorNGoldenWins']))
movies_data_df_final_fillna.DirectorNGoldenNominations = movies_data_df_final_fillna['DirectorNGoldenNominations'].fillna(np.mean(movies_data_df_final_fillna['DirectorNGoldenNominations']))
movies_data_df_final_fillna.DirectorNWins = movies_data_df_final_fillna['DirectorNWins'].fillna(np.mean(movies_data_df_final_fillna['DirectorNWins']))
movies_data_df_final_fillna.DirectorNPlaces = movies_data_df_final_fillna['DirectorNPlaces'].fillna(np.mean(movies_data_df_final_fillna['DirectorNPlaces']))
movies_data_df_final_fillna.DirectorNNominations = movies_data_df_final_fillna['DirectorNNominations'].fillna(np.mean(movies_data_df_final_fillna['DirectorNNominations']))
movies_data_df_final_fillna.Num_Trailers = movies_data_df_final_fillna['Num_Trailers'].fillna(np.mean(movies_data_df_final_fillna['Num_Trailers']))

"""

# Drop Rest NA's (469 movies)
movies_data_df_final_fillna = movies_data_df_final_fillna.dropna()



"""
   ############################################################################
   ############################# END Fill NA  #################################
   ############################################################################
"""









"""
   ############################################################################
   ############################ Final Datasets ################################
   ############################################################################
"""

# Export json of final df with NA's ------------------------------------------------
col_names = list(movies_data_df_final.columns)
col_names.remove('Movie_Name')
movies_data_df_final.Release_Date = ['N/A' if x == 'N/A' else x.strftime('%Y-%m-%d') for x in movies_data_df_final['Release_Date']] 
# Convert dataframe to dictionary 
movies_data_dic_final_preproc = movies_data_df_final.set_index('Movie_Name')[col_names].T.to_dict()
# Save dic to json file
with open('final_data/movies_data_df_final_preproc.json', 'w') as output:
    json.dump(movies_data_dic_final_preproc, output, sort_keys=False, indent=4)



# Export json of final df with filled NA's ------------------------------------------
col_names = list(movies_data_df_final_fillna.columns)
col_names.remove('Movie_Name')
movies_data_df_final_fillna.Release_Date = ['N/A' if x == 'N/A' else x.strftime('%Y-%m-%d') for x in movies_data_df_final_fillna['Release_Date']] 
# Convert dataframe to dictionary 
movies_data_dic_final_preproc = movies_data_df_final_fillna.set_index('Movie_Name')[col_names].T.to_dict()
# Save dic to json file
with open('final_data/movies_data_df_final_fillna_preproc_Disagreg.json', 'w') as output:
    json.dump(movies_data_dic_final_preproc, output, sort_keys=False, indent=4)



"""
   ############################################################################
   ######################### Final Dataset Modelling ##########################
   ############################################################################
"""

# Delete reference columns ----------------------------------------------
movies_data_df_final_modelling = movies_data_df_final.drop(['Movie_Name','Country','Genre','Release_Date','Duration','Keywords','Age_Restrictions'], axis = 1)
movies_data_df_final_modelling_fillna = movies_data_df_final_fillna.drop(['Movie_Name','Country','Genre','Release_Date','Duration','Keywords','Age_Restrictions'], axis = 1)
# Save df  -------------------------------------------------------------- 
movies_data_df_final_modelling.to_csv('final_data/movies_data_df_final_modelling.csv', index = False)
movies_data_df_final_modelling_fillna.to_csv('final_data/movies_data_df_fina_modellingl_fillna.csv', index = False)


"""
 Example dataset with no filled NA's (dropped NA's) ------------------------------------
"""
movies_data_df_final_dropna = movies_data_df_final_modelling.dropna()

# Save df  ---------------------------------------------------------------------
movies_data_df_final_dropna.to_csv('final_data/movies_data_df_final_modelling.csv', index = False)



"""
 Example dataset Filled NA's--------------------------------------------------------------
"""
movies_data_df_final_modelling_fillna = movies_data_df_final_fillna.drop(['Movie_Name','Country','Genre','Release_Date','Duration','Keywords','Age_Restrictions'], axis = 1)

movies_data_df_final_fillna_ex = movies_data_df_final_modelling_fillna.dropna()

# Save df  ---------------------------------------------------------------------
movies_data_df_final_fillna_ex.to_csv('final_data/movies_data_df_final_fillna_modelling.csv', index = False)




