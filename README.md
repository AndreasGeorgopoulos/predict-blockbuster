# Predict a blockbuster using open data sources
At this project a comprehensive model is developed in order to predict local box-office receipts of any movie screened in the U.S. prior to its official released date using open data sources.

### Data Acquisition
`scrap_movie_data.py`  
Most of the movie data is acquired using several Application Programming Interfaces as well as through screen scraping of two specialized open data sources: The International Movie Database and The Numbers. Some auxiliary information is also extracted from Box Office Mojo and Fxtop websites. The main code for the this part is presented in the `asdsadasda` file. 
1. Generate a list of movies with provided gross revenues in the U.S. from [the-numbers.com](http://www.the-numbers.com/movies/letter) website.
2. Having a list of movies, additional data is extracted by generating screen scraper algorithms for IMDb as well as using IMDb API's database. More specifically, for each movie the following data is acquired:
- URL at IMDb website
- Movie ID at IMDb website
- Genre
- Released Date
- Country
- Duration
- Leading Actors IDs and Names, List of all 
- Age restrictions
- Directors IDs and Names
- Production Studios IDs and Names
- Production Budget  
- Domestic USA Gross
- Plot keywords
- Number of trailers
- URL of corresponding movie poster at IMDb website
3. Having the URL of each movie's poster at IMDb website, an algorithm is generated in order to download all available posters and implement a facial recognition algorithm aimed at capturing the number of depicted faces at each poster. (`poster_face_recognition.py` file)
4. Supplement the newly created dataset with additional data and features and correct for any missmatches during data acquisition**
### Feature Enineering
`preprocessing.py`  
 At this part of the code, the aforementioned acquired dataset is preprocessed and different features are engineered in order to capture the variation in box-office receipts of different movies. Some important features are the following:
 - Thirty most and least important genres combinations 
 - Thirty most and least important country cominations
 - Most important keywords in movie plots through TF-IDF calculations and sentiment analysis (`word2vec` modelling)

### Predictive Modelling
`predictive_modelling_classification.py`, `predictive_modelling_regression.py`  
The prediction task is approached both as a classification, with four classes representing different box-office ranges, and as a regression problem. That is, an Ensemble Stacked Classification (ESCM) and Ensemble Stacked Regression (ESRM) two-level models are generated. The ESCM consists of an Artificial Neural Network, a Support Vector Machine and a Random Forest at its first level and a Logistic Regression at the second level. After predicted classes of all the movies are acquired, the ESCM performs Locality Sensitive Hashing (LHS) in order to provide box-office predictions. It predicts Box-Office Receipts of each movie based on the average value of its five approximate nearest neighbours within the corresponding predicted class from the training dataset, controlling for the curse of dimensionality. The ESRM, on the other hand, has an Artificial Neural Network and a Support Vector Regressor with a Gaussian kernel at the first level and another Artificial Neural Network at the second level.

At the provided code, individual regression and classification models are also present as well as additional techiques that are used to improve the predictive power of the models. For instance, in the classification part, a Synthetic Minority Oversampling Technique (SMOTE) is implemented due to the unbalanced data. 


** Since the presented project was a team project, only the code that was developed by myself is presented. That is, code regarding actor and director additional features as well as twitter data is not presented. However, the concept behind the aforementioned features is explained in the detailed report and the corresponding datasets are provided in the `additional_data.zip` file.

*** To read full report [click here](http://www.andreasgeorgopoulos.com/predict-blockbuster/)
