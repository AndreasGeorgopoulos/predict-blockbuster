# predict-blockbuster
At this project a comprehensive model is developed in order to predict local box-office receipts of any movie screened in the U.S. prior to its official released date using open data sources.

### Data Acquisition
`jkhjhlkjlkjlk`
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
- Directors IDs and Names
- Production Studios IDs and Names
- Production Budget  
- Domestic USA Gross
- Plot keywords
- Number of trailers
- URL of corresponding movie poster at IMDb website
3. Having the URL of each movie's poster at IMDb website, an algorithm is generated in order to download all available posters and implement a facial recognition algorithm aimed at capturing the number of depicted faces at each poster. (`dsfsdfsdfsdfdsf` file)
4. Supplement the newly created dataset with additional features and correct for any missmatches during data acquisition**
### Feature Enineering
`as`
This data is then used to engineer multiple important movie features that are responsible for the variation in box-office receipts of different movies.

### Predictive Modelling


** Since the presented project was a team project, only the code that was developed by myself is presented. That is, code regarding actor and director additional features as well as twitter data is not presented. However, the concept behind the aforementioned features is explained in the detailed report and the final generated dataset is provided in the `jjkjkj` file.

*** To read full report [click here](http://www.andreasgeorgopoulos.com/predict-blockbuster/)
