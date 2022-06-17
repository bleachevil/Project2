# PROJECT OVERVIEW

The initial goal of our project was to determine if twitter sentiments had an effect on Ethereum prices. As we analyzed our data and conducted a sentiment analysis on tweets from 2013-2021, we concluded there was no correlation between these tweets and Ethereum prices. 

We then set out to determine which features are corrrelated with Ethereum closing prices using various machine learning models. 

# THE TEAM

Amira Ali | Jeff Zhang | Nadeem Hassan

# PRESENTATION



# TOOLS & RESOURCES

## SOFTWARE & TECHNOLOGIES USED

- Alpaca API 
- CoinGecko
- Google Finance
- Kaggle [Data set](https://www.kaggle.com/datasets/fabioturazzi/cryptocurrency-tweets-with-sentiment-analysis)
- Jupyter Notebook
- Streamlit 

# DATA

For the sentiment portion of our project, we analyzed tweets from 2013 to 2021 and compared it against historial ethereum prices for the same timeframe. We decided to use pre-existing data from Kaggle rather than using Twitter's API since there was a lot more data offered this way. 

The Kaggle data came with its own sentiment scores and keyword column making it a lot easier to clean the data. First, we filtered for keyword `Ethereum` only kept the necessary columns, for example ` tweet`, `date`, `like count`, `retweet counts` etc. 

We cleaned the tweets and removed all punctuation to prepare us to apply NLP. We then created a function to get the sentiment score, subjecivity, and polarity. 

````python

# Create a function to get the sentiment text
def getSentiment(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'
        
# Create a function to get the subjectivity
def getSubjectivity(twt):
    return TextBlob(twt).sentiment.subjectivity

# Create a function to get the polarity
def getPolarity(twt):
    return TextBlob(twt).sentiment.polarity      
````

The scores were added as a column to the dataframe and ranked : Negative = 0, Neutral = 1, Positive = 2. 

<img width="874" alt="Screen Shot 2022-06-16 at 8 27 00 PM" src="https://user-images.githubusercontent.com/99091066/174199099-1bcb07a0-091e-4665-8247-42c73948356d.png">


We were hoping to analyze tweets published at certain times of the day compared to the ethereum price a few minutes after. To do this, we needed 1 minute interval data for Ethereum.

Using Alpaca's API, we generated the following code

```python

# Set tickers
ticker = ["ETHUSD"]

# Set timeframe to '1Minute'
timeframe = "1Min"

# Set start and end datetimes.
start_date = pd.Timestamp("2017-01-01", tz="America/New_York").isoformat()
end_date = pd.Timestamp("2017-06-30", tz="America/New_York").isoformat()

# Get 1 year's 
df_ticker = alpaca.get_crypto_bars(
    ticker,
    timeframe,
    start=start_date,
    end=end_date,
).df

# Display sample data
df_ticker.head(10)

````

We had to adjust our date range to account for available historical data for Ethereum. Our time frame was then adjusted to 2016-2021. 

Due to the volume of the data, we had to split the code into semi annual periods. Each of these were combined into one data frame `Total_Eth_Prices`, which we then used for further analysis.

# RESULTS 






