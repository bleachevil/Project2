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

We calculated the correlation in terms of Ethereum closing prices, below are the results and these were also visualized on a heatmap.

```python

plt.figure(figsize=(16, 6))

# Set the range of values to be displayed on the colormap from -1 to 1, and set the annotation to True to display the correlation values on the heatmap.
heatmap = sns.heatmap(eth_df.corr(), vmin=-1, vmax=1, annot=True, fmt='.4f')

# Give a title to the heatmap. Pad defines the distance of the title from the top of the heatmap.
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);

````

<img width="253" alt="Screen Shot 2022-06-16 at 8 29 52 PM" src="https://user-images.githubusercontent.com/99091066/174199307-ca22c1ea-67fb-4043-a0b4-b02e2ba86ea0.png">

<img width="851" alt="Screen Shot 2022-06-16 at 8 30 27 PM" src="https://user-images.githubusercontent.com/99091066/174199355-5e53724d-8740-4bc4-8d28-b83b2eac8d61.png">

We noticed many of the correlation scores on the heatmap were 1, we changed the rounding to 4 decimals in hopes it would make a difference. High, open, and low prices are all highly correlatd since Ethereum is priced 24 hours. The price at the end of the day near midnight (close) would be almost the same as the price at midnight (open). Due to the high correlation, we decided to create a dataframe only using `close`, `vwap`, `ethereum market cap`, `number of tweets` and `trade counts`. 
















