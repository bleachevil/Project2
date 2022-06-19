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
- Google Colab

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

We then plotted the sentiment scores with the Ethereum prices and analyzed the figures below. 

<img width="374" alt="Screen Shot 2022-06-18 at 9 59 49 PM" src="https://user-images.githubusercontent.com/99091066/174462768-69d103ec-8269-4635-b22a-190ca317bc15.png">


<img width="392" alt="image" src="https://user-images.githubusercontent.com/99091066/174462714-12ed3d45-af2d-477d-9b32-d23e5280f2c2.png">


The Y-axis shows the percentage change in ethereum price and sentiment scores of the tweets are on the X-axis. At each price percentge change, there were tweets with various sentiment scores. There is no definitive proof connecting twitter sentiments with prices minutes after. 

Our initial objective was to find a relationship between tweets and price changes. Since we did not get the results we hoped for, we altered the scope of our project. We decided to determine which features do have an impact on ethereum prices, and if we were able to build a model that could effectively predict ethereum prices. 


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

We noticed many of the correlation scores on the heatmap were 1, we changed the rounding to 4 decimals in hopes it would make a difference. High, open, and low prices are all highly correlatd since Ethereum is priced continuously. The price at the end of the day near midnight (close) would be almost the same as the price at midnight (open). Due to the high correlation, we decided to create a dataframe only using `close`, `vwap`, `ethereum market cap`, `number of tweets` and `trade counts`. 

The price trends were plotted below

<img width="863" alt="Screen Shot 2022-06-16 at 8 40 02 PM" src="https://user-images.githubusercontent.com/99091066/174200090-469d0143-74ed-4c8e-806d-3d831e28cbcb.png">

Before feeding the data into our machine, we had to ensure our data was standardized to ensure a more accurate prediction.
The distribution was also plotted. It is showing a right skew, which was adjusted for using a cube root transformation.

<img width="852" alt="Screen Shot 2022-06-18 at 10 08 00 PM" src="https://user-images.githubusercontent.com/99091066/174462944-c1a7afba-c292-4074-a45c-838884931e1e.png">

<img width="726" alt="Screen Shot 2022-06-18 at 10 08 21 PM" src="https://user-images.githubusercontent.com/99091066/174462952-257ceacd-8d10-4eb1-943f-cfcf720493b7.png">

This normalized the data

<img width="853" alt="Screen Shot 2022-06-18 at 10 09 47 PM" src="https://user-images.githubusercontent.com/99091066/174462976-c92c565b-4199-4e31-95c9-a23d5afb3637.png">

These values were added to `eth2_df`. We then added lags, rolling mean, and expanding means to the dataframe.

## DATA TESTING

We set X as `Date`, `vwap_cbrt`, `Ethereum Market Cap`, `Number_of_tweet`, `trade_count`, `close_cbrt`,`lag_1`, `lag_2`,`lag_3`, `lag_4`, `lag_5`, `lag_6`, `lag_7`, `rolling_3_mean`, `rolling_4_mean`, `rolling_5_mean`, `rolling_6_mean`, `rolling_7_mean`, `expanding_2_mean`, `expanding_3_mean`, `expanding_4_mean`.

We set Y as `Date` and `tomorrow_close_cbrt` and set the training & testing variables. 

```python

X_train = X.loc[(X.Date >= '2016-01-01') & (X.Date <= '2021-01-31')]
X_test = X.loc[X.Date >= '2021-02-01']
y_train = y.loc[(y.Date >= '2016-01-01') & (y.Date <= '2021-01-31')]
y_test = y.loc[y.Date >= '2021-02-01']
y_test_actual = y_test.copy()
y_test_actual.tomorrow_close_cbrt = y_test_actual.tomorrow_close_cbrt ** 3
y_test_actual


````
    > The data that will be plotted, `y_test_actual`, is cubed to convert the values back to the price.

We fit the data using Linear Regression, Random Forest, and XGBoost. For each model, we calculate the `MAPE`, `MAE`, and `RMSE`.

`MAPE` - Mean absolute percentage error. It measures accuracy of a forecast model as a percentage. The higher the number, the less accurate the results are. 

`MAE` - Mean absolute error. This score tells us the mean difference between the actual and predicted values. The lower the better.

`RMSE` - Root mean square deviation. This is the standard deviation of the prediction errors. Lower RMSE indicates a better model.

Functions:

````python 

def mape(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100

def mae(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    return np.mean(np.abs(y_true - predictions))

def rmse(actual, pred):
    return np.sqrt(mean_squared_error(actual, pred))

````

### Linear Regression 

The linear regression model scored a MAPE of 1.95, MAE of 0.213, and RMSE of 0.2609. Plotted on a chart, it is very close to the next day's price. 

<img width="396" alt="Screen Shot 2022-06-16 at 9 08 48 PM" src="https://user-images.githubusercontent.com/99091066/174202405-2f8b5340-06ad-4151-aa6e-313c7743c1f7.png">

<img width="428" alt="Screen Shot 2022-06-16 at 9 09 05 PM" src="https://user-images.githubusercontent.com/99091066/174202431-0924b463-bb34-402a-aaee-8fdb8191e397.png">


### Random Forest 

The random forest model scored a MAPE of 6.18 , MAE of 0.708, and RMSE of 0.928. Plotted on a chart, the predicted and actual values do not appear to be similar.


<img width="394" alt="Screen Shot 2022-06-16 at 9 10 52 PM" src="https://user-images.githubusercontent.com/99091066/174202600-fce77500-088c-4d5e-b56f-4dd3bef8b7d6.png">

<img width="429" alt="Screen Shot 2022-06-16 at 9 11 08 PM" src="https://user-images.githubusercontent.com/99091066/174202622-e920ffc2-661e-4401-a420-737f8c1dc624.png">

### XGBOOST

The XGBoost model scored a MAPE of 7.86, MAE of 0.89, and RMSE of 1.088. Plotted on a chart, the predicted and actual values do not appear to be similar.

<img width="388" alt="Screen Shot 2022-06-16 at 9 12 18 PM" src="https://user-images.githubusercontent.com/99091066/174202719-2ee177b4-3a82-4b3f-be6e-ab7e278b1a14.png">

<img width="426" alt="Screen Shot 2022-06-16 at 9 12 36 PM" src="https://user-images.githubusercontent.com/99091066/174202750-0b881a56-4e91-4952-b671-a532844fb6d3.png">



