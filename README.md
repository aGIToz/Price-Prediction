# Price-Prediction
Stock price prediction using sklearn, it uses pandas web datareader to load the stock data of APPLE from 2011-1-1 to 2019-9-4. Then trains 5 models {simple linear regression, polynominal2 regression, polynomial3 regression, knn regression and bayesian regression} to predict the stock price for the next 18 days.

## Result
Apple stock prediction using the best (**polynomial3 regression** in this case). Orange curve shows the prediction for the next 18 days starting  from 2019-sep-5.
![prediction](./prediction.png)

## usage
See config.py, to choose a company {apple, microsoft, ge, ibm, google} and choose a date to predict and number of days to predict from that date.
```shell
$ python predict.py
```
## Requirements
python3 and see requirement.txt

