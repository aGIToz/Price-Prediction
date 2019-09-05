import datetime
# number of day to predict
days = 18

# choose the company 
company = "AAPL"
""" 
"AAPL", "GE", "GOOG", "IBM", "MSFT" 
"""

# choose the end and start date for training
start = datetime.datetime(2011,1,1)
end = datetime.datetime(2019,9,4)
"""
select the end date as yesterday's date as you shall
be predicting for N number of days starting form today 
"""
