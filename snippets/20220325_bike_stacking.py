import os
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

# source: https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset
data = pd.read_csv(os.environ['PATH_TO_DATA']+'bike_day.csv')
data['dteday'] =pd.to_datetime(data['dteday'])
data = data.sort_values(by='dteday',ascending=True,ignore_index=True)

tss = TimeSeriesSplit(n_splits=3)

for j,(train_index,test_index) in enumerate(tss.split(data)):
    print(f'--- Split {j+1}')
    train_start = data.loc[train_index,'dteday'].min().strftime('%Y-%m-%d')
    train_end = data.loc[train_index,'dteday'].max().strftime('%Y-%m-%d')
    test_start = data.loc[test_index,'dteday'].min().strftime('%Y-%m-%d')
    test_end = data.loc[test_index,'dteday'].max().strftime('%Y-%m-%d')
    print('Train - 'f'start: {train_start} ',f'end: {train_end} ',f'size: {len(train_index)}')
    print('Test - ',f'start: {test_start} ',f'end: {test_end} ',f'size: {len(test_index)}')

# --- Split 1
# Train - start: 2011-01-01  end: 2011-07-04  size: 185
# Test -  start: 2011-07-05  end: 2012-01-02  size: 182
# --- Split 2
# Train - start: 2011-01-01  end: 2012-01-02  size: 367
# Test -  start: 2012-01-03  end: 2012-07-02  size: 182
# --- Split 3
# Train - start: 2011-01-01  end: 2012-07-02  size: 549
# Test -  start: 2012-07-03  end: 2012-12-31  size: 182