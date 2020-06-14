import pandas as pd
import datetime 
from datetime import datetime as dt
import numpy as np
import os 
from pandas import DataFrame
from pandas import concat


def read_file_with_date(date_in , function = 1):
    # dataset/beijing_20160101-20161231/beijing_extra_2016
    year = date_in.strftime('%Y')
    date = date_in.strftime('%Y%m%d')
    if function == 2:
        char = 'extra_'
    elif function == 1:
        char = 'all_'
    else:
        char = 'processed_'
    try:
        if date_in.year == 2020: 
            return pd.read_csv('dataset/beijing_' + year + '0101-'  + year + '0509/beijing_' + char + date + '.csv')
        else:
            return pd.read_csv('dataset/beijing_' + year + '0101-'  + year + '1231/beijing_' + char + date + '.csv')
    except:
        return pd.DataFrame()


def insert_row(i,df):
    line = pd.DataFrame({"hour": i}, index=[i])
    if i == 0:
        return pd.concat([line, df.iloc[i:]], sort=True).reset_index(drop=True)
    else:
        head = df.iloc[0:i]
        leg = df.iloc[i:]
        return pd.concat([head, line, leg], sort=True).reset_index(drop=True)

def append_row(length,df):
    df1 = pd.DataFrame({"hour": length}, index=[length])
    return pd.concat([df,df1], axis=0, ignore_index=True, sort=True).reset_index(drop=True)


############### for your conveninence ############### 

start_y = 2014
start_m = 1
start_d = 1
end_y = 2014
end_m = 12
end_d = 31

# full data has 888
empty_threshold = 800

#####################################################


d_begin = datetime.datetime(start_y, start_m, start_d , 0,0,0)
d_end = datetime.datetime(end_y,end_m,end_d)
d_delta = datetime.timedelta(days=1)
d = d_begin
df = pd.DataFrame()

noDate_f = os.open("nodate.txt", os.O_WRONLY|os.O_CREAT) 

print("Trial only run from 1/1/2014 to 31/12/2014...")
print("Reading data... Please wait")
while d <= d_end:
    # dataset2 = read_file_with_date(d, extra = True)

    
    df1 = read_file_with_date(d)

    # if data is empty or false
    if (df1.empty) or (not 'date' in df1.columns):
        line = str.encode(d.strftime('%Y%m%d')+'\n') 
        os.write(noDate_f,line)

        # #reverse input the previous day
        # df1 = df1temp.iloc[::-1]
        # df = df.append(df1)


        # renew date time
        # dateTime = [(d + datetime.timedelta(hours=i)).strftime("%Y:%m:%d, %H:%M:%S") for i in range(24)]
        # if not 'date' in df1.columns: #if the dataframe is false
        #     df1 = pd.DataFrame()
        # df1['date'] = dateTime
        d += d_delta
        # add to main file
        # df = df.append(df1)
        continue

    # remove extra rows
    df1 = df1.set_index('type')
    df1 = df1.drop(['PM2.5_24h','PM10','PM10_24h','AQI'], axis=0)
    

    # fill in missing rows
    while len(df1.index) < 24:
        for i in range(int(df1.iloc[-1, 1])):
            value = df1.iloc[i, 1]
            if i != value:
                df1 = insert_row(i,df1)
                # break
                
        if len(df1.index) < 24:
            df1 = append_row(len(df1.index),df1)
    

    # renew date time
    dateTime = [(d + datetime.timedelta(hours=i)).strftime("%Y:%m:%d, %H:%M:%S") for i in range(24)]
    df1['date'] = dateTime

    # renew hour column
    df1['hour'] = [i for i in range(24)]

    # add month column
    month_arr = [d.month for i in range(24)]
    df1.insert(0, "month", month_arr, allow_duplicates=False) 

    # remove extra columns
    # df1 = df1.drop(columns=['hour'],axis = 1)

    d += d_delta

    # if too much empty cells, don't use
    if df1.isnull().sum().sum() > empty_threshold:
        continue

    # remove empty rows
    df1.dropna(axis=0, thresh=34, subset=None, inplace=True)

    # df1temp = df1

    # add to main file
    df = df.append(df1)

    
# last touch up!
# df = df.reset_index(drop=True)
df.set_index('date', inplace = True)

# # swith place of hour and month

columns_titles = list(df.columns)
columns_titles[0] = 'month'
columns_titles[1] = 'hour'
df=df.reindex(columns=columns_titles)

def fill_in_empty_cells(df):
    # find location of empty cells
    temp = df.drop(columns=['month', 'hour'],inplace = False)
    summm_ = temp.sum(axis=1) 
    count_ = temp.count(axis=1) 

    for i,j in zip(*np.where(pd.isnull(df))):
        # sum up all the non empty cells
        df.iloc[i,j] = summm_.iloc[i] / count_.iloc[i]
    
    return df

def fill_in_zero(df):
    # find location of empty cells

    for i,j in zip(*np.where(pd.isnull(df))):
        # fill in zero
        df.iloc[i,j] = 0
    
    return df

df = fill_in_empty_cells(df)


print("Read complete, file wrote to dataset/full_data.csv")

print("Processing data to supervised sequence data")
# write to main file
df.to_csv('dataset/full_data.csv', header = True)

# end
os.close(noDate_f) 

############################## series to supervised ############################## 

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	"""
    Ref to https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/

	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


############### for your conveninence ############### 

X_length = 10
Y_length = 6

#####################################################

# df = pd.read_csv('dataset/full_data.csv')
# remove extra columns
# 
df.drop(columns=['month', 'hour'],inplace = True)

processedData = series_to_supervised(df,n_in=X_length, n_out=Y_length)

processedData = processedData.reset_index(drop=True)
# write file
processedData.to_csv('dataset/processed_data.csv', header = True)

print("Processing complete")
