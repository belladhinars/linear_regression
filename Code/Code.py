#Importing library
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from yellowbrick.cluster import KElbowVisualizer

import statsmodels.api as sm

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt
from statsmodels.tsa.arima.model import ARIMA
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller

import warnings
warnings.filterwarnings ('ignore')

#Membaca 5 data csv dan juga memisahkan dat yang ada
df_customer = pd.read_csv("Customer.csv", delimiter=';')
df_legend = pd.read_csv("Legend.csv", delimiter=';')
df_product = pd.read_csv("Product.csv", delimiter=';')
df_store = pd.read_csv("Store.csv", delimiter=';')
df_transaction = pd.read_csv("Transaction.csv", delimiter=';')

#Data Cleaning
def check_and_clean_data(df_customer):
  return df_customer
def check_and_clean_data(df_Product):
  return df_product
def check_and_clean_data(df_store):
  return df_store
def check_and_clean_data(df_transaction):
  return df_transaction
  
df_customer = check_and_clean_data(df_customer)
df_product = check_and_clean_data(df_product)
df_store = check_and_clean_data(df_store)
df_transaction = check_and_clean_data(df_transaction)

#Mengubah format tanggal agar memudahkan dalam pembacaan data tanggal
df_transaction['Date'] = pd.to_datetime(df_transaction['Date'])

#Group by 'TransactionID' and select the row with the maximum 'Date'
df_transaction = df_transaction.sort_values(by='Date', ascending=False) \
  .groupby('TransactionID', as_index=False).first()

#Merge data
df_merge = pd.merge(df_transaction, df_customer, on='CustomerID', how='inner')
df_merge = pd.merge(df_merge, df_product.drop(columns=('Price')), on='ProductID', how='inner')
df_merge = pd.merge(df_merge, df_store, on='StoreID', how="inner")

df_regresi = df_merge.groupbya(['Date']).agg({
    'Qty' = 'sum'
}).eset_index()

#Melihat plot seasonal
decomposed = seasonal_decompose(df_regresi.set_index('Date'))
plt.figure(figsize=(8, 8))
plt.subplot(311)
decomposed.trend.plot(ax=plt.gca())
plt.title('Trend')
plt.subplot(312)
decomposed.seasonal.plot(ax=plt.gca())
plt.title('Seasonality')
plt.subplot(313)
decomposed.resid.plot(ax=plt.gca())
plt.title('Residuals')
plt.tight_layout()

cut_off = round(df_regresi.shape[0]*0.8)
df_train = df_regresi[:cut_off]
df_test = df_regresi[cut_off: ].reset_index(drop=True)
df_train.shape, df_train.shape

#Membuat plot grafik regresi
plt.figure(figsize=(12,5))
sns.lineplot(data=df_train, x=df_train.index, y=df_train['Qty'])
sns.lineplot(data=df_test, x=df_test.index, y=df_test['Qty'])
plt. show()
                              
