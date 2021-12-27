############################################
# Demand Forecasting
############################################
# Store Item Demand Forecasting Challenge
# Retail Demand forecasting Problem
# Hierarchical or non-hierarchical forecasting
# Store stock sales forecasts
# Data set--> 5-year daily store sales data
# 3-month forecast is requested
# 10 different stores and 50 different items

#####################################################
# Libraries
#####################################################

import time
import numpy as np
import pandas as pd
#!pip install lightgbm
import lightgbm as lgb
import warnings
from Helpers.eda import *
from Helpers.data_prep import *

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


#####################################################
# Loading the data
#####################################################

train = pd.read_csv('Datasets/demand_forecasting/train.csv', parse_dates=['date'])
test = pd.read_csv('Datasets/demand_forecasting/test.csv', parse_dates=['date'])
sample_sub = pd.read_csv('Datasets/demand_forecasting/sample_submission.csv')
df = pd.concat([train, test], sort=False)

#####################################################
# EDA
#####################################################

df["date"].min(), df["date"].max()
# (Timestamp('2013-01-01 00:00:00'), Timestamp('2018-03-31 00:00:00'))
check_df(train)
# (913000, 4)

check_df(test)
# (45000, 4)

check_df(sample_sub)
# (45000, 2)
# yalnızca id ve satış sayısı olacak artık

check_outlier(df, "sales")
# False
missing_values_table(df)
#        n_miss  ratio
# id     913000   95.3
# sales   45000    4.7

# Sales distribution?
df[["sales"]].describe().T
#           count       mean        std  min   25%   50%   75%    max
# sales  913000.0  52.250287  28.801144  0.0  30.0  47.0  70.0  231.0

# how many stores?
df[["store"]].nunique()
# store    10

# how many items?
df[["item"]].nunique()
# item    50

# number of unique items in each store?
df.groupby(["store"])["item"].nunique()

# sales for each store?
df.groupby(["store", "item"]).agg({"sales": ["sum"]})
# sales are different in each store

# store-item breakdown sales statistics
df.groupby(["store", "item"]).agg({"sales": ["sum", "mean", "median", "std"]})

#####################################################
# FEATURE ENGINEERING
#####################################################

#####################################################
# Date Features
#####################################################

df.head()
#         date  store  item  sales  id
# 0 2013-01-01      1     1   13.0 NaN
# 1 2013-01-02      1     1   11.0 NaN
# 2 2013-01-03      1     1   14.0 NaN
# 3 2013-01-04      1     1   13.0 NaN
# 4 2013-01-05      1     1   10.0 NaN

df.shape
# (958000, 5)

# what can be extracted from date feature:

def create_date_features(df):
    df['month'] = df.date.dt.month # capture seasonality more than trend
    df['day_of_month'] = df.date.dt.day
    df['day_of_year'] = df.date.dt.dayofyear
    df['week_of_year'] = df.date.dt.weekofyear # important
    df['day_of_week'] = df.date.dt.dayofweek + 1
    df['year'] = df.date.dt.year
    df["is_wknd"] = df.date.dt.weekday // 4
    df['is_month_start'] = df.date.dt.is_month_start.astype(int)
    df['is_month_end'] = df.date.dt.is_month_end.astype(int)
    return df


df = create_date_features(df)
df.head(20)


df.groupby(["store", "item", "month"]).agg({"sales":["sum","mean","median","std"], "month":"count"})
#                    sales                              month
#                       sum       mean median        std count
# store item month
# 1     1    1       2125.0  13.709677   13.0   4.397413   186
#            2       2063.0  14.631206   14.0   4.668146   169
# ...

#####################################################
# Random Noise
#####################################################
# random noise to prevent over-fitting

def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))
# scale = standard deviation of the distribution


#####################################################
# Lag/Shifted Features
#####################################################


lag1 = df["sales"].shift(1).values[0]
# we want to forecast 3 months forward that's why lag features are 3 months periods
# df = lag_features(df, [91,98,105,...546,728])


df.sort_values(by=['store', 'item', 'date'], axis=0, inplace=True)

check_df(df)
df["sales"].head(10)
df["sales"].shift(1).values[0:10]

pd.DataFrame({"sales": df["sales"].values[0:10],
              "lag1": df["sales"].shift(1).values[0:10],
              "lag2": df["sales"].shift(2).values[0:10],
              "lag3": df["sales"].shift(3).values[0:10],
              "lag4": df["sales"].shift(4).values[0:10]})


df.groupby(["store", "item"])['sales'].head()
# 0         13.0
# 1         11.0
# 2         14.0
# ...

# sales a bir shift uygulamak için:
df.groupby(["store", "item"])['sales'].transform(lambda x: x.shift(1))
# 0         NaN
# 1        13.0
# 2        11.0
# ...


def lag_features(dataframe, lags):
    dataframe = dataframe.copy()
    for lag in lags:
        dataframe['sales_lag_' + str(lag)] = dataframe.groupby(["store", "item"])['sales'].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe


df = lag_features(df, [91, 98, 105, 112, 119, 126, 182, 364, 546, 728])


df.head()

df[df["sales"].isnull()]


pd.to_datetime("2018-01-01") - pd.DateOffset(91)
# Timestamp('2017-10-02 00:00:00')

df[df["date"] == "2017-10-02"]
# 91 gün öncesindeki sales değeri ile eşleşmedi
# tarih farkını bulmamız lazım

#####################################################
# Rolling Mean Features
#####################################################


df["sales"].head(10)
df["sales"].rolling(window=2).mean().values[0:10]
df["sales"].rolling(window=3).mean().values[0:10]
df["sales"].rolling(window=5).mean().values[0:10]

pd.DataFrame({"sales": df["sales"].values[0:10],
              "roll2": df["sales"].rolling(window=2).mean().values[0:10],
              "roll3": df["sales"].rolling(window=3).mean().values[0:10],
              "roll5": df["sales"].rolling(window=5).mean().values[0:10]})

#    sales  roll2      roll3  roll5
# 0   13.0    NaN        NaN    NaN
# 1   11.0   12.0        NaN    NaN
# 2   14.0   12.5  12.666667    NaN
# ...


pd.DataFrame({"sales": df["sales"].values[0:10],
              "roll2": df["sales"].shift(1).rolling(window=2).mean().values[0:10],
              "roll3": df["sales"].shift(1).rolling(window=3).mean().values[0:10],
              "roll5": df["sales"].shift(1).rolling(window=5).mean().values[0:10]})

#    sales  roll2      roll3  roll5
# 0   13.0    NaN        NaN    NaN
# 1   11.0    NaN        NaN    NaN
# 2   14.0   12.0        NaN    NaN
# 3   13.0   12.5  12.666667    NaN
# ...


def roll_mean_features(dataframe, windows):
    dataframe = dataframe.copy()
    for window in windows:
        dataframe['sales_roll_mean_' + str(window)] = dataframe.groupby(["store", "item"])['sales']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(dataframe)
    return dataframe


df = roll_mean_features(df, [365, 546])

df.head()
df.tail()

#####################################################
# Exponentially Weighted Mean Features
#####################################################


pd.DataFrame({"sales": df["sales"].values[0:10],
              "roll2": df["sales"].shift(1).rolling(window=2).mean().values[0:10],
              "ewm099": df["sales"].shift(1).ewm(alpha=0.99).mean().values[0:10],
              "ewm095": df["sales"].shift(1).ewm(alpha=0.95).mean().values[0:10],
              "ewm07": df["sales"].shift(1).ewm(alpha=0.7).mean().values[0:10],
              "ewm01": df["sales"].shift(1).ewm(alpha=0.1).mean().values[0:10]})
#    sales  roll2     ewm099     ewm095      ewm07      ewm01
# 0   13.0    NaN        NaN        NaN        NaN        NaN
# 1   11.0    NaN  13.000000  13.000000  13.000000  13.000000
# 2   14.0   12.0  11.019802  11.095238  11.461538  11.947368
# 3   13.0   12.5  13.970201  13.855107  13.287770  12.704797
# 4   10.0   13.5  13.009702  13.042750  13.084686  12.790637


def ewm_features(dataframe, alphas, lags):
    dataframe = dataframe.copy()
    for alpha in alphas:
        for lag in lags:
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby(["store", "item"])['sales']. \
                    transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe


alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lags = [91, 98, 105, 112, 180, 270, 365, 546, 728]

df = ewm_features(df, alphas, lags)

check_df(df)
df.columns

#####################################################
# One-Hot Encoding
#####################################################

df = pd.get_dummies(df, columns=['store', 'item', 'day_of_week', 'month'])


#####################################################
# Converting sales to log(1+sales)
#####################################################

df['sales'] = np.log1p(df["sales"].values)

#####################################################
# Custom Cost Function
#####################################################

# MAE: mean absolute error
# MAPE: mean absolute percentage error
# SMAPE: Symmetric mean absolute percentage error (adjusted MAPE)


def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds-target)
    denom = np.abs(preds)+np.abs(target)
    smape_val = (200*np.sum(num/denom))/n
    return smape_val

def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False

#####################################################
# MODEL VALIDATION
#####################################################

#####################################################
# Time-Based Validation Sets
#####################################################


test["date"].min(), test["date"].max()
# (Timestamp('2018-01-01 00:00:00'), Timestamp('2018-03-31 00:00:00'))
train["date"].min(), train["date"].max()
# (Timestamp('2013-01-01 00:00:00'), Timestamp('2017-12-31 00:00:00'))

# we need to test ourselves first

train = df.loc[(df["date"] < "2017-01-01"), :]
train["date"].min(), train["date"].max()
# (Timestamp('2013-01-01 00:00:00'), Timestamp('2016-12-31 00:00:00'))

# 2017 first 3 month validation set
val = df.loc[(df["date"] >= "2017-01-01") & (df["date"] < "2017-04-01"), :]

df.columns
# take out variables which are not independent
cols = [col for col in train.columns if col not in ['date', 'id', "sales", "year"]]

Y_train = train['sales']
X_train = train[cols]

Y_val = val['sales']
X_val = val[cols]

Y_train.shape, X_train.shape, Y_val.shape, X_val.shape
# ((730500,), (730500, 142), (45000,), (45000, 142))
# we created validation set with the same size of the test set


#####################################################
# LightGBM Model
#####################################################

# LightGBM parameters
lgb_params = {'metric': {'mae'},
              'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'num_boost_round': 1000,
              'early_stopping_rounds': 200,
              'nthread': -1}

# 'early_stopping_rounds': if error does not fall with 200 iterations;
# num_boost_round': do not go until 1000 iterations - stop
# lower the time for training


# metric mae: l1, absolute loss, mean_absolute_error, regression_l1
# l2, square loss, mean_squared_error, mse, regression_l2, regression
# rmse, root square loss, root_mean_squared_error, l2_root
# mape, MAPE loss, mean_absolute_percentage_error

# learning_rate: shrinkage_rate, eta
# num_boost_round: n_estimators, number of boosting iterations.
# nthread: num_thread, nthread, nthreads, n_jobs

# transform train and validation sets to lgb form
lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)
lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)
type(lgbtrain)
# lightgbm.basic.Dataset

model = lgb.train(lgb_params, lgbtrain,
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=lgb_params['num_boost_round'],
                  early_stopping_rounds=lgb_params['early_stopping_rounds'],
                  feval=lgbm_smape,
                  verbose_eval=100)
# verbose_eval=100 - report in every 100
# feval : Customized evaluation function
# [1000]	training's l1: 0.130076	training's SMAPE: 13.3519	valid_1's l1: 0.134468	valid_1's SMAPE: 13.8186
y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)
smape(np.expm1(y_pred_val), np.expm1(Y_val))
# 13.818629488897852
# we need to take exp. since the calculations made with log values at first
# %13 error

##########################################
# Feature importance
##########################################

def plot_lgb_importances(model, plot=False, num=10):
    from matplotlib import pyplot as plt
    import seaborn as sns
    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show()
    else:
        print(feat_imp.head(num))


plot_lgb_importances(model, num=30)

#                          feature  split       gain
# 17           sales_roll_mean_546    930  54.551425
# 13                 sales_lag_364   1231  12.968768
# 16           sales_roll_mean_365    637   9.868275
# 60    sales_ewm_alpha_05_lag_365    347   4.866203
# 18    sales_ewm_alpha_095_lag_91     71   2.186980
# 54     sales_ewm_alpha_05_lag_91     89   1.873444
# 1                    day_of_year    754   1.873200
# 3                        is_wknd    229   1.216783
# 123                day_of_week_1    239   1.197569
#..
# gain : entropy change between before and after split --> information gain
# sales_lag_364 --> split 1231 times there may be a yearly sales pattern
# we can ask for 10 year data to see this pattern for instance
# sales_ewm_alpha_05_lag_365    347   4.866203 --> focused on far-past but most important

plot_lgb_importances(model, plot=True, num=30)

lgb.plot_importance(model, max_num_features=20, figsize=(10, 10), importance_type="gain")
plt.show()


##########################################
# Final Model
##########################################

train = df.loc[~df.sales.isna()]
Y_train = train['sales']
X_train = train[cols]

test = df.loc[df.sales.isna()]
X_test = test[cols]

lgb_params = {'metric': {'mae'},
              'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'nthread': -1,
              "num_boost_round": model.best_iteration}

# we fit the model with whole data no need for early_stopping

# LightGBM dataset
lgbtrain_all = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)
final_model = lgb.train(lgb_params, lgbtrain_all, num_boost_round=model.best_iteration)
test_preds = final_model.predict(X_test, num_iteration=model.best_iteration)

# Create submission
submission_df_ts = test.loc[:, ['id', 'sales']]
submission_df_ts['sales'] = np.expm1(test_preds)
submission_df_ts['id'] = submission_df.id.astype(int)
submission_df_ts.to_csv('submission.csv', index=False)








