# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

# %%
city = pd.read_csv("city_day.csv")
city.head()

# %%
city.info()

# %%
#### Extra Added ###
city['Date'] = city['Date'].str.split("-").str[1]
city['Date']=city['Date'].astype(int)
city = city.rename(columns={"Date": "Month"})
city.head()

# %% [markdown]
# ## DROP AQI BUCKET(NOT NECCESSARY)

# %%
city.drop("AQI_Bucket",axis=1,inplace=True)
city.head()

# %% [markdown]
# # Cleaning dataset

# %%
city.info()

# %%
city.isnull().sum()

# %%
city.shape

# %%
null_percentage = city.isnull().mean() * 100
print(null_percentage)

# %% [markdown]
# ## We'll drop all the features having null values more than 30%

# %%
threshold = 0.30
city = city.loc[:, city.isnull().mean() <= threshold]

# %%
city.isnull().mean() * 100

# %%
city.shape

# %% [markdown]
# ### Relacing null values by three methods and choosed the best one
# 1. Mean Imputation
# 2. Median Imputation
# 3. Random Sample Imputation : Remove null values and fill those null values by taking random samples from the dataset

# %%
def impute_nan(df,variable):
    
    df[variable+"_median"] = df[variable].fillna(df[variable].median())
    df[variable+"_mean"] = df[variable].fillna(df[variable].mean())
    df[variable+"_random"] = df[variable]
    # It will have the random sample to fill nan values
    random_sample = df[variable].dropna().sample(df[variable].isnull().sum(),
                                                 random_state=0)
    # Pandas need to have same index in order to merge the dataset
    random_sample.index = df[df[variable].isnull()].index
    df.loc[df[variable].isnull(),variable+'_random'] = random_sample

    fig = plt.figure()
    ax = fig.add_subplot([1.2,1.2,1.2,1.2])
    
    
    df[variable+"_mean"].plot(kind='kde',ax=ax, color='orange')
    df[variable+"_median"].plot(kind='kde',ax=ax, color='green')
    df[variable+"_random"].plot(kind='kde',ax=ax, color='deeppink')

    df[variable].plot(kind='kde', ax=ax, color='black')
    
    lines, labels = ax.get_legend_handles_labels()
    ax.legend(lines, labels, loc = 'best')

# %%
mean = city["PM2.5"].mean()
median = city["PM2.5"].median()
(mean, median)

# %%
impute_nan(city,"PM2.5")

# %%
# We'll take random(performing best)
city.drop(["PM2.5", "PM2.5_median", "PM2.5_mean"], axis=1, inplace=True)
city.head()

# %%
city.isnull().sum()

# %% [markdown]
# #### changing NO_day 

# %%
mean = city["NO"].mean()
median = city["NO"].median()
(mean, median)

# %%
impute_nan(city,"NO")

# %%
# We'll take random(performing best)
city.drop(["NO", "NO_median", "NO_mean"], axis=1, inplace=True)
city.head()

# %%
city.isnull().sum()

# %% [markdown]
# #### Changing NO2_day

# %%
mean = city["NO2"].mean()
median = city["NO2"].median()
(mean, median)

# %%
impute_nan(city,"NO2")

# %%
# We'll take random(performing best)
city.drop(["NO2", "NO2_median", "NO2_mean"], axis=1, inplace=True)
city.head()

# %%
city.isnull().sum()

# %% [markdown]
# #### Changing NOx

# %%
mean = city["NOx"].mean()
median = city["NOx"].median()
(mean, median)

# %%
impute_nan(city,"NOx")

# %%
# We'll take random(performing best)
city.drop(["NOx", "NOx_median", "NOx_mean"], axis=1, inplace=True)
city.head()

# %%
city.isnull().sum()

# %% [markdown]
# #### Changing CO

# %%
mean = city["CO"].mean()
median = city["CO"].median()
(mean, median)

# %%
impute_nan(city,"CO")

# %%
# We'll take mean
city.drop(["CO", "CO_random", "CO_median"], axis=1, inplace=True)
city.head()

# %%
city.isnull().sum()

# %%
def impute_nan_grp(df, variables):
   
    # Create subplots for 4 variables
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))  # 2x2 grid for 4 variables
    axs = axs.ravel()  # Flatten the array of axes for easier indexing
    
    for idx, variable in enumerate(variables):
        median = df[variable].median()
        mean = df[variable].mean()
        
        # Impute missing values with median, mean, and random sample
        df[variable+"_median"] = df[variable].fillna(median)
        df[variable+"_mean"] = df[variable].fillna(mean)
        df[variable+"_random"] = df[variable]
        
        # Random sample imputation
        random_sample = df[variable].dropna().sample(df[variable].isnull().sum(), random_state=0)
        random_sample.index = df[df[variable].isnull()].index
        df.loc[df[variable].isnull(), variable+'_random'] = random_sample
        
        # Plot KDEs on the corresponding subplot
        ax = axs[idx]
        
        df[variable+"_mean"].plot(kind='kde', ax=ax, color='orange', label='Mean Imputation')
        df[variable+"_median"].plot(kind='kde', ax=ax, color='green', label='Median Imputation')
        df[variable+"_random"].plot(kind='kde', ax=ax, color='deeppink', label='Random Sample Imputation')
        df[variable].plot(kind='kde', ax=ax, color='black', label='Original')
        
        # Customize plot for the current variable
        ax.set_title(f'Distribution of {variable}')
        ax.legend(loc='best')

    # Adjust layout and display
    plt.tight_layout()
    plt.show()

# %%
# Making list of all the columns having null values
variables = ['SO2','O3','Benzene','Toluene']
impute_nan_grp(city,variables)

# %% [markdown]
# #### Changing columns with their best replacement

# %%
city.drop(["SO2", "SO2_median", "SO2_mean",
          "O3","O3_median","O3_mean",
          "Benzene","Benzene_random","Benzene_median",
          "Toluene","Toluene_median","Toluene_mean"], axis=1, inplace=True)
city.head()

# %%
city.tail()

# %%
city.isnull().sum()

# %% [markdown]
# ### Seeing if data has any outliers

# %%
import seaborn as sns
sns.histplot(city['PM2.5_random'].dropna())

# %%
city.boxplot(column='PM2.5_random')

# %% [markdown]
# #### Too much outliers we have to remove them 
# ***Skewed data so we have to remove using below technique***

# %%
#### Let's compute the Interquantile range to calculate the boundaries
IQR = city["PM2.5_random"].quantile(0.75) - city["PM2.5_random"].quantile(0.25)
IQR

# %%
### Calculating the boundaries
lower_bridge = city["PM2.5_random"].quantile(0.25) - (IQR*1.5)
upper_bridge = city["PM2.5_random"].quantile(0.75) + (IQR*1.5)
(lower_bridge, upper_bridge)

# %%
### Extreme Outliers
lower_bridge = city["PM2.5_random"].quantile(0.25) - (IQR*3)
upper_bridge = city["PM2.5_random"].quantile(0.75) + (IQR*3)
(lower_bridge, upper_bridge)

# %% [markdown]
# ### Not considring lower boundary as it only has poitive values

# %% [markdown]
# ## Remvoing Outliers

# %%
# Taking value b/w 158 and 236
city.loc[city["PM2.5_random"]>=200,"PM2.5_random"] = 200
city.boxplot(column='PM2.5_random')

# %%
sns.histplot(city['PM2.5_random'].dropna())

# %%
city.head()

# %%
sns.histplot(city['NO_random'].dropna())

# %%
city.boxplot(column='NO_random')

# %%
#### Let's compute the Interquantile range to calculate the boundaries
IQR = city["NO_random"].quantile(0.75) - city["NO_random"].quantile(0.25)
IQR

# %%
### Calculating the boundaries
lower_bridge = city["NO_random"].quantile(0.25) - (IQR*1.5)
upper_bridge = city["NO_random"].quantile(0.75) + (IQR*1.5)
(lower_bridge, upper_bridge)

# %%
### Extreme Outliers
lower_bridge = city["NO_random"].quantile(0.25) - (IQR*3)
upper_bridge = city["NO_random"].quantile(0.75) + (IQR*3)
(lower_bridge, upper_bridge)

# %%
# Removing Outliers
city.loc[city["NO_random"]>=41.71000000000001,"NO_random"] = 41.71000000000001
sns.histplot(city['NO_random'].dropna())

# %%
city.boxplot(column='NO_random')

# %%
city.head()

# %%
sns.histplot(city['NO2_random'].dropna())

# %%
city.boxplot(column='NO2_random')

# %%
#### Let's compute the Interquantile range to calculate the boundaries
IQR = city["NO2_random"].quantile(0.75) - city["NO2_random"].quantile(0.25)
IQR

# %%
### Calculating the boundaries
lower_bridge = city["NO2_random"].quantile(0.25) - (IQR*1.5)
upper_bridge = city["NO2_random"].quantile(0.75) + (IQR*1.5)
(lower_bridge, upper_bridge)

# %%
### Extreme Outliers
lower_bridge = city["NO2_random"].quantile(0.25) - (IQR*3)
upper_bridge = city["NO2_random"].quantile(0.75) + (IQR*3)
(lower_bridge, upper_bridge)

# %%
# Removing Outliers
city.loc[city["NO2_random"]>=76.26,"NO2_random"] = 76.26
sns.histplot(city['NO2_random'].dropna())

# %%
city.boxplot(column='NO2_random')

# %%
city.head()

# %%
sns.histplot(city['NOx_random'].dropna())

# %%
city.boxplot(column='NOx_random')

# %%
#### Let's compute the Interquantile range to calculate the boundaries
IQR = city["NOx_random"].quantile(0.75) - city["NOx_random"].quantile(0.25)
IQR

# %%
### Calculating the boundaries
lower_bridge = city["NOx_random"].quantile(0.25) - (IQR*1.5)
upper_bridge = city["NOx_random"].quantile(0.75) + (IQR*1.5)
(lower_bridge, upper_bridge)

# %%
### Extreme Outliers
lower_bridge = city["NOx_random"].quantile(0.25) - (IQR*3)
upper_bridge = city["NOx_random"].quantile(0.75) + (IQR*3)
(lower_bridge, upper_bridge)

# %%
# Removing Outliers
city.loc[city["NOx_random"]>=89,"NOx_random"] = 89
sns.histplot(city['NOx_random'].dropna())

# %%
city.boxplot(column='NOx_random')

# %%
city.head()

# %%
city.boxplot(column='CO_mean')

# %%
sns.histplot(city['CO_mean'].dropna())

# %%
#### Let's compute the Interquantile range to calculate the boundaries
IQR = city["CO_mean"].quantile(0.75) - city["CO_mean"].quantile(0.25)
IQR

# %%
### Calculating the boundaries
lower_bridge = city["CO_mean"].quantile(0.25) - (IQR*1.5)
upper_bridge = city["CO_mean"].quantile(0.75) + (IQR*1.5)
(lower_bridge, upper_bridge)

# %%
### Extreme Outliers
lower_bridge = city["CO_mean"].quantile(0.25) - (IQR*3)
upper_bridge = city["CO_mean"].quantile(0.75) + (IQR*3)
(lower_bridge, upper_bridge)

# %%
# Removing Outliers
city.loc[city["CO_mean"]>=3.465,'CO_mean'] = 3.465
sns.histplot(city['CO_mean'].dropna())

# %%
city.boxplot(column='CO_mean')

# %%
city.head()

# %%
sns.histplot(city['SO2_random'].dropna())

# %%
city.boxplot(column='SO2_random')

# %%
#### Let's compute the Interquantile range to calculate the boundaries
IQR = city["SO2_random"].quantile(0.75) - city["SO2_random"].quantile(0.25)
IQR

# %%
### Calculating the boundaries
lower_bridge = city["SO2_random"].quantile(0.25) - (IQR*1.5)
upper_bridge = city["SO2_random"].quantile(0.75) + (IQR*1.5)
(lower_bridge, upper_bridge)

# %%
### Extreme Outliers
lower_bridge = city["SO2_random"].quantile(0.25) - (IQR*3)
upper_bridge = city["SO2_random"].quantile(0.75) + (IQR*3)
(lower_bridge, upper_bridge)

# %%
# Removing Outliers
city.loc[city["SO2_random"]>=29.36,'SO2_random'] = 29.36
sns.histplot(city['SO2_random'].dropna())

# %%
city.boxplot(column='SO2_random')

# %%
city.head()

# %%
sns.histplot(city['O3_random'].dropna())

# %%
city.boxplot(column='O3_random')

# %%
#### Let's compute the Interquantile range to calculate the boundaries
IQR = city["O3_random"].quantile(0.75) - city["O3_random"].quantile(0.25)
IQR

# %%
### Calculating the boundaries
lower_bridge = city["O3_random"].quantile(0.25) - (IQR*1.5)
upper_bridge = city["O3_random"].quantile(0.75) + (IQR*1.5)
(lower_bridge, upper_bridge)

# %%
# Removing Outliers
city.loc[city["O3_random"]>=85.55499999999999,'O3_random'] = 85.55499999999999
sns.histplot(city['O3_random'].dropna())

# %%
city.boxplot(column='O3_random')

# %%
city.head()

# %%
sns.histplot(city['Benzene_mean'].dropna())

# %%
city.boxplot(column='Benzene_mean')

# %%
#### Let's compute the Interquantile range to calculate the boundaries
IQR = city["Benzene_mean"].quantile(0.75) - city["Benzene_mean"].quantile(0.25)
IQR

# %%
### Calculating the boundaries
lower_bridge = city["Benzene_mean"].quantile(0.25) - (IQR*1.5)
upper_bridge = city["Benzene_mean"].quantile(0.75) + (IQR*1.5)
(lower_bridge, upper_bridge)

# %%
# Removing Outliers
city.loc[city["Benzene_mean"]>=7.842100761251463,'Benzene_mean'] = 7.842100761251463
sns.histplot(city['Benzene_mean'].dropna())

# %%
city.boxplot(column='Benzene_mean')

# %%
city.head()

# %%
sns.histplot(city['Toluene_random'].dropna())

# %%
city.boxplot(column='Toluene_random')

# %%
#### Let's compute the Interquantile range to calculate the boundaries
IQR = city["Toluene_random"].quantile(0.75) - city["Toluene_random"].quantile(0.25)
IQR

# %%
### Calculating the boundaries
lower_bridge = city["Toluene_random"].quantile(0.25) - (IQR*1.5)
upper_bridge = city["Toluene_random"].quantile(0.75) + (IQR*1.5)
(lower_bridge, upper_bridge)

# %%
# Removing Outliers
city.loc[city["Toluene_random"]>=22.075,'Toluene_random'] = 22.075
sns.histplot(city['Toluene_random'].dropna())

# %%
city.boxplot(column='Toluene_random')

# %%
city.head()

# %% [markdown]
# # Categorical Features to Numerical Features

# %%
city.info()

# %% [markdown]
# ***LABEL encoding***

# %%
city.City.unique()

# %%
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

# Convert City column to string type to ensure uniformity
city['City'] = city['City'].astype(str)
# Apply Label Encoding to the 'City' column
city['City'] = label_encoder.fit_transform(city['City'])

# %%
city.head()

# %%
city.info()

# %% [markdown]
# ## Remvoing data with `NAN` AQI values

# %%
df_test = city[city['AQI'].isnull()]
df_test.head()

# %%
city = city[~city['AQI'].isnull()]

# %%
city.head()

# %% [markdown]
# # Test - Trian split

# %%
X = city.drop('AQI',axis=1)
X.head()

# %%
y = city['AQI']
y.head()

# %%
# Split data into train and test sets
np.random.seed(42)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,
                                                   y,
                                                   test_size = 0.2)

# %% [markdown]
# # Feature Selection
# ### 1. Checking and dropping constant features
# ***Removing Low-Variance features***

# %%
from sklearn.feature_selection import VarianceThreshold
var_thres = VarianceThreshold(threshold=0)
var_thres.fit(X_train)

# %%
var_thres.get_support()

# %% [markdown]
# ### No low variance feature

# %% [markdown]
# ## 2.Feature Selection with Correlation

# %%
X_train.corr()

# %%
import seaborn as sns
plt.figure(figsize=(12,10))
cor = X_train.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.CMRmap_r)

# Save the plot as an image
plt.savefig("corr_image.png")  # Specify the desired filename and format
plt.show()

# %% [markdown]
# ### Selecting highly correlated features

# %%
# It will remove the first feature that is correlated with any other feature
def correlation(dataset, threshold):
    col_corr = set() # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i,j] > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr

# %%
corr_features = correlation(X_train,0.85)
len(set(corr_features))

# %%
corr_features

# %%
### Dropping highly correlated features
X_train.drop(corr_features,axis=1,inplace=True)
X_test.drop(corr_features,axis=1,inplace=True)

# %%
X_train.head()

# %%
city.shape

# %% [markdown]
# ## Building Model

# %%
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model to the training data
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

rf_model.score(X_test, y_test)

# %%
rf_model.predict([[289895, 677766, 100000, 499980, 276876.85, 17684.05, 89859, 2868.10, 17799.05, 3768.28084, 189898.92]])

# %% [markdown]
# ***Applying 6 models and choosing best***
# 1. RidgeRegression
# 2. SupportVectorRegression(kernal='linear')
# 3. RandomForestRegressors
# 4. SupportVectorRegression(kernal='rbf')
# 5. Lasso
# 6. ElasticNet

# %%
"""from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Create a dictionary with regression models
regression_models = {
    "Ridge Regression": Ridge(),
    "Support Vector Regression (Linear)": SVR(kernel='linear'),
    "Random Forest Regressor": RandomForestRegressor(),
    "Support Vector Regression (RBF)": SVR(kernel='rbf'),
    "Lasso Regression": Lasso(),
    "ElasticNet": ElasticNet()
}

# Function to fit and score regression models
def fit_and_score_regressors(models, X_train, X_test, y_train, y_test):
    # Set random seed
    np.random.seed(42)
    # Dictionary to keep model scores
    model_scores = {}
    # Loop through models
    for name, model in models.items():
        # Fit the model to the training data
        model.fit(X_train, y_train)
        # Evaluate the model on the test data and store the score (R^2 by default for regression)
        model_scores[name] = model.score(X_test, y_test)
    return model_scores

# Example usage:
# scores = fit_and_score_regressors(regression_models, X_train, X_test, y_train, y_test)
"""

# %%
"""model_scores = fit_and_score_regressors(models=regression_models,
                             X_train=X_train,
                             X_test=X_test,
                             y_train=y_train,
                             y_test=y_test)

model_scores"""

# %% [markdown]
# ## Applying LSTM

# %%
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# %%
data = city['AQI'].values
data = data.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# %%
# Create a function to create sequences of data
def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 10  # You can adjust this
X, y = create_sequences(scaled_data, seq_length)

# %%
# Split into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# %%
# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# %%
# Train the model
model.fit(X_train, y_train, batch_size=1, epochs=10)

# Make predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# %%
# Evaluate the model
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(scaler.inverse_transform(y_test.reshape(-1, 1)), predictions)
print(f'Mean Squared Error: {mse}')

# %% [markdown]
# ## Mitigation System

# %%
aqi_thresholds = {
    (0, 50): {
        "category": "Good",
        "warnings": ["No immediate action required."],
        "actions": ["Enjoy outdoor activities."]
    },
    (51, 80): {
        "category": "Moderate",
        "warnings": ["Air quality is acceptable."],
        "actions": ["Sensitive groups should consider reducing prolonged exertion."]
    },
    (81, 120): {
        "category": "Unhealthy for Sensitive Groups",
        "warnings": ["Wear masks outdoors."],
        "actions": ["Install air purifiers indoors."]
    },
    (121, 150): {
        "category": "Unhealthy",
        "warnings": ["Wear N95 masks."],
        "actions": ["Avoid outdoor activities.", "Use air purifiers."]
    },
    (151, 200): {
        "category": "Very Unhealthy",
        "warnings": ["Health alert: Everyone may experience effects."],
        "actions": ["Close windows.", "Run air purifiers at max."]
    },
    (201, 300): {
        "category": "Hazardous",
        "warnings": ["Emergency conditions."],
        "actions": ["Stay indoors.", "Use oxygen masks if necessary."]
    },
    (301, 500): {
        "category": "Severe",
        "warnings": ["Evacuate if possible."],
        "actions": ["Seek medical help for breathing issues."]
    }
}

# %%
def get_mitigation_measures(aqi):
    for (low, high), measures in aqi_thresholds.items():
        if low <= aqi <= high:
            return {
                "AQI": aqi,
                "Category": measures["category"],
                "Warnings": measures["warnings"],
                "Actions": measures["actions"]
            }
    return {
        "AQI": aqi,
        "Category": "Unhealthy",
        "Warnings": ["Wear N95 masks"],
        "Actions": ["Seek medical help for breathing issues"]
    }

# %%
# Generate mitigation advice for test predictions
mitigation_advice = [get_mitigation_measures(pred) for pred in y_pred]

# Convert to DataFrame for better visualization
import pandas as pd
mitigation_df = pd.DataFrame(mitigation_advice)

# %% [markdown]
# # AQI Predictions with Mitigation Advice:

# %%
mitigation_df

# %% [markdown]
# ## Alert System

# %%
def send_alert(aqi):
    measures = get_mitigation_measures(aqi)
    message = f"""
    ALERT! AQI Level: {aqi} ({measures['Category']})
    Warnings: {', '.join(measures['Warnings'])}
    Recommended Actions: {', '.join(measures['Actions'])}
    """
    print(message)  # Email Bhej dega.....


for pred in y_pred:
    if pred > 150:
        send_alert(pred)

# %% [markdown]
# ### Alerting beforehand

# %%
import pandas as pd

# Calculate monthly averages from available data
historical_monthly_avg = city.groupby('Month')['AQI'].mean().to_dict()

def predict_aqi(month):
    """Predict AQI based on historical monthly averages"""
    return historical_monthly_avg.get(month, None)

# Enhanced alert function with trend comparison
def send_alert(month, current_aqi):
    predicted_aqi = historical_monthly_avg.get(month)
    
    if predicted_aqi is None or pd.isna(current_aqi):
        return
    
    measures = get_mitigation_measures(current_aqi)
    
    message = f"""
    {'!'*20} AQI ALERT {'!'*20}
    Month: {month}
    Current AQI: {current_aqi:.1f}
    Historical Average: {predicted_aqi:.1f}
    Deviation: {(current_aqi - predicted_aqi)/predicted_aqi:.0%}
    
    Category: {measures['Category']}
    Warnings: {', '.join(measures['Warnings'])}
    Actions Required: {', '.join(measures['Actions'])}
    """
    print(message)

# Generate alerts for all months in dataset
for month in city['Month'].unique():
    current_aqi = city[city['Month'] == month]['AQI'].mean()
    
    if current_aqi > 150:
        send_alert(month, current_aqi)

# %%
def predict_future_aqi(target_month):
    """Predict next month's AQI using simple moving average"""
    historical_values = city[city['Month'] == target_month]['AQI']
    
    if len(historical_values) == 0:
        return None
        
    # Simple moving average prediction
    return historical_values.mean()

# Example usage for month 3 prediction
next_month = 3
prediction = predict_future_aqi(next_month)

if prediction and prediction > 150:
    print(f"Predicted AQI for month {next_month}: {prediction:.1f}")
    send_alert(next_month, prediction)

# %%
import numpy as np
y_pred = np.array(y_pred).tolist()

# %% [markdown]
# ## Uploading data on firebase

# %%
import firebase_admin
from firebase_admin import credentials, db
import pandas as pd

# Initialize Firebase App
def initialize_firebase():
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate("n-kings-project-id-firebase-adminsdk-fbsvc-04d8a0117e.json")
            firebase_admin.initialize_app(cred, {
                'databaseURL': 'https://n-kings-project-id-default-rtdb.asia-southeast1.firebasedatabase.app/'  # Verify this URL
            })
        print("Firebase initialized successfully.")
    except Exception as e:
        print(f"Firebase initialization failed: {str(e)}")



# Upload Mitigation Advice
def upload_mitigation_advice(mitigation_df):
    try:
        ref = db.reference('/mitigation_advice')
        if not isinstance(mitigation_df, pd.DataFrame) or mitigation_df.empty:
            raise ValueError("Invalid or empty DataFrame provided for mitigation advice.")

        advice_dict = mitigation_df.to_dict(orient='records')
        ref.set(advice_dict)
        print(f"Successfully uploaded {len(advice_dict)} mitigation records.")
    except Exception as e:
        print(f"Upload failed: {str(e)}")

#  Upload Alert History
def upload_alerts(y_pred):
    try:
        ref = db.reference('/alerts')
        if not isinstance(y_pred, list) or not y_pred:
            raise ValueError("Invalid or empty predictions list provided.")

        alerts = [get_mitigation_measures(pred) for pred in y_pred]
        
        for idx, alert in enumerate(alerts):
            ref.child(f'alert_{idx}').set(alert)
        
        print(f"Successfully uploaded {len(alerts)} alerts.")
    except Exception as e:
        print(f"Alert upload failed: {str(e)}")

# Main Execution
if __name__ == "__main__":
    # Initialize Firebase connection
    initialize_firebase()
    
    # Check if variables exist before uploading data
    try:
        if 'city' in globals() and isinstance(city, pd.DataFrame):
            upload_aqi_data(city)
        else:
            print("Skipping AQI data upload: 'city' DataFrame not found.")

        if 'y_pred' in globals() and isinstance(y_pred, list):
            mitigation_advice = [get_mitigation_measures(pred) for pred in y_pred]
            mitigation_df = pd.DataFrame(mitigation_advice)
            upload_mitigation_advice(mitigation_df)
            upload_alerts(y_pred)
        else:
            print("Skipping alerts and mitigation upload: 'y_pred' list not found.")

        print("All data uploaded to Firebase.")
    except Exception as e:
        print(f"Main execution failed: {str(e)}")


# %%



