#!/usr/bin/env python
# coding: utf-8

# ### Importing Necessary Libraries

# In[216]:


# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec #for more complex arrangements of subplots within a figure
import plotly.graph_objects as go #create a wide variety of visualizations
from matplotlib.colors import LinearSegmentedColormap #create custom colormaps
from matplotlib import colors as mcolors #handling colors in Matplotlib
from scipy.stats import linregress #performing simple linear regression
from sklearn.ensemble import IsolationForest #used for anomaly detection
from sklearn.preprocessing import StandardScaler # used to standardize features
from sklearn.decomposition import PCA #dimensionality reduction technique (reducing the number of features)
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer #visualizations to evaluate the performance of clustering algorithms 
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score #performance of clustering algorithms
from sklearn.cluster import KMeans #import algorithm for clustering
from sklearn.cluster import DBSCAN
from tabulate import tabulate #used to create formatted tables from data
from collections import Counter #specialized dictionary used to count the frequency of elements in an iterable

#magic command (display of Matplotlib plots directly within the notebook cells)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[217]:


# Initialize Plotly for use in the notebook
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)


# In[218]:


# Configure Seaborn plot styles: Set background color and use dark grid
sns.set(rc={'axes.facecolor': '#fcf0dc'}, style='darkgrid')


# ### Loading Dataset

# In[219]:


df = pd.read_csv('online retail.csv', encoding="ISO-8859-1")


# #### Dataset Description
# 
# InvoiceNo:	   Code representing each unique transaction. If this code starts with letter 'c', it indicates a cancellation.
# 
# StockCode:	   Code uniquely assigned to each distinct product.
# 
# Description:   Description of each product.
# 
# Quantity:	   The number of units of a product in a transaction.
# 
# InvoiceDate:   The date and time of the transaction.
# 
# UnitPrice:	   The unit price of the product in sterling.
# 
# CustomerID:	   Identifier uniquely assigned to each customer.
# 
# Country:	    The country of the customer.

# ### Data Understanding

# In[220]:


#Datset Overview
df.head(10)


# In[221]:


#check null values and datatype
df.info()


# #### 1) The Dataset consists of 541,909 entries and 8 columns
# #### 2)  Description:  This Column has some missing values, with 540,455 non-null entries out of 541,909.
# #### 3) CustomerID: This column has a significant number of missing values, with only 406,829 non-null entries out of 541,909.
# #### 4) InvoiceDate: This column is already in datetime format, which will facilitate further time series analysis.
# #### 5) We also observe that a single customer can have multiple transactions as inferred from the repeated CustomerID in the initial rows.

# In[222]:


#Statistical Information about Numerical Column
df.describe().T


# In[223]:


#Statistical Information about Categorical Column
df.describe(include='object').T


# ### Data Cleaning

# ### Handling Missing Value

# In[224]:


#checking null values
df.isnull().sum()


# In[225]:


#percentage of null values
per_null = round( 100 * (df.isnull().sum()) / len(df), 2 )
print(per_null)


# **CustomerID** Column Contains 24.93% missing values
# 
# **Description** Column Contains 0.27% missing values

# In[226]:


# Extracting rows with missing values in 'CustomerID' or 'Description' columns
df[df['CustomerID'].isnull() | df['Description'].isnull()].head()


# In[227]:


# Removing rows with missing values in 'CustomerID' and 'Description' columns
df = df.dropna(subset=['CustomerID', 'Description'])


# In[228]:


# Verifying the removal of missing values
df.isnull().sum().sum()


# ### Handling Duplicates

# In[229]:


# Finding duplicate rows
dup = df[df.duplicated(keep=False)]

# Sorting the data by columns to see the duplicate
dup_sort = dup.sort_values(by=['InvoiceNo', 'StockCode', 'Description', 'CustomerID', 'Quantity'])

# Displaying first 10 records
dup_sort.head(10)


# ### Removing Duplicates

# In[230]:


# Displaying N0. of duplicates
print(f"The dataset contains {df.duplicated().sum()} duplicate rows that need to be removed.")


# In[231]:


# Removing duplicate rows
df.drop_duplicates(inplace=True)


# In[232]:


# Getting the number of rows in the dataframe
df.shape[0]


# ### Treating Cancelled Transactions
# 
# To refine our understanding of customer behavior and preferences, we need to take into account the transactions that were cancelled. Initially, we will identify these transactions by filtering the rows where the InvoiceNo starts with "C". Subsequently, we will analyze these rows to understand their common characteristics or patterns:

# In[233]:


# Filter out the rows with InvoiceNo starting with "C" and create a new column indicating the transaction status
df['Transaction_Status'] = np.where(df['InvoiceNo'].astype(str).str.startswith('C'), 'Cancelled', 'Completed')

# Analyze the characteristics of these rows (considering the new column)
cancelled_transactions = df[df['Transaction_Status'] == 'Cancelled']
cancelled_transactions.describe().drop('CustomerID', axis=1)


# In[234]:


df


# In[235]:


cancelled_transactions.head(10)


# All quantities in the cancelled transactions are negative, indicating that these are indeed orders that were cancelled.
# 
# The UnitPrice column has a considerable spread, showing that a variety of products, from low to high value, were part of the cancelled transactions.

# ### Handling Cancelled Transactions:
# 
# Considering the project's objective to cluster customers based on their purchasing behavior and preferences,it's imperative to understand the cancellation patterns of customers. Therefore, the strategy is to retain these cancelled transactions in the dataset, marking them distinctly to facilitate further analysis. This approach will:
# 
# Enhance the clustering process by incorporating patterns and trends observed in cancellation data, which might represent certain customer behaviors or preferences.
# 

# In[236]:


# Finding the percentage of cancelled transactions
cancelled_percentage = (cancelled_transactions.shape[0] / df.shape[0]) * 100

# Printing the percentage of cancelled transactions
print(f"The percentage of cancelled transactions in the dataset is: {cancelled_percentage:.2f}%")


# ### Treating Zero Unit Prices

# In[237]:


df['UnitPrice'].describe()


# -The minimum unit price value is zero. This suggests that there are some transactions where the unit price is zero, potentially indicating a free item or a data entry error. To understand their nature, it is essential to investigate these zero unit price transactions further. A detailed analysis of the product descriptions associated with zero unit prices will be conducted to determine if they adhere to a specific pattern:

# In[238]:


df[df['UnitPrice']==0].describe()[['Quantity']]


# -The transactions with a unit price of zero are relatively few in number (33 transactions).
# 
# -These transactions have a large variability in the quantity of items involved, ranging from 1 to 12540, with a substantial standard deviation.
# 
# -Including these transactions in the clustering analysis might introduce noise and could potentially distort the customer behavior patterns identified by the clustering algorithm.
# 
# -Given the small number of these transactions and their potential to introduce noise in the data analysis, the strategy should be to remove these transactions from the dataset. This would help in maintaining a cleaner and more consistent dataset, which is essential for building an accurate and reliable clustering model and recommendation system.

# In[239]:


# Removing records with a unit price of zero to avoid potential data entry errors
df = df[df['UnitPrice'] > 0]


# In[240]:


# Getting the number of rows in the dataframe
df.shape[0]


# ### Outlier Treatment

# In K-means clustering, the algorithm is sensitive to both the scale of data and the presence of outliers, as they can significantly influence the position of centroids, potentially leading to incorrect cluster assignments. However, considering the context of this project where the final goal is to understand customer behavior and preferences through K-means clustering, it would be more prudent to address the issue of outliers after the feature engineering phase where we create a customer-centric dataset. At this stage, the data is transactional, and removing outliers might eliminate valuable information that could play a crucial role in segmenting customers later on. Therefore, we will postpone the outlier treatment and proceed to the next stage for now.

# In[241]:


# Resetting the index of the cleaned dataset
df.reset_index(drop=True, inplace=True)


# In[242]:


# Getting the number of rows in the dataframe
df.shape[0]


# ### Feature Engineering

# ### RFM Feature
# 
# RFM is a method used for analyzing customer value and segmenting the customer base. It is an acronym that stands for:
# 
# Recency (R): This metric indicates how recently a customer has made a purchase. A lower recency value means the customer has purchased more recently, indicating higher engagement with the brand.
# 
# Frequency (F): This metric signifies how often a customer makes a purchase within a certain period. A higher frequency value indicates a customer who interacts with the business more often, suggesting higher loyalty or satisfaction.
# 
# Monetary (M): This metric represents the total amount of money a customer has spent over a certain period. Customers who have a higher monetary value have contributed more to the business, indicating their potential high lifetime value.
# 
# Together, these metrics help in understanding a customer's buying behavior and preferences, which is pivotal in personalizing marketing strategies and creating a recommendation system.

# #### Recency

# In this step, we focus on understanding how recently a customer has made a purchase. This is a crucial aspect of customer segmentation as it helps in identifying the engagement level of customers. Here, I am going to define the following feature:
# 
# -Days Since Last Purchas: This feature represents the number of days that have passed since the customer's last purchase. A lower value indicates that the customer has purchased recently, implying a higher engagement level with the business, whereas a higher value may indicate a lapse or decreased engagement. By understanding the recency of purchases, businesses can tailor their marketing strategies to re-engage customers who have not made purchases in a while, potentially increasing customer retention and fostering loyalty.

# In[243]:


# Convert InvoiceDate to datetime type
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Convert InvoiceDate to datetime and extract only the date
df['InvoiceDay'] = df['InvoiceDate'].dt.date

# Find the most recent purchase date for each customer
customer_data = df.groupby('CustomerID')['InvoiceDay'].max().reset_index()

# Find the most recent date in the entire dataset
most_recent_date = df['InvoiceDay'].max()

# Convert InvoiceDay to datetime type before subtraction
customer_data['InvoiceDay'] = pd.to_datetime(customer_data['InvoiceDay'])
most_recent_date = pd.to_datetime(most_recent_date)

# Calculate the number of days since the last purchase for each customer
customer_data['Recency'] = (most_recent_date - customer_data['InvoiceDay']).dt.days

# Remove the InvoiceDay column
customer_data.drop(columns=['InvoiceDay'], inplace=True)


# In[244]:


customer_data.head()


# In[245]:


customer_data.shape


# #### Frequency

# In this step, I am going to create two features that quantify the frequency of a customer's engagement with the retailer:
# 
# -Total Transactions: This feature represents the total number of transactions made by a customer. It helps in understanding the engagement level of a customer with the retailer.
# 
# -Total Products Purchased: This feature indicates the total number of products (sum of quantities) purchased by a customer across all transactions. It gives an insight into the customer's buying behavior in terms of the volume of products purchased.
# 
# These features will be crucial in segmenting customers based on their buying frequency, which is a key aspect in determining customer segments for targeted marketing and personalized recommendations.

# In[246]:


# Calculate the total number of transactions made by each customer
total_transactions = df.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()
total_transactions.rename(columns={'InvoiceNo': 'Frequency'}, inplace=True)

# Calculate the total number of products purchased by each customer
#total_products_purchased = df.groupby('CustomerID')['Quantity'].sum().reset_index()
#total_products_purchased.rename(columns={'Quantity': 'Total_Products_Purchased'}, inplace=True)

# Merge the new features into the customer_data dataframe
customer_data = pd.merge(customer_data, total_transactions, on='CustomerID')
#customer_data = pd.merge(customer_data, total_products_purchased, on='CustomerID')

# Display the first few rows of the customer_data dataframe
customer_data.head()


# #### Monetry
# 
# In this step, I am going to create two features that represent the monetary aspect of customer's transactions:
# 
# -Total Spend: This feature represents the total amount of money spent by each customer. It is calculated as the sum of the product of UnitPrice and Quantity for all transactions made by a customer. This feature is crucial as it helps in identifying the total revenue generated by each customer, which is a direct indicator of a customer's value to the business.
# 
# -Average Transaction Value: This feature is calculated as the Total Spend divided by the Total Transactions for each customer. It indicates the average value of a transaction carried out by a customer. This metric is useful in understanding the spending behavior of customers per transaction, which can assist in tailoring marketing strategies and offers to different customer segments based on their average spending patterns.

# In[247]:


# Calculate the total spend by each customer
df['Monetry'] = df['UnitPrice'] * df['Quantity']
total_spend = df.groupby('CustomerID')['Monetry'].sum().reset_index()

# Calculate the average transaction value for each customer
#average_transaction_value = total_spend.merge(total_transactions, on='CustomerID')
#average_transaction_value['Average_Transaction_Value'] = average_transaction_value['Total_Spend'] / average_transaction_value['Total_Transactions']

# Merge the new features into the customer_data dataframe
customer_data = pd.merge(customer_data, total_spend, on='CustomerID')
#customer_data = pd.merge(customer_data, average_transaction_value[['CustomerID', 'Average_Transaction_Value']], on='CustomerID')

# Display the first few rows of the customer_data dataframe
customer_data.head()


# #### Product Diversity
# 
# In this step, we are going to understand the diversity in the product purchase behavior of customers. Understanding product diversity can help in crafting personalized marketing strategies and product recommendations. Here, I am going to define the following feature:
# 
# Unique Products Purchased: This feature represents the number of distinct products bought by a customer. A higher value indicates that the customer has a diverse taste or preference, buying a wide range of products, while a lower value might indicate a focused or specific preference. Understanding the diversity in product purchases can help in segmenting customers based on their buying diversity, which can be a critical input in personalizing product recommendations.

# In[248]:


# Calculate the number of unique products purchased by each customer
different_products_purchased = df.groupby('CustomerID')['StockCode'].nunique().reset_index()
different_products_purchased.rename(columns={'StockCode': 'Diff_product'}, inplace=True)

# Merge the new feature into the customer_data dataframe
customer_data = pd.merge(customer_data, different_products_purchased, on='CustomerID')

# Display the first few rows of the customer_data dataframe
customer_data.head()


# #### Behavirol Features
# 
# In this step, we aim to understand and capture the shopping patterns and behaviors of customers. These features will give us insights into the customers' preferences regarding when they like to shop, which can be crucial information for personalizing their shopping experience. Here are the features I am planning to introduce:
# 
# -Average Days Between Purchases: This feature represents the average number of days a customer waits before making another purchase. Understanding this can help in predicting when the customer is likely to make their next purchase, which can be a crucial metric for targeted marketing and personalized promotions.
# 
# -Favorite Shopping Day: This denotes the day of the week when the customer shops the most. This information can help in identifying the preferred shopping days of different customer segments, which can be used to optimize marketing strategies and promotions for different days of the week.
# 
# -Favorite Shopping Hour: This refers to the hour of the day when the customer shops the most. Identifying the favorite shopping hour can aid in optimizing the timing of marketing campaigns and promotions to align with the times when different customer segments are most active.
# 
# By including these behavioral features in our dataset, we can create a more rounded view of our customers, which will potentially enhance the effectiveness of the clustering algorithm, leading to more meaningful customer segments.

# In[249]:


# Extract day of week and hour from InvoiceDate
df['Fav_Day'] = df['InvoiceDate'].dt.dayofweek
#df['Hour'] = df['InvoiceDate'].dt.hour

# Calculate the average number of days between consecutive purchases
#days_between_purchases = df.groupby('CustomerID')['InvoiceDay'].apply(lambda x: (x.diff().dropna()).apply(lambda y: y.days))
#average_days_between_purchases = days_between_purchases.groupby('CustomerID').mean().reset_index()
#average_days_between_purchases.rename(columns={'InvoiceDay': 'Average_Days_Between_Purchases'}, inplace=True)

# Find the favorite shopping day of the week
favorite_shopping_day = df.groupby(['CustomerID', 'Fav_Day']).size().reset_index(name='Count')
favorite_shopping_day = favorite_shopping_day.loc[favorite_shopping_day.groupby('CustomerID')['Count'].idxmax()][['CustomerID', 'Fav_Day']]

# Find the favorite shopping hour of the day
#favorite_shopping_hour = df.groupby(['CustomerID', 'Hour']).size().reset_index(name='Count')
#favorite_shopping_hour = favorite_shopping_hour.loc[favorite_shopping_hour.groupby('CustomerID')['Count'].idxmax()][['CustomerID', 'Hour']]

# Merge the new features into the customer_data dataframe
#customer_data = pd.merge(customer_data, average_days_between_purchases, on='CustomerID')
customer_data = pd.merge(customer_data, favorite_shopping_day, on='CustomerID')
#customer_data = pd.merge(customer_data, favorite_shopping_hour, on='CustomerID')

# Display the first few rows of the customer_data dataframe
customer_data.head()


# In[250]:


customer_data['Fav_Day'].nunique()


# #### Geographic Feature
# 
# In this step, we will introduce a geographic feature that reflects the geographical location of customers. Understanding the geographic distribution of customers is pivotal for several reasons:
# 
# -Country: This feature identifies the country where each customer is located. Including the country data can help us understand region-specific buying patterns and preferences. Different regions might have varying preferences and purchasing behaviors which can be critical in personalizing marketing strategies and inventory planning. Furthermore, it can be instrumental in logistics and supply chain optimization, particularly for an online retailer where shipping and delivery play a significant role.

# In[251]:


df['Country'].value_counts(normalize=True).head()


# Given that a substantial portion (89%) of transactions are originating from the United Kingdom, we might consider creating a binary feature indicating whether the transaction is from the UK or not. This approach can potentially streamline the clustering process without losing critical geographical information, especially when considering the application of algorithms like K-means which are sensitive to the dimensionality of the feature space.

# -First, I will group the data by CustomerID and Country and calculate the number of transactions per country for each customer.
# 
# -Next, I will identify the main country for each customer (the country from which they have the maximum transactions).
# 
# -Then, I will create a binary column indicating whether the customer is from the UK or not.
# 
# -Finally, I will merge this information with the customer_data dataframe to include the new feature in our analysis.

# In[252]:


# Group by CustomerID and Country to get the number of transactions per country for each customer
customer_country = df.groupby(['CustomerID', 'Country']).size().reset_index(name='Number_of_Transactions')

# Get the country with the maximum number of transactions for each customer (in case a customer has transactions from multiple countries)
customer_main_country = customer_country.sort_values('Number_of_Transactions', ascending=False).drop_duplicates('CustomerID')

# Create a binary column indicating whether the customer is from the UK or not
customer_main_country['UK'] = customer_main_country['Country'].apply(lambda x: 1 if x == 'United Kingdom' else 0)

# Merge this data with our customer_data dataframe
customer_data = pd.merge(customer_data, customer_main_country[['CustomerID', 'UK']], on='CustomerID', how='left')

# Display the first few rows of the customer_data dataframe
customer_data.head()


# In[253]:


# Display feature distribution
customer_data['UK'].value_counts()


# #### Cancellation Insights
# 
# In this step, I am going to delve deeper into the cancellation patterns of customers to gain insights that can enhance our customer segmentation model. The features I am planning to introduce are:
# 
# -Cancellation Frequency: This metric represents the total number of transactions a customer has canceled. Understanding the frequency of cancellations can help us identify customers who are more likely to cancel transactions. This could be an indicator of dissatisfaction or other issues, and understanding this can help us tailor strategies to reduce cancellations and enhance customer satisfaction.
# 
# -Cancellation Rate: This represents the proportion of transactions that a customer has canceled out of all their transactions. This metric gives a normalized view of cancellation behavior. A high cancellation rate might be indicative of an unsatisfied customer segment. By identifying these segments, we can develop targeted strategies to improve their shopping experience and potentially reduce the cancellation rate.
# 
# By incorporating these cancellation insights into our dataset, we can build a more comprehensive view of customer behavior, which could potentially aid in creating more effective and nuanced customer segmentation.

# In[254]:


# Calculate the total number of transactions made by each customer
total_transactions = df.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()

# Calculate the number of cancelled transactions for each customer
cancelled_transactions = df[df['Transaction_Status'] == 'Cancelled']
cancellation_frequency = cancelled_transactions.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()
cancellation_frequency.rename(columns={'InvoiceNo': 'Can_Freq'}, inplace=True)

# Merge the Cancellation Frequency data into the customer_data dataframe
customer_data = pd.merge(customer_data, cancellation_frequency, on='CustomerID', how='left')

# Replace NaN values with 0 (for customers who have not cancelled any transaction)
customer_data['Can_Freq'].fillna(0, inplace=True)

# Calculate the Cancellation Rate
#customer_data['Cancellation_Rate'] = customer_data['Cancellation_Frequency'] / total_transactions['InvoiceNo']

# Display the first few rows of the customer_data dataframe
customer_data.head()


# In[255]:


# Changing the data type of 'CustomerID' to string as it is a unique identifier and not used in mathematical operations
customer_data['CustomerID'] = customer_data['CustomerID'].astype(str)

# Convert data types of columns to optimal types
customer_data = customer_data.convert_dtypes()


# In[256]:


customer_data.head(10)


# In[257]:


customer_data.info()


# #### Let's review the descriptions of the columns in our newly created customer_data dataset:

# ### Customer Dataset Description:
# 
# **CustomerID:**	Identifier uniquely assigned to each customer, used to distinguish individual customers.
# 
# **Days_Since_Last_Purchase:**	The number of days that have passed since the customer's last purchase.
# 
# **Total_Transactions:**	The total number of transactions made by the customer.
# 
# **Total_Products_Purchased:**	The total quantity of products purchased by the customer across all transactions.
# 
# **Total_Spend:**	The total amount of money the customer has spent across all transactions.
# 
# **Average_Transaction_Value:**	The average value of the customer's transactions, calculated as total spend divided by the number of transactions.
# 
# **Unique_Products_Purchased:**	The number of different products the customer has purchased.
# 
# **Average_Days_Between_Purchases:**	The average number of days between consecutive purchases made by the customer.
# 
# **Day_Of_Week:**	The day of the week when the customer prefers to shop, represented numerically (0 for Monday, 6 for Sunday).
# 
# **Hour:**	The hour of the day when the customer prefers to shop, represented in a 24-hour format.
# 
# **Is_UK:**	A binary variable indicating whether the customer is based in the UK (1) or not (0).
# 
# **Cancellation_Frequency:**	The total number of transactions that the customer has cancelled.
# 
# **Cancellation_Rate:**	The proportion of transactions that the customer has cancelled, calculated as cancellation frequency divided by total transactions.
# 
# **Monthly_Spending_Mean:**	The average monthly spending of the customer.
# 
# **Monthly_Spending_Std:**	The standard deviation of the customer's monthly spending, indicating the variability in their spending pattern.
# 
# **Spending_Tren:**	A numerical representation of the trend in the customer's spending over time. A positive value indicates an increasing trend, a negative value indicates a decreasing trend, and a value close to zero indicates a stable trend.

# Now that our dataset is ready, we can move on to the next steps of our project. This includes looking at our data more closely to find any patterns or trends, making sure our data is in the best shape by checking for and handling any outliers, and preparing our data for the clustering process. All of these steps will help us build a strong foundation for creating our customer segments.

# ### Outlier Detection & Treatment
# 
# In this section, I will identify and handle outliers in our dataset. Outliers are data points that are significantly different from the majority of other points in the dataset. These points can potentially skew the results of our analysis, especially in k-means clustering where they can significantly influence the position of the cluster centroids. Therefore, it is essential to identify and treat these outliers appropriately to achieve more accurate and meaningful clustering results.
# 
# Given the multi-dimensional nature of the data, it would be prudent to use algorithms that can detect outliers in multi-dimensional spaces. I am going to use the Isolation Forest algorithm for this task. This algorithm works well for multi-dimensional data and is computationally efficient. It isolates observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature.

# In[258]:


# Initializing the IsolationForest model with a contamination parameter of 0.05
model = IsolationForest(contamination=0.05, random_state=0)

# Fitting the model on our dataset (converting DataFrame to NumPy to avoid warning)
Outlier_Scores = model.fit_predict(customer_data.iloc[:, 1:].to_numpy())

# Creating a new column to identify outliers (1 for inliers and -1 for outliers)
customer_data['Outlier'] = [1 if x == -1 else 0 for x in Outlier_Scores]

# Display the first few rows of the customer_data dataframe
customer_data.head()


# After applying the Isolation Forest algorithm, we have identified the outliers and marked them in a new column named Is_Outlier. We have also calculated the outlier scores which represent the anomaly score of each record.
# 
# Now let's visualize the distribution of these scores and the number of inliers and outliers detected by the model:

# In[259]:


# Calculate the percentage of inliers and outliers
outlier_percentage = customer_data['Outlier'].value_counts(normalize=True) * 100

# Plotting the percentage of inliers and outliers
plt.figure(figsize=(12, 4))
outlier_percentage.plot(kind='barh', color='blue')

# Adding the percentage labels on the bars
for index, value in enumerate(outlier_percentage):
    plt.text(value, index, f'{value:.2f}%', fontsize=15)

plt.title('Inliers & Outliers Percentage')
plt.xticks(ticks=np.arange(0, 115, 5))
plt.xlabel('Percentage (%)')
plt.ylabel('Outlier')
plt.gca().invert_yaxis()
plt.show()


# From the above plot, we can observe that about 5% of the customers have been identified as outliers in our dataset. This percentage seems to be a reasonable proportion, not too high to lose a significant amount of data, and not too low to retain potentially noisy data points. It suggests that our isolation forest algorithm has worked well in identifying a moderate percentage of outliers, which will be critical in refining our customer segmentation.

# In[260]:


# Separate the outliers for analysis
outliers_data = customer_data[customer_data['Outlier'] == 1]

# Remove the outliers from the main dataset
customer_data_cleaned = customer_data[customer_data['Outlier'] == 0]

# Drop the 'Outlier_Scores' and 'Is_Outlier' columns
customer_data_cleaned = customer_data_cleaned.drop(columns=['Outlier'])

# Reset the index of the cleaned data
customer_data_cleaned.reset_index(drop=True, inplace=True)


# We have successfully separated the outliers for further analysis and cleaned our main dataset by removing these outliers. This cleaned dataset is now ready for the next steps in our customer segmentation project, which includes scaling the features and applying clustering algorithms to identify distinct customer segments.

# In[261]:


# Getting the number of rows in the cleaned customer dataset
customer_data_cleaned.shape[0]


# ### Correlation Analysis
# 
# Before we proceed to KMeans clustering, it's essential to check the correlation between features in our dataset. The presence of multicollinearity, where features are highly correlated, can potentially affect the clustering process by not allowing the model to learn the actual underlying patterns in the data, as the features do not provide unique information. This could lead to clusters that are not well-separated and meaningful.
# 
# If we identify multicollinearity, we can utilize dimensionality reduction techniques like PCA. These techniques help in neutralizing the effect of multicollinearity by transforming the correlated features into a new set of uncorrelated variables, preserving most of the original data's variance. This step not only enhances the quality of clusters formed but also makes the clustering process more computationally efficient.

# In[262]:


# Reset background style
sns.set_style('whitegrid')

# Calculate the correlation matrix excluding the 'CustomerID' column
corr = customer_data_cleaned.drop(columns=['CustomerID']).corr()

# Define a custom colormap
colors = ['#ff6200', '#ffcaa8', 'white', '#ffcaa8', '#ff6200']
my_cmap = LinearSegmentedColormap.from_list('custom_map', colors, N=256)

# Create a mask to only show the lower triangle of the matrix (since it's mirrored around its 
# top-left to bottom-right diagonal)
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask, k=1)] = True

# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr, mask=mask, cmap=my_cmap, annot=True, center=0, fmt='.2f', linewidths=2)
plt.title('Correlation Matrix', fontsize=14)
plt.show()


# Looking at the heatmap, we can see that there are some pairs of variables that have high correlations, for instance:
# 
# - `Monthly_Spending_Mean` and `Average_Transaction_Value`
#     
#     
# - `Total_Spend` and `Total_Products_Purchased`
# 
#     
# - `Total_Transactions` and `Total_Spend`
#     
#     
# - `Cancellation_Rate` and `Cancellation_Frequency`
#     
#     
# - `Total_Transactions` and `Total_Products_Purchased`
#  
#     
# These high correlations indicate that these variables move closely together, implying a degree of multicollinearity.

# Before moving to the next steps, considering the impact of multicollinearity on KMeans clustering, it might be beneficial to treat this multicollinearity possibly through dimensionality reduction techniques such as PCA to create a set of uncorrelated variables. This will help in achieving more stable clusters during the KMeans clustering process.

# ### Feature Scaling
# 
# Before we move forward with the clustering and dimensionality reduction, it's imperative to scale our features. This step holds significant importance, especially in the context of distance-based algorithms like K-means and dimensionality reduction methods like PCA. Here's why:
# 
# **-For K-means Clustering:** K-means relies heavily on the concept of 'distance' between data points to form clusters. When features are not on a similar scale, features with larger values can disproportionately influence the clustering outcome, potentially leading to incorrect groupings.
# 
# **-For PCA:** PCA aims to find the directions where the data varies the most. When features are not scaled, those with larger values might dominate these components, not accurately reflecting the underlying patterns in the data.

# Therefore, to ensure a balanced influence on the model and to reveal the true patterns in the data, I am going to standardize our data, meaning transforming the features to have a mean of 0 and a standard deviation of 1. However, not all features require scaling. Here are the exceptions and the reasons why they are excluded:
# 
# -CustomerID: This feature is just an identifier for the customers and does not contain any meaningful information for clustering.
# 
# -Is_UK: This is a binary feature indicating whether the customer is from the UK or not. Since it already takes a value of 0 or 1, scaling it won't make any significant difference.
# 
# -Day_Of_Week: This feature represents the most frequent day of the week that the customer made transactions. Since it's a categorical feature represented by integers (1 to 7), scaling it would not be necessary.
# I will proceed to scale the other features in the dataset to prepare it for PCA and K-means clustering.

# In[263]:


# Initialize the StandardScaler
scaler = StandardScaler()

# List of columns that don't need to be scaled
columns_to_exclude = ['CustomerID', 'UK', 'Fav_Day']

# List of columns that need to be scaled
columns_to_scale = customer_data_cleaned.columns.difference(columns_to_exclude)

# Copy the cleaned dataset
customer_data_scaled = customer_data_cleaned.copy()

# Applying the scaler to the necessary columns in the dataset
customer_data_scaled[columns_to_scale] = scaler.fit_transform(customer_data_scaled[columns_to_scale])

# Display the first few rows of the scaled data
customer_data_scaled.head()


# ### Dimensionality Reduction

# #### Why We Need Dimensionality Reduction? 
# 
# **Multicollinearity Detected:** In the previous steps, we identified that our dataset contains multicollinear features. Dimensionality reduction can help us remove redundant information and alleviate the multicollinearity issue.
# 
# **Better Clustering with K-means:** Since K-means is a distance-based algorithm, having a large number of features can sometimes dilute the meaningful underlying patterns in the data. By reducing the dimensionality, we can help K-means to find more compact and well-separated clusters.
# 
# **Noise Reduction:** By focusing only on the most important features, we can potentially remove noise in the data, leading to more accurate and stable clusters.
# 
# **Enhanced Visualization:** In the context of customer segmentation, being able to visualize customer groups in two or three dimensions can provide intuitive insights. Dimensionality reduction techniques can facilitate this by reducing the data to a few principal components which can be plotted easily.
# 
# **Improved Computational Efficiency:** Reducing the number of features can speed up the computation time during the modeling process, making our clustering algorithm more efficient.
# Let's proceed to select an appropriate dimensionality reduction method to our data.

# #### Which Dimensionality Reduction Method? 
# 
# In this step, we are considering the application of dimensionality reduction techniques to simplify our data while retaining the essential information. Among various methods such as KernelPCA, ICA, ISOMAP, TSNE, and UMAP, I am starting with PCA (Principal Component Analysis). Here's why:
# 
# PCA is an excellent starting point because it works well in capturing linear relationships in the data, which is particularly relevant given the multicollinearity we identified in our dataset. It allows us to reduce the number of features in our dataset while still retaining a significant amount of the information, thus making our clustering analysis potentially more accurate and interpretable. Moreover, it is computationally efficient, which means it won't significantly increase the processing time.
# 
# However, it's essential to note that we are keeping our options open. After applying PCA, if we find that the first few components do not capture a significant amount of variance, indicating a loss of vital information, we might consider exploring other non-linear methods. These methods can potentially provide a more nuanced approach to dimensionality reduction, capturing complex patterns that PCA might miss, albeit at the cost of increased computational time and complexity.

# I will apply PCA on all the available components and plot the cumulative variance explained by them. This process will allow me to visualize how much variance each additional principal component can explain, thereby helping me to pinpoint the optimal number of components to retain for the analysis:

# In[264]:


# Setting CustomerID as the index column
customer_data_scaled.set_index('CustomerID', inplace=True)

# Apply PCA
pca = PCA().fit(customer_data_scaled)

# Calculate the Cumulative Sum of the Explained Variance
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

# Set the optimal k value (based on our analysis, we can choose 3)
optimal_k = 3

# Set seaborn plot style
sns.set(rc={'axes.facecolor': '#fcf0dc'}, style='darkgrid')

# Plot the cumulative explained variance against the number of components
plt.figure(figsize=(20, 10))

# Bar chart for the explained variance of each component
barplot = sns.barplot(x=list(range(1, len(cumulative_explained_variance) + 1)),
                      y=explained_variance_ratio,
                      color='#fcc36d',
                      alpha=0.8)

# Line plot for the cumulative explained variance
lineplot, = plt.plot(range(0, len(cumulative_explained_variance)), cumulative_explained_variance,
                     marker='o', linestyle='--', color='#ff6200', linewidth=2)

# Plot optimal k value line
optimal_k_line = plt.axvline(optimal_k - 1, color='red', linestyle='--', label=f'Optimal k value = {optimal_k}') 

# Set labels and title
plt.xlabel('Number of Components', fontsize=14)
plt.ylabel('Explained Variance', fontsize=14)
plt.title('Cumulative Variance vs. Number of Components', fontsize=18)

# Customize ticks and legend
plt.xticks(range(0, len(cumulative_explained_variance)))
plt.legend(handles=[barplot.patches[0], lineplot, optimal_k_line],
           labels=['Explained Variance of Each Component', 'Cumulative Explained Variance', f'Optimal k value = {optimal_k}'],
           loc=(0.62, 0.1),
           frameon=True,
           framealpha=1.0,  
           edgecolor='#ff6200')  

# Display the variance values for both graphs on the plots
x_offset = -0.3
y_offset = 0.01
for i, (ev_ratio, cum_ev_ratio) in enumerate(zip(explained_variance_ratio, cumulative_explained_variance)):
    plt.text(i, ev_ratio, f"{ev_ratio:.2f}", ha="center", va="bottom", fontsize=10)
    if i > 0:
        plt.text(i + x_offset, cum_ev_ratio + y_offset, f"{cum_ev_ratio:.2f}", ha="center", va="bottom", fontsize=10)

plt.grid(axis='both')   
plt.show()


# #### Conclusion:
# 
# The plot and the cumulative explained variance values indicate how much of the total variance in the dataset is captured by each principal component, as well as the cumulative variance explained by the first n components.
# 
# Here, we can observe that:
# 
# -The first component explains approximately 40% of the variance.
# 
# -The first two components together explain about 34% of the variance.
# 
# -The first three components explain approximately 10% of the variance, and so on.
# 
# To choose the optimal number of components, we generally look for a point where adding another component doesn't significantly increase the cumulative explained variance, often referred to as the "elbow point" in the curve.
# 
# From the plot, we can see that the increase in cumulative variance starts to slow down after the 6th component (which captures about 81% of the total variance).
# 
# Considering the context of customer segmentation, we want to retain a sufficient amount of information to identify distinct customer groups effectively. Therefore, retaining the first 6 components might be a balanced choice, as they together explain a substantial portion of the total variance while reducing the dimensionality of the dataset.

# In[265]:


# Creating a PCA object with 3 components
pca = PCA(n_components=3)

# Fitting and transforming the original data to the new PCA dataframe
customer_data_pca = pca.fit_transform(customer_data_scaled)

# Creating a new dataframe from the PCA dataframe, with columns labeled PC1, PC2, etc.
customer_data_pca = pd.DataFrame(customer_data_pca, columns=['PC'+str(i+1) for i in range(pca.n_components_)])

# Adding the CustomerID index back to the new PCA dataframe
customer_data_pca.index = customer_data_scaled.index


# In[266]:


# Displaying the resulting dataframe based on the PCs
customer_data_pca.head()


# Now, let's extract the coefficients corresponding to each principal component to better understand the transformation performed by PCA:

# In[267]:


# Define a function to highlight the top 3 absolute values in each column of a dataframe
def highlight_top3(column):
    top3 = column.abs().nlargest(3).index
    return ['background-color:  #ffeacc' if i in top3 else '' for i in column.index]

# Create the PCA component DataFrame and apply the highlighting function
pc_df = pd.DataFrame(pca.components_.T, columns=['PC{}'.format(i+1) for i in range(pca.n_components_)],  
                     index=customer_data_scaled.columns)

pc_df.style.apply(highlight_top3, axis=0)


# ### K-Means Clustering
# 
# K-Means is an unsupervised machine learning algorithm that clusters data into a specified number of groups (K) by minimizing the within-cluster sum-of-squares (WCSS), also known as inertia. The algorithm iteratively assigns each data point to the nearest centroid, then updates the centroids by calculating the mean of all assigned points. The process repeats until convergence or a stopping criterion is reached.

# #### Determining Optimal Number of Cluster
# 
# To ascertain the optimal number of clusters (k) for segmenting customers, I will explore two renowned methods:
# 
# -Elbow Method
# 
# -Silhouette Method
# 
# It's common to utilize both methods in practice to corroborate the results.

# #### Elbow Method

# #### What is the Elbow Method?
# 
# The Elbow Method is a technique for identifying the ideal number of clusters in a dataset. It involves iterating through the data, generating clusters for various values of k. The k-means algorithm calculates the sum of squared distances between each data point and its assigned cluster centroid, known as the inertia or WCSS score. By plotting the inertia score against the k value, we create a graph that typically exhibits an elbow shape, hence the name "Elbow Method". The elbow point represents the k-value where the reduction in inertia achieved by increasing k becomes negligible, indicating the optimal stopping point for the number of clusters.

# #### Utilizing the YellowBrick Library:
# 
# In this section, I will employ the YellowBrick library to facilitate the implementation of the Elbow method. YellowBrick, an extension of the Scikit-Learn API, is renowned for its ability to rapidly generate insightful visualizations in the field of machine learning.

# In[269]:


# Set plot style, and background color
sns.set(style='darkgrid', rc={'axes.facecolor': 'lightblue'})

# Set the color palette for the plot
sns.set_palette(['red'])

# Instantiate the clustering model with the specified parameters
km = KMeans(init='k-means++', n_init=10, max_iter=100, random_state=0)

# Create a figure and axis with the desired size
fig, ax = plt.subplots(figsize=(12, 5))

# Instantiate the KElbowVisualizer with the model and range of k values, and disable the timing plot
visualizer = KElbowVisualizer(km, k=(2, 15), timings=False, ax=ax)

# Fit the data to the visualizer
visualizer.fit(customer_data_pca)

# Finalize and render the figure
visualizer.show();


# #### Optimal k Value: Elbow Method Insights:
# 
# The optimal value of k for the KMeans clustering algorithm can be found at the elbow point. Using the YellowBrick library for the Elbow method, we observe that the suggested optimal k value is 5. However, we don't have a very distinct elbow point in this case, which is common in real-world data. From the plot, we can see that the inertia continues to decrease significantly up to k=5, indicating that the optimum value of k could be between 3 and 7. To choose the best k within this range, we can employ the silhouette analysis, another cluster quality evaluation method. Additionally, incorporating business insights can help determine a practical k value.

# ### Silhouette Method

# What is the Silhouette Method?
# 
# The Silhouette Method is an approach to find the optimal number of clusters in a dataset by evaluating the consistency within clusters and their separation from other clusters. It computes the silhouette coefficient for each data point, which measures how similar a point is to its own cluster compared to other clusters.

# In[274]:


'''
def silhouette_analysis(df, start_k, stop_k, figsize=(15, 16)):
    """
    Perform Silhouette analysis for a range of k values and visualize the results.
    """

    # Set the size of the figure
    plt.figure(figsize=figsize)

    # Create a grid with (stop_k - start_k + 1) rows and 2 columns
    grid = gridspec.GridSpec(stop_k - start_k + 1, 2)

    # Assign the first plot to the first row and both columns
    first_plot = plt.subplot(grid[0, :])

    # First plot: Silhouette scores for different k values
    sns.set_palette(['red'])

    silhouette_scores = []

    # Iterate through the range of k values
    for k in range(start_k, stop_k + 1):
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=100, random_state=0)
        km.fit(df)
        labels = km.predict(df)
        score = silhouette_score(df, labels)
        silhouette_scores.append(score)

    best_k = start_k + silhouette_scores.index(max(silhouette_scores))

    plt.plot(range(start_k, stop_k + 1), silhouette_scores, marker='o')
    plt.xticks(range(start_k, stop_k + 1))
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette score')
    plt.title('Average Silhouette Score for Different k Values', fontsize=15)

    # Add the optimal k value text to the plot
    #optimal_k_text = f'The k value with the highest Silhouette score is: {best_k}'
    #plt.text(10, 0.23, optimal_k_text, fontsize=12, verticalalignment='bottom', 
             #horizontalalignment='left', bbox=dict(facecolor='#fcc36d', edgecolor='#ff6200', boxstyle='round, pad=0.5'))
             

    # Second plot (subplot): Silhouette plots for each k value
    colors = sns.color_palette("bright")

    for i in range(start_k, stop_k + 1):    
        km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=0)
        row_idx, col_idx = divmod(i - start_k, 2)

        # Assign the plots to the second, third, and fourth rows
        ax = plt.subplot(grid[row_idx + 1, col_idx])

        visualizer = SilhouetteVisualizer(km, colors=colors, ax=ax)
        visualizer.fit(df)

        # Add the Silhouette score text to the plot
        score = silhouette_score(df, km.labels_)
        ax.text(0.97, 0.02, f'Silhouette Score: {score:.2f}', fontsize=12, \
                ha='right', transform=ax.transAxes, color='black')

        ax.set_title(f'Silhouette Plot for {i} Clusters', fontsize=15)

    plt.tight_layout()
    plt.show()
    '''


# In[275]:


'''
silhouette_analysis(customer_data_pca, 3, 8, figsize=(20, 40))
'''


# In[276]:


customer_data_pca


# In[273]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np

# Function to calculate silhouette scores for different numbers of clusters using K-Means
def find_optimal_clusters_kmeans(data, max_clusters):
    silhouette_scores = []
    cluster_range = range(2, max_clusters + 1)  # Start with 2 clusters
    
    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(data)
        
        # Calculate the silhouette score for this number of clusters
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        print(f'Number of Clusters: {n_clusters}, Silhouette Score: {silhouette_avg:.3f}')
    
    # Plot silhouette scores
    plt.figure(figsize=(8, 5))
    plt.plot(cluster_range, silhouette_scores, marker='o')
    plt.title('Silhouette Scores for Different Numbers of Clusters (K-Means)')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.show()
    
    # Find the optimal number of clusters based on the highest silhouette score
    optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
    print(f'Optimal number of clusters (K-Means): {optimal_clusters}')
    return optimal_clusters

# Find optimal clusters using silhouette score with K-Means for the dataset `customer_data_pca`
optimal_clusters_kmeans = find_optimal_clusters_kmeans(customer_data_pca, max_clusters=10)


# ### Model 1: K-Means Clustering

# In[277]:


from sklearn.cluster import KMeans

# Assuming customer_data_pca is your PCA-reduced dataset.
kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=100, random_state=0)
labels = kmeans.fit_predict(customer_data_pca)

# Append the new cluster labels back to the original dataset
customer_data_cleaned['K-Means cluster'] = labels

#Append the new cluster labels to the PCA version of the dataset
customer_data_pca['K-Means cluster'] = labels


# In[278]:


# Display the first few rows of the original dataframe
customer_data_cleaned.head()


# ### Clustering Evaluation

# ### 3D Visualization of Top Principal Component 

# In[279]:


# Setting up the color scheme for the clusters (RGB order)
colors = ['red', 'green', 'blue']


# In[280]:


# Create separate data frames for each cluster
cluster_0 = customer_data_pca[customer_data_pca['K-Means cluster'] == 0]
cluster_1 = customer_data_pca[customer_data_pca['K-Means cluster'] == 1]
cluster_2 = customer_data_pca[customer_data_pca['K-Means cluster'] == 2]

# Create a 3D scatter plot
fig = go.Figure()

# Add data points for each cluster separately and specify the color
fig.add_trace(go.Scatter3d(x=cluster_0['PC1'], y=cluster_0['PC2'], z=cluster_0['PC3'], 
                           mode='markers', marker=dict(color=colors[0], size=5, opacity=0.4), name='Cluster 0'))
fig.add_trace(go.Scatter3d(x=cluster_1['PC1'], y=cluster_1['PC2'], z=cluster_1['PC3'], 
                           mode='markers', marker=dict(color=colors[1], size=5, opacity=0.4), name='Cluster 1'))
fig.add_trace(go.Scatter3d(x=cluster_2['PC1'], y=cluster_2['PC2'], z=cluster_2['PC3'], 
                           mode='markers', marker=dict(color=colors[2], size=5, opacity=0.4), name='Cluster 2'))

# Set the title and layout details
fig.update_layout(
    title=dict(text='3D Visualization of Customer Clusters in PCA Space', x=0.5),
    scene=dict(
        xaxis=dict(backgroundcolor="#fcf0dc", gridcolor='white', title='PC1'),
        yaxis=dict(backgroundcolor="#fcf0dc", gridcolor='white', title='PC2'),
        zaxis=dict(backgroundcolor="#fcf0dc", gridcolor='white', title='PC3'),
    ),
    width=900,
    height=800
)

# Show the plot
fig.show()


# ### Clustering Distribution Visualization

# In[281]:


# Calculate the percentage of customers in each cluster
cluster_percentage = (customer_data_pca['K-Means cluster'].value_counts(normalize=True) * 100).reset_index()
cluster_percentage.columns = ['Cluster', 'Percentage']
cluster_percentage.sort_values(by='Cluster', inplace=True)

# Create a horizontal bar plot
plt.figure(figsize=(10, 4))
sns.barplot(x='Percentage', y='Cluster', data=cluster_percentage, orient='h', palette = colors)

# Adding percentages on the bars
for index, value in enumerate(cluster_percentage['Percentage']):
    plt.text(value+0.5, index, f'{value:.2f}%')

plt.title('Distribution of Customers Across Clusters', fontsize=14)
plt.xticks(ticks=np.arange(0, 50, 5))
plt.xlabel('Percentage (%)')

# Show the plot
plt.show()


# ### Evaluation Metrics
# 
# To further scrutinize the quality of our clustering, I will employ the following metrics:
# 
# **-Silhouette Score:** A measure to evaluate the separation distance between the clusters. Higher values indicate better cluster separation. It ranges from -1 to 1.
# 
# **-Calinski Harabasz Score:** This score is used to evaluate the dispersion between and within clusters. A higher score indicates better defined clusters.
# 
# **-Davies Bouldin Score:** It assesses the average similarity between each cluster and its most similar cluster. Lower values indicate better cluster separation.

# In[282]:


# Compute number of customers
num_observations = len(customer_data_pca)

# Separate the features and the cluster labels
X = customer_data_pca.drop('K-Means cluster', axis=1)
clusters = customer_data_pca['K-Means cluster']

# Compute the metrics
sil_score = silhouette_score(X, clusters)
calinski_score = calinski_harabasz_score(X, clusters)
davies_score = davies_bouldin_score(X, clusters)

# Create a table to display the metrics and the number of observations
table_data = [
    ["Number of Observations", num_observations],
    ["Silhouette Score", sil_score],
    ["Calinski Harabasz Score", calinski_score],
    ["Davies Bouldin Score", davies_score]
]

# Print the table
print(tabulate(table_data, headers=["Metric", "Value"], tablefmt='pretty'))


# ### Clustering Analysis & Profiling

# In[283]:


# Setting 'CustomerID' column as index and assigning it to a new dataframe
df_customer = customer_data_cleaned.set_index('CustomerID')

# Standardize the data (excluding the cluster column)
scaler = StandardScaler()
df_customer_standardized = scaler.fit_transform(df_customer.drop(columns=['K-Means cluster'], axis=1))

# Create a new dataframe with standardized values and add the cluster column back
df_customer_standardized = pd.DataFrame(df_customer_standardized, columns=df_customer.columns[:-1], index=df_customer.index)
df_customer_standardized['K-Means cluster'] = df_customer['K-Means cluster']

# Calculate the centroids of each cluster
cluster_centroids = df_customer_standardized.groupby('K-Means cluster').mean()

# Function to create a radar chart
def create_radar_chart(ax, angles, data, color, cluster):
    # Plot the data and fill the area
    ax.fill(angles, data, color=color, alpha=0.4)
    ax.plot(angles, data, color=color, linewidth=2, linestyle='solid')
    
    # Add a title
    ax.set_title(f'Cluster {cluster}', size=20, color=color, y=1.1)

# Set data
labels=np.array(cluster_centroids.columns)
num_vars = len(labels)

# Compute angle of each axis
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

# The plot is circular, so we need to "complete the loop" and append the start to the end
labels = np.concatenate((labels, [labels[0]]))
angles += angles[:1]

# Initialize the figure
fig, ax = plt.subplots(figsize=(22, 22), subplot_kw=dict(polar=True), nrows=1, ncols=3)

# Create radar chart for each cluster
for i, color in enumerate(colors):
    data = cluster_centroids.loc[i].tolist()
    data += data[:1]  # Complete the loop
    create_radar_chart(ax[i], angles, data, color, i)

# Add input data
ax[0].set_xticks(angles[:-1])
ax[0].set_xticklabels(labels[:-1])

ax[1].set_xticks(angles[:-1])
ax[1].set_xticklabels(labels[:-1])

ax[2].set_xticks(angles[:-1])
ax[2].set_xticklabels(labels[:-1])

# Add a grid
ax[0].grid(color='grey', linewidth=0.5)

# Display the plot
plt.tight_layout()
plt.show()


# #### Customer Profiles Derived from Radar Chart Analysis
# 
# #### Cluster 0 (Red Chart):
# 
# Profile: Sporadic Shoppers with a Preference for Weekend Shopping
# 
# -Customers in this cluster tend to spend less, with a lower number of transactions and products purchased.
# 
# -They have a slight tendency to shop during the weekends, as indicated by the very high Day_of_Week value.
# 
# -Their spending trend is relatively stable but on the lower side, and they have a low monthly spending variation (low Monthly_Spending_Std).
# 
# #### Cluster 1 (Green)
# 
# Profile: Infrequent Big Spenders with a High Spending Trend
# 
# -These customers prefer shopping late in the day, as indicated by the high Hour value, and they mainly reside in the UK.
# 
# -They have a very high spending trend, indicating that their spending has been increasing over time.
# 
# #### Cluster 2 (Blue)
# 
# Profile: Frequent High-Spenders with a High Rate of Cancellations
# 
# -Customers in this cluster are high spenders with a very high total spend, and they purchase a wide variety of unique products.
# 
# -They engage in frequent transactions, but also have a high cancellation frequency and rate.

# ### Hierarchical clustering

# In[268]:


import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt

# Plotting the dendrogram for hierarchical clustering
plt.figure(figsize=(10, 7))

# Use the linkage method to perform hierarchical clustering
# You can choose 'ward', 'single', 'complete', or 'average' as linkage methods.
dend = shc.dendrogram(shc.linkage(customer_data_pca, method='ward')) 

plt.title("Dendrogram for Hierarchical Clustering")
plt.xlabel("Customer Data Points")
plt.ylabel("Euclidean distances")
plt.show()


# In[284]:


import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

# Function to calculate silhouette scores for different numbers of clusters
def find_optimal_clusters(data, max_clusters):
    silhouette_scores = []
    cluster_range = range(2, max_clusters + 1)  # Start with 2 clusters
    
    for n_clusters in cluster_range:
        clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
        cluster_labels = clustering.fit_predict(data)
        
        # Calculate the silhouette score for this number of clusters
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        print(f'Number of Clusters: {n_clusters}, Silhouette Score: {silhouette_avg:.3f}')
    
    # Plot silhouette scores
    plt.figure(figsize=(8, 5))
    plt.plot(cluster_range, silhouette_scores, marker='o')
    plt.title('Silhouette Scores for Different Numbers of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.show()
    
    # Find the optimal number of clusters based on the highest silhouette score
    optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
    print(f'Optimal number of clusters: {optimal_clusters}')
    return optimal_clusters

# Find optimal clusters using silhouette score with your dataset `customer_data_pca`
optimal_clusters = find_optimal_clusters(customer_data_pca.drop('K-Means cluster',axis=1), max_clusters=10)


# In[287]:


customer_data_pca


# ### Model 2: Hierarchical Clustering

# In[288]:


from sklearn.cluster import AgglomerativeClustering

# Applying hierarchical clustering
# n_clusters is the number of clusters you want to create
# linkage can be 'ward', 'complete', 'average', or 'single'
hc = AgglomerativeClustering(n_clusters=3, linkage='ward')
label1 = hc.fit_predict(customer_data_pca.drop(['K-Means cluster'], axis=1))

customer_data_pca['Hierarchical cluster'] = label1

# Append the new cluster labels back to the original dataset
customer_data_cleaned['Hierarchical cluster'] = label1


# In[289]:


customer_data_pca


# In[290]:


colors = ['red','green','blue']

# Create separate data frames for each cluster
cluster_0 = customer_data_pca[customer_data_pca['Hierarchical cluster'] == 0]
cluster_1 = customer_data_pca[customer_data_pca['Hierarchical cluster'] == 1]
cluster_2 = customer_data_pca[customer_data_pca['Hierarchical cluster'] == 2]

# Create a 3D scatter plot
fig = go.Figure()

# Add data points for each cluster separately and specify the color
fig.add_trace(go.Scatter3d(x=cluster_0['PC1'], y=cluster_0['PC2'], z=cluster_0['PC3'], 
                           mode='markers', marker=dict(color=colors[0], size=5, opacity=0.4), name='Cluster 0'))
fig.add_trace(go.Scatter3d(x=cluster_1['PC1'], y=cluster_1['PC2'], z=cluster_1['PC3'], 
                           mode='markers', marker=dict(color=colors[1], size=5, opacity=0.4), name='Cluster 1'))
fig.add_trace(go.Scatter3d(x=cluster_2['PC1'], y=cluster_2['PC2'], z=cluster_2['PC3'], 
                           mode='markers', marker=dict(color=colors[2], size=5, opacity=0.4), name='Cluster 2'))


# Set the title and layout details
fig.update_layout(
    title=dict(text='3D Visualization of Customer Clusters in PCA Space', x=0.5),
    scene=dict(
        xaxis=dict(backgroundcolor="#fcf0dc", gridcolor='white', title='PC1'),
        yaxis=dict(backgroundcolor="#fcf0dc", gridcolor='white', title='PC2'),
        zaxis=dict(backgroundcolor="#fcf0dc", gridcolor='white', title='PC3'),
    ),
    width=900,
    height=800
)

# Show the plot
fig.show()


# In[291]:


# Calculate the percentage of customers in each cluster
cluster_percentage = (customer_data_pca['Hierarchical cluster'].value_counts(normalize=True) * 100).reset_index()
cluster_percentage.columns = ['Cluster', 'Percentage']
cluster_percentage.sort_values(by='Cluster', inplace=True)

# Create a horizontal bar plot
plt.figure(figsize=(13, 4))
sns.barplot(x='Percentage', y='Cluster', data=cluster_percentage, orient='h', palette=colors)

# Adding percentages on the bars
for index, value in enumerate(cluster_percentage['Percentage']):
    plt.text(value+0.5, index, f'{value:.2f}%')

plt.title('Distribution of Customers Across Clusters', fontsize=14)
plt.xticks(ticks=np.arange(0, 50, 5))
plt.xlabel('Percentage (%)')

# Show the plot
plt.show()


# In[292]:


# Compute number of customers
num_observations = len(customer_data_pca)

# Separate the features and the cluster labels
W = customer_data_pca.drop(['K-Means cluster','Hierarchical cluster'], axis=1)
cluster = customer_data_pca['Hierarchical cluster']

# Compute the metrics
sil_score = silhouette_score(W, cluster)
calinski_score = calinski_harabasz_score(W, cluster)
davies_score = davies_bouldin_score(W, cluster)

# Create a table to display the metrics and the number of observations
table_data = [
    ["Number of Observations", num_observations],
    ["Silhouette Score", sil_score],
    ["Calinski Harabasz Score", calinski_score],
    ["Davies Bouldin Score", davies_score]
]

# Print the table
print(tabulate(table_data, headers=["Metric", "Value"], tablefmt='pretty'))


# In[293]:


# Setting 'CustomerID' column as index and assigning it to a new dataframe
df_customer = customer_data_cleaned.set_index('CustomerID')

# Standardize the data (excluding the cluster column)
scaler = StandardScaler()
df_customer_standardized = scaler.fit_transform(df_customer.drop(columns=['K-Means cluster','Hierarchical cluster'], axis=1))

# Create a new dataframe with standardized values and add the cluster column back
df_customer_standardized = pd.DataFrame(df_customer_standardized, columns=df_customer.columns[:-2], index=df_customer.index)
df_customer_standardized['Hierarchical cluster'] = df_customer['Hierarchical cluster']

# Calculate the centroids of each cluster
cluster_centroids = df_customer_standardized.groupby('Hierarchical cluster').mean()

# Function to create a radar chart
def create_radar_chart(ax, angles, data, color, cluster):
    # Plot the data and fill the area
    ax.fill(angles, data, color=color, alpha=0.4)
    ax.plot(angles, data, color=color, linewidth=2, linestyle='solid')
    
    # Add a title
    ax.set_title(f'Cluster {cluster}', size=20, color=color, y=1.1)

# Set data
labels=np.array(cluster_centroids.columns)
num_vars = len(labels)

# Compute angle of each axis
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

# The plot is circular, so we need to "complete the loop" and append the start to the end
labels = np.concatenate((labels, [labels[0]]))
angles += angles[:1]

# Initialize the figure
fig, ax = plt.subplots(figsize=(22, 22), subplot_kw=dict(polar=True), nrows=1, ncols=3)

# Create radar chart for each cluster
for i, color in enumerate(colors):
    data = cluster_centroids.loc[i].tolist()
    data += data[:1]  # Complete the loop
    create_radar_chart(ax[i], angles, data, color, i)

# Add input data
ax[0].set_xticks(angles[:-1])
ax[0].set_xticklabels(labels[:-1])

ax[1].set_xticks(angles[:-1])
ax[1].set_xticklabels(labels[:-1])

ax[2].set_xticks(angles[:-1])
ax[2].set_xticklabels(labels[:-1])

# Add a grid
ax[0].grid(color='grey', linewidth=0.5)

# Display the plot
plt.tight_layout()
plt.show()


# ### Comparison

# In[294]:


import plotly.graph_objects as go

# Data
metrics = ['Silhouette Score', 'Calinski Harabasz Score', 'Davies Bouldin Score']
kmeans_values = [0.36818690061031567, 2591.5130514023303, 0.9540154660027369]
hierarchical_values = [0.33894413472782053, 1884.7583724526405, 1.096599125185812]

fig = go.Figure()

# Add bars for Silhouette and Davies Bouldin Scores on the primary axis (left)
fig.add_trace(go.Bar(
    x=['Silhouette Score', 'Davies Bouldin Score'],
    y=[kmeans_values[0], kmeans_values[2]],
    name='K-Means (Silhouette/Davies)',
    marker_color='blue',
    text=[f'{v:.2f}' for v in [kmeans_values[0], kmeans_values[2]]],  # Text annotation for K-Means
    textposition='auto'
))

fig.add_trace(go.Bar(
    x=['Silhouette Score', 'Davies Bouldin Score'],
    y=[hierarchical_values[0], hierarchical_values[2]],
    name='Hierarchical (Silhouette/Davies)',
    marker_color='red',
    text=[f'{v:.2f}' for v in [hierarchical_values[0], hierarchical_values[2]]],  # Text annotation for Hierarchical
    textposition='auto'
))

# Add bars for Calinski Harabasz Score on the secondary axis (right)
fig.add_trace(go.Bar(
    x=['Calinski Harabasz Score'],
    y=[kmeans_values[1]],
    name='K-Means (Calinski)',
    marker_color='blue',
    text=[f'{kmeans_values[1]:.2f}'],  # Text annotation for K-Means (Calinski)
    textposition='auto',
    yaxis='y2'  # Attach to secondary axis
))

fig.add_trace(go.Bar(
    x=['Calinski Harabasz Score'],
    y=[hierarchical_values[1]],
    name='Hierarchical (Calinski)',
    marker_color='red',
    text=[f'{hierarchical_values[1]:.2f}'],  # Text annotation for Hierarchical (Calinski)
    textposition='auto',
    yaxis='y2'  # Attach to secondary axis
))

# Update layout for dual y-axis
fig.update_layout(
    title='Comparison of K-Means & Hierarchical Clustering',
    xaxis_title='Metric',
    
    # Y-axis for Silhouette and Davies Bouldin Scores (left side)
    yaxis=dict(
        title='Score (Silhouette/Davies Bouldin)',
        titlefont=dict(color='blue'),
        tickfont=dict(color='blue'),
    ),
    
    # Secondary y-axis for Calinski Harabasz Score (right side)
    yaxis2=dict(
        title='Score (Calinski Harabasz)',
        titlefont=dict(color='red'),
        tickfont=dict(color='red'),
        overlaying='y',
        side='right'
    ),
    
    barmode='group',
    legend=dict(x=0.1, y=1.1)
)

# Show plot
fig.show()

