#!/usr/bin/env python
# coding: utf-8

# # US Accidents Exploratory Data Analysis Project
# 
# In this Project, We are going to Explore the countrywide car accident dataset of the US. The accident data are collected from February 2016 to Dec 2020, there are about `3 million accident` records in this dataset. We are going to analyse the data to explore various questions like Hotspot locations of the Accidents, What time of the day is the frequency higher? and the impact of environmental stimuli on accident occurrence.
# 

# ## 1. Import data and libraries

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


sns.set_style('darkgrid')


# In[3]:


data = 'F:\\US_Accidents_Dec20_Updated.csv'


# In[4]:


df = pd.read_csv(data)


# In[5]:


df


# ## 2. Data Preparation & Cleaning
# 
# Before We get to exploring the data, first and foremost we should prepare the data for the analysis. We'll first do data cleaning. we'll check for the null values and remove all the columns with a lot of null values. Also we'll imput appropriate values for the required columns for our analysis then We'll do memory optimzation since our data is too large.

# In[6]:


df.info()


# Now let's see some statistics of the data with Panda's `.describe()` method.

# In[7]:


df.describe()


# We'll first see how many null values are there in the dataset. We'll drop the columns containing large number of null values since they won't be much useful. We'll also get rid of few of the columns which aren't too important.

# In[8]:


df.isna()


# In[9]:


df.isna().sum()


# So we can see there some columns which has quiet a lot of missing values.
# 
# Now we will see the percentage of the missing values per column.

# In[5]:


missing_percent = df.isna().sum().sort_values(ascending=False)*100/len(df)
missing_percent


# In[37]:


missing_percent[missing_percent != 0].plot(kind='barh')


# In the plot we can see the maximum number of NA containing columns, we will get rid of those and some useless columns also.

# In[12]:


df.drop(columns = ['Precipitation(in)', 'Wind_Speed(mph)', 'End_Lat', 
                   'End_Lng','Civil_Twilight', 'Nautical_Twilight', 
                   'Wind_Chill(F)', 'Astronomical_Twilight', 
                   'Wind_Direction', 'Pressure(in)', 'Weather_Timestamp', 
                   'Airport_Code', 'Timezone'], axis= 'columns', inplace=True)


# In[13]:


df.drop('Number', axis='columns', inplace=True)


# In[14]:


df.drop('Zipcode', axis='columns', inplace=True)


# In[15]:


df.columns


# In[16]:


df.isna().sum()


# After droping the useless columns these are the columns which are important.

# Now we will impute the missing values

# In[17]:


# Filling NA values for the numeric columns
df['Temperature(F)'].fillna(df['Temperature(F)'].median(), inplace=True)
df['Humidity(%)'].fillna(df['Humidity(%)'].median(), inplace=True)
df['Visibility(mi)'].fillna(df['Visibility(mi)'].median(), inplace=True)

# Filling NA values for the categorical columns
df['Weather_Condition'].fillna(df['Weather_Condition'].mode()[0], inplace=True)
df['Sunrise_Sunset'].fillna(df['Sunrise_Sunset'].mode()[0], inplace=True)
df['City'].fillna(value = 'None', inplace = True)


# In[18]:


df.isna().sum()


# So as we can see all the missing value are filled now we can explore the dataset, create visualization and make decision

# In[19]:


df.head(3).transpose()


# In[20]:


convert_columns = ['Start_Time', 'End_Time']
df[convert_columns] = df[convert_columns].astype('datetime64[ns]')
df.info()


# Here we converted the datatype of `Start_Time`, `End_Time` from `object` to `datetime`

# ## Exploratory analysis and Visualization
# 
# In this, We'll analyse each column of our dataset excluding some which don't impact or have any meaningful insights whatsoever. There are many columns worth exploring like `State`, `City`, `Street`, `County`, `Start_Time`, `Temperature(F)`, `Weather_Condition`, `Visibility(mi)`. We'll gain many insights and will try to answer a lot of questions about the dataset.
# 
# ### A. City

# In[21]:


cities = df.City.unique()
cities


# In[22]:


len(cities)


# In[23]:


cities = df["City"].value_counts(ascending = False).reset_index()
cities.columns = ["City", "Number_of_Accidents"]
cities["% of_Accidents"] =(cities["Number_of_Accidents"]*100)/len(df)
cities.sort_values(by = "Number_of_Accidents",ascending = False, inplace = True)
cities_accidents = cities.head(50)
cities_accidents.head()


# In[38]:


plt.figure(figsize=(15,20))
plt.xticks(rotation = 90)
plt.title("Accident by Cities (Top 50)",fontsize= 20)
sns.barplot(y = "City", x = "Number_of_Accidents", data = cities_accidents )


# In[25]:


cities["% of_Accidents"].head(1000).sum()


# In[26]:


cities["% of_Accidents"].head(100).sum()


# In[27]:


cities["% of_Accidents"].head(10).sum()


# We have 11790 cities. Of that `Top 1000 cities` account for `80%` of the Accidents, `Top 100 cities` account for `44%` of the Accidents and `Top 10 cities` account for `15%` of the Accidents.

# ### B. Start_Time

# In[28]:


df.Start_Time


# In[29]:


df["Year"] = df["Start_Time"].dt.year


# In[39]:


plt.figure(figsize = (15,10))
df["Start_Time"].dt.year.value_counts().plot(kind = "line")
plt.title("Yearly Accidents Trend", fontsize = 15)


# In[31]:


plt.figure(figsize = (25,10))
colors = ['#c2c2f0','#ffb3e6', '#99ff99', '#66b3ff', '#ffcc99']
(df["Start_Time"].dt.year.value_counts(ascending = True)*100/len(df)).plot(kind = "pie", autopct = "%1.1f%%", colors = colors)
plt.title("Percentage of yearly Accidents", fontsize = 20)


# #### Out of all the accident records `35.6%` of accidents have happened in 2020. Accidents are increasing at an alarming rate every year.

# In[40]:


sns.distplot(df.Start_Time.dt.hour, bins=24, kde=False, norm_hist=True)


# - A high percentage of accidents occur between 6 am to 10 am (probably people in a hurry to get to work)
# - Next higest percentage is 3 pm to 6 pm.

# In[41]:


sns.distplot(df.Start_Time.dt.dayofweek, bins=7, kde=False, norm_hist=True)


# This is the distribution of accidents by hour the same on weekends as on weekdays.

# In[42]:


sundays_start_time = df.Start_Time[df.Start_Time.dt.dayofweek == 6]
sns.distplot(sundays_start_time.dt.hour, bins=24, kde=False, norm_hist=True)


# - On Sundays, the peak occurs between 10 am and 3 pm, unlike weekdays

# In[43]:


monday_start_time = df.Start_Time[df.Start_Time.dt.dayofweek == 0]
sns.distplot(monday_start_time.dt.hour, bins=24, kde=False, norm_hist=True)


# - On weekdays again the time duration is between `6am to 5pm`

# Let's see the month wise data

# In[44]:


df["Month"] = df["Start_Time"].dt.month_name()


# In[45]:


plt.figure(figsize = (20,10))
(df["Start_Time"].dt.month_name().value_counts(ascending = True)*100/len(df)).plot(kind = "bar", color = "m")
plt.title("Percentage of Monthly Accidents", fontsize = 20)


# - As we can see in the month of `December` we have the highest number of accidents and `July` has the least number of accidents.

# ### C. Severity

# In[46]:


plt.figure(figsize = (25,10))
df["Severity"].value_counts().plot(kind = "pie", autopct = "%1.1f%%", colors = ('#c2c2f0','#ffb3e6', '#99ff99', '#66b3ff' ))
plt.title("Percentage of Severity of Accidents", fontsize = 20)


# - 73% reported accidents have Severity 2 which could mean that there are a lot of accidents which caused some injuries and had little impact.

# In[47]:


plt.figure(figsize = (25,10))
sns.countplot(x = "Severity", hue = "Year", data = df)


# - It seems there are little to no records of Severity 1. Year 2020 had the most number of Severity 2 Accidents though it doesn't seem to be the case in Severity 3 and 4 which is an interesting find.

# ### D. Weather_Condition

# In[5]:


weather = df["Weather_Condition"].value_counts().reset_index()
weather.columns = ["Weather", "Number_of_Accidents"]
weather["% of_Accidents"] =(weather["Number_of_Accidents"]*100)/len(df)
weather.sort_values(by = "Number_of_Accidents",ascending = False, inplace = True)
weather_condition = weather.head(10)
weather_condition.head()


# In[8]:


plt.rcParams["figure.figsize"] = (10,8)
weather_condition.plot(x = "Weather", y = "% of_Accidents", kind = "bar")
plt.title("Accidents by Weather Condition (Top 10)", fontsize = 20)
plt.xticks(rotation = 90)
plt.show()


# In[50]:


weather_condition["% of_Accidents"].head(6).sum()


# - "Fair" weather condition has the large number of accidents i.e 26% of the accidents. Clear and Mostly Cloudy also have 17% and 13% respectively. Also for Partly Cloudy has 9%, Cloudy has 8%, Overcast has 8% accidents. These top 6 Weather conditions amounts to `82.6%` of total accidents.

# In[10]:


group = df.groupby(["Weather_Condition", "Severity"])["Severity"].count().sort_values(ascending = False).unstack("Weather_Condition")


# In[11]:


weather_severity = group[["Fair","Clear", "Mostly Cloudy", "Partly Cloudy", "Cloudy", "Overcast"]].unstack()


# In[14]:


plt.figure(figsize = (30,30))
colors = ('lightblue', "beige", "cyan", 'lightsteelblue')
explode = (0, 0, 0.1, 0)

# Fair weather
plt.subplot(2,3,1)
weather_severity.loc["Fair"].sort_values().plot(kind = "pie",autopct = "%1.1f%%", textprops={'fontsize': 20}, colors =colors, explode = explode)
plt.title("Fair", fontsize = 30)
plt.ylabel("")

# Clear weather
plt.subplot(2,3,2)
weather_severity.loc["Clear"].sort_values().plot(kind = "pie",autopct = "%1.1f%%", textprops={'fontsize': 20}, colors = colors, explode = explode)
plt.title("Clear", fontsize = 30)
plt.ylabel("")

# Mosty cloudy weather
plt.subplot(2,3,3)
weather_severity.loc["Mostly Cloudy"].sort_values().plot(kind = "pie",autopct = "%1.1f%%", textprops={'fontsize': 20}, colors = colors, explode = explode)
plt.title("Mostly Cloudy", fontsize = 30)
plt.ylabel("")

#PArtly cloudy weather
plt.subplot(2,3,4)
weather_severity.loc["Partly Cloudy"].sort_values().plot(kind = "pie",autopct = "%1.1f%%", textprops={'fontsize': 20}, colors = colors, explode = explode)
plt.title("Partly Cloudy", fontsize = 30)
plt.ylabel("")

# Cloudy weather
plt.subplot(2,3,5)
weather_severity.loc["Cloudy"].sort_values().plot(kind = "pie",autopct = "%1.1f%%", textprops={'fontsize': 20}, colors = colors, explode = explode)
plt.title("Cloudy", fontsize = 30)
plt.ylabel("")

# Overcast weather
plt.subplot(2,3,6)
weather_severity.loc["Overcast"].sort_values().plot(kind = "pie",autopct = "%1.1f%%", textprops={'fontsize': 20}, colors = colors, explode = explode)
plt.title("Overcast", fontsize = 30)
plt.ylabel("")


# - It seems all six Weather Conditions has most accidents happened in Severity 2 i.e above 65%. Clear and Overcast Weather had no Severity 1 accidents.

# In[55]:


group1 = df.groupby(["Weather_Condition", "Year"])["Year"].count().sort_values(ascending = False).unstack("Weather_Condition")


# In[56]:


weather_year = group1[["Fair","Clear", "Mostly Cloudy", "Partly Cloudy", "Cloudy", "Overcast"]].unstack()


# In[65]:


plt.figure(figsize = (30,30))
explode = (0,0, 0, 0.1, 0)
colors = ['#c2c2f0', '#ffcc99', '#99ff99', '#66b3ff','#ff6666']

plt.subplot(2,3,1)
weather_year.loc["Fair"].sort_values().plot(kind = "pie",autopct = "%1.1f%%", textprops={'fontsize': 20}, colors = colors, explode = explode)
plt.title("Fair", fontsize = 30)
plt.ylabel("")

plt.subplot(2,3,2)
weather_year.loc["Clear"].sort_values().plot(kind = "pie",autopct = "%1.1f%%", textprops={'fontsize': 20}, colors = colors, explode = explode)
plt.title("Clear", fontsize = 30)
plt.ylabel("")

plt.subplot(2,3,3)
weather_year.loc["Mostly Cloudy"].sort_values().plot(kind = "pie",autopct = "%1.1f%%", textprops={'fontsize': 20}, colors = colors, explode = explode)
plt.title("Mostly Cloudy", fontsize = 30)
plt.ylabel("")

plt.subplot(2,3,4)
weather_year.loc["Partly Cloudy"].sort_values().plot(kind = "pie",autopct = "%1.1f%%", textprops={'fontsize': 20}, colors = colors, explode = explode)
plt.title("Partly Cloudy", fontsize = 30)
plt.ylabel("")

plt.subplot(2,3,5)
weather_year.loc["Cloudy"].sort_values().plot(kind = "pie",autopct = "%1.1f%%", textprops={'fontsize': 20}, colors = colors, explode = explode)
plt.title("Cloudy", fontsize = 30)
plt.ylabel("")

plt.subplot(2,3,6)
weather_year.loc["Overcast"].sort_values().plot(kind = "pie",autopct = "%1.1f%%", textprops={'fontsize': 20}, colors = colors, explode = explode)
plt.title("Overcast", fontsize = 30)
plt.ylabel("")


# - Overcast and Clear weather has no accident records for 2020. This could be an error while collecting data since 2020 recorded most accidents overall. Fair and Cloudy weather conditions had more than 65% accidents happen in 2020.

# In[ ]:




