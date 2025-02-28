# %%
import pandas as pd
#load data set
data = pd.read_csv('heart-disease.csv')

# %%
data

# %%


# %%
print(data)

# %%
print(data.head(10))

# %%
print(data.tail(5))

# %%
print(data.describe())

# %%
data.info()

# %%
import numpy as np
#importing numpy
#ckecking for missing values
print(data.isnull().sum())

# %%
#replacing missing values with mean
data['chol'] = data['chol'].replace(np.nan, data['chol'].mean())


# %%
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# %%
#initializing the plot
#initializing the x and y points
xpoints = np.array([0, 6])
ypoints = np.array([0, 251])
#drawing the plot
plt.plot(xpoints, ypoints)
plt.show()

# %%
xpoints = np.array([0,5,6,9,3,6])
ypoints = np.array([0,3,8,1,10,251])
plt.plot(xpoints, ypoints)
plt.show()

# %%
#plotting a pie chart
y = np.array([35, 25, 25, 15])
plt.pie(y)
plt.show()

# %%
x = np.random.normal(170, 10, 250)  
#mean, standard deviation, number of values
print(x)
print("mean",np.mean(x))
print("std",np.std(x))
print("var",np.var(x))
#finding no of values in the array
print("size",np.size(x))


# %%
#plotting a histogram
plt.hist(x, bins=5) #bins is the number of bars in the histogram

# %%
sns.displot([0,1,2,3,4,5])
plt.show()

# %%
sns.histplot(data['age'], kde=True) #kde is the kernel density estimation

# %%
#seaborn graphs
sns.histplot(x, kde=True) #kde is the line on the histogram


# %%

from sklearn.model_selection import train_test_split 
#importing the train_test_split function to split the data into training and testing data
from sklearn.ensemble import RandomForestRegressor
#importing the RandomForestRegressor model to train the model
#RandomForestRegressor is used for regression problems
#Regressor is used for continuous values
#Classifier is used for discrete values
#Regression problems are used to predict continuous values example: predicting the price of a house using the area of the house
#Classification problems are used to predict discrete values example: predicting if a person has a heart disease or not
from sklearn.metrics import mean_squared_error, r2_score


# %%
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# %%
#data preprocessing (cleaning the data)
#loading the data
import pandas as pd
apps_df = pd.read_csv('Play Store Data.csv')
reviews_df = pd.read_csv('User Reviews.csv')

# %%
apps_df.head()

# %%
reviews_df.head(5)#displaying the first 5 rows of the data
#pd.read_csv
#pd.read_excel can be used to read data from an excel file
#pd.read_json can be used to read data from a json file
#pd.read_sql can be used to read data from a database
#pd.read_html can be used to read data from a html file


# %%
#to check for missing values
apps_df.isnull().sum()
#.dropna() is used to drop the missing values
#.fillna() is used to fill the missing values
#to print all the columns
apps_df.columns



# %%
#for duplicate values in the data
apps_df.duplicated().sum()
#to drop the duplicate values
#apps_def.drop_duplicates()
# Step 2: Data Cleaning
apps_df = apps_df.dropna(subset=['Rating'])
for column in apps_df.columns:
    apps_df[column].fillna(apps_df[column].mode()[0], inplace=True)
apps_df.drop_duplicates(inplace=True)
apps_df = apps_df[apps_df['Rating'] <= 5]
reviews_df.dropna(subset=['Translated_Review'], inplace=True)

# %%
import pandas as pd

# Assuming apps_df and reviews_df are already defined

# Step 1: Merge DataFrames
merged_df = pd.merge(apps_df, reviews_df, on='App', how='inner')

# Step 2: Data Transformation
# Ensure 'Reviews' column is of type string before converting to int
apps_df['Reviews'] = apps_df['Reviews'].astype(str).astype(int)

# Ensure 'Installs' column is of type string before replacing characters and converting to float
apps_df['Installs'] = apps_df['Installs'].astype(str).str.replace(',', '').str.replace('+', '').astype(float)

# Ensure 'Price' column is of type string before replacing characters and converting to float
apps_df['Price'] = apps_df['Price'].astype(str).str.replace('$', '').astype(float)

# Check the data types of the columns
print(apps_df.dtypes)

# %%
merged_df.head()

# %%
#data transformation



# %%

def convert_size(size):
    if 'M' in size:
        return float(size.replace('M',''))
    elif 'k' in size:
        return float(size.replace('k',''))/1024
    else:
        return np.nan
apps_df['Size']=apps_df['Size'].apply(convert_size)

# %%
#Lograrithmic
apps_df['Log_Installs']=np.log(apps_df['Installs'])

# %%
apps_df['Reviews']=apps_df['Reviews'].astype(int)

# %%
apps_df['Log_Reviews']=np.log(apps_df['Reviews'])

# %%

reviews_df.dtypes

# %%
def rating_group(rating):
    if rating >= 4:
        return 'Top rated app'
    elif rating >=3:
        return 'Above average'
    elif rating >=2:
        return 'Average'
    else:
        return 'Below Average'
apps_df['Rating_Group']=apps_df['Rating'].apply(rating_group)

# %%
#Revenue column
apps_df['Revenue']=apps_df['Price']*apps_df['Installs']

# %%
sia = SentimentIntensityAnalyzer()

# %%
review = "This app is amazing! I love the new features."
sentiment_score= sia.polarity_scores(review)
print(sentiment_score)

# %%
review = "This app is very bad! I hate the new features."
sentiment_score= sia.polarity_scores(review)
print(sentiment_score)

# %%

review = "This app is okay."
sentiment_score= sia.polarity_scores(review)
print(sentiment_score)

# %%
reviews_df['Sentiment_Score']=reviews_df['Translated_Review'].apply(lambda x: sia.polarity_scores(str(x))['compound'])

# %%
reviews_df.head()

# %%
apps_df['Last Updated']=pd.to_datetime(apps_df['Last Updated'],errors='coerce')

# %%
apps_df['Year']=apps_df['Last Updated'].dt.year

# %%
import os
import plotly.io as pio

html_files_path="./"
if not os.path.exists(html_files_path):
    os.makedirs(html_files_path)
plot_containers = ""
plot_containers_1 = ""

# %%
plot_width=400
plot_height=300
plot_bg_color='black'
text_color='white'
title_font={'size':16}
axis_font={'size':12}

# %%
import os
import plotly.io as pio


# Function to save Plotly figure as HTML
def save_plot_as_html(fig, filename, insight):
    global plot_containers
    # global plot_containers_1
    filepath = os.path.join(html_files_path, filename)
    html_content = pio.to_html(fig, full_html=False, include_plotlyjs='inline')  # pio

    # Check if the figure is figurea or figureb and handle their time windows
    plot_containers += f"""
        <div class="plot-container" id="{filename}" onclick="openPlot('{filename}')">
            <div class="plot">{html_content}</div>
            <div class="insights">{insight}</div>
        </div>
        """
        
    
    fig.write_html(filepath, full_html=False, include_plotlyjs='inline')


# %%
#Figure 1
category_counts=apps_df['Category'].value_counts().nlargest(10)
fig1=px.bar(
    x=category_counts.index,
    y=category_counts.values,
    labels={'x':'Category','y':'Count'},
    title='Top Categories on Play Store',
    color=category_counts.index,
    color_discrete_sequence=px.colors.sequential.Plasma,
    width=400,
    height=300
)
fig1.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10,r=10,t=30,b=10)
)
#fig1.update_traces(marker=dict(pattern=dict(line=dict(color='white',width=1))))
save_plot_as_html(fig1,"Category Graph 1.html","The top categories on the Play Store are dominated by tools, entertainment, and productivity apps")
            

# %%
#Figure 2
type_counts=apps_df['Type'].value_counts()
fig2=px.pie(
    values=type_counts.values,
    names=type_counts.index,
    title='App Type Distribution',
    color_discrete_sequence=px.colors.sequential.RdBu,
    width=400,
    height=300
)
fig2.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    margin=dict(l=10,r=10,t=30,b=10)
)
#fig1.update_traces(marker=dict(pattern=dict(line=dict(color='white',width=1))))
save_plot_as_html(fig2,"Type Graph 2.html","Most apps on the Playstore are free, indicating a strategy to attract users first and monetize through ads or in app purchases")

# %%
#Figure 3
fig3=px.histogram(
    apps_df,
    x='Rating',
    nbins=20,
    title='Rating Distribution',
    color_discrete_sequence=['#636EFA'],
    width=400,
    height=300
)
fig3.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10,r=10,t=30,b=10)
)
#fig1.update_traces(marker=dict(pattern=dict(line=dict(color='white',width=1))))
save_plot_as_html(fig3,"Rating Graph 3.html","Ratings are skewed towards higher values, suggesting that most apps are rated favorably by users")

# %%
#Figure 4
sentiment_counts=reviews_df['Sentiment_Score'].value_counts()
fig4=px.bar(
    x=sentiment_counts.index,
    y=sentiment_counts.values,
    labels={'x':'Sentiment Score','y':'Count'},
    title='Sentiment Distribution',
    color=sentiment_counts.index,
    color_discrete_sequence=px.colors.sequential.RdPu,
    width=400,
    height=300
)
fig4.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10,r=10,t=30,b=10)
)
#fig1.update_traces(marker=dict(pattern=dict(line=dict(color='white',width=1))))
save_plot_as_html(fig4,"Sentiment Graph 4.html","Sentiments in reviews show a mix of positive and negative feedback, with a slight lean towards positive sentiments")

# %%
#Figure 5
installs_by_category=apps_df.groupby('Category')['Installs'].sum().nlargest(10)
fig5=px.bar(
    x=installs_by_category.index,
    y=installs_by_category.values,
    orientation='h',
    labels={'x':'Installs','y':'Category'},
    title='Installs by Category',
    color=installs_by_category.index,
    color_discrete_sequence=px.colors.sequential.Blues,
    width=400,
    height=300
)
fig5.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10,r=10,t=30,b=10)
)
#fig1.update_traces(marker=dict(pattern=dict(line=dict(color='white',width=1))))
save_plot_as_html(fig5,"Installs Graph 5.html","The categories with the most installs are social and communication apps, reflecting their broad appeal and daily usage")

# %%
# Updates Per Year Plot
updates_per_year = apps_df['Last Updated'].dt.year.value_counts().sort_index()
fig6 = px.line(
    x=updates_per_year.index,
    y=updates_per_year.values,
    labels={'x': 'Year', 'y': 'Number of Updates'},
    title='Number of Updates Over the Years',
    color_discrete_sequence=['#AB63FA'],
    width=plot_width,
    height=plot_height
)
fig6.update_layout(
    plot_bgcolor=plot_bg_color,
    paper_bgcolor=plot_bg_color,
    font_color=text_color,
    title_font=title_font,
    xaxis=dict(title_font=axis_font),
    yaxis=dict(title_font=axis_font),
    margin=dict(l=10, r=10, t=30, b=10)
)
save_plot_as_html(fig6, "Updates Graph 6.html", "Updates have been increasing over the years, showing that developers are actively maintaining and improving their apps.")

# %%
#Figure 7
revenue_by_category=apps_df.groupby('Category')['Revenue'].sum().nlargest(10)
fig7=px.bar(
    x=installs_by_category.index,
    y=installs_by_category.values,
    labels={'x':'Category','y':'Revenue'},
    title='Revenue by Category',
    color=installs_by_category.index,
    color_discrete_sequence=px.colors.sequential.Greens,
    width=400,
    height=300
)
fig7.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10,r=10,t=30,b=10)
)
#fig1.update_traces(marker=dict(pattern=dict(line=dict(color='white',width=1))))
save_plot_as_html(fig7,"Revenue Graph 7.html","Categories such as Business and Productivity lead in revenue generation, indicating their monetization potential")

# %%
#Figure 8
genre_counts=apps_df['Genres'].str.split(';',expand=True).stack().value_counts().nlargest(10)
fig8=px.bar(
    x=genre_counts.index,
    y=genre_counts.values,
    labels={'x':'Genre','y':'Count'},
    title='Top Genres',
    color=installs_by_category.index,
    color_discrete_sequence=px.colors.sequential.OrRd,
    width=400,
    height=300
)
fig8.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10,r=10,t=30,b=10)
)
#fig1.update_traces(marker=dict(pattern=dict(line=dict(color='white',width=1))))
save_plot_as_html(fig8,"Genre Graph 8.html","Action and Casual genres are the most common, reflecting users' preference for engaging and easy-to-play games")

# %%
#Figure 9
fig9=px.scatter(
    apps_df,
    x='Last Updated',
    y='Rating',
    color='Type',
    title='Impact of Last Update on Rating',
    color_discrete_sequence=px.colors.qualitative.Vivid,
    width=400,
    height=300
)
fig9.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10,r=10,t=30,b=10)
)
#fig1.update_traces(marker=dict(pattern=dict(line=dict(color='white',width=1))))
save_plot_as_html(fig9,"Update Graph 9.html","The Scatter Plot shows a weak correlation between the last update and ratings, suggesting that more frequent updates dont always result in better ratings.")

# %%
#Figure 10
fig10=px.box(
    apps_df,
    x='Type',
    y='Rating',
    color='Type',
    title='Rating for Paid vs Free Apps',
    color_discrete_sequence=px.colors.qualitative.Pastel,
    width=400,
    height=300
)
fig10.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10,r=10,t=30,b=10)
)
#fig1.update_traces(marker=dict(pattern=dict(line=dict(color='white',width=1))))
save_plot_as_html(fig10,"Paid Free Graph 10.html","Paid apps generally have higher ratings compared to free apps, suggesting that users expect higher quality from apps they pay for")

# %%

from wordcloud import WordCloud, STOPWORDS
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# %%
nltk.download('stopwords')
nltk.download('punkt')

# %%
print("Initial Shape:", reviews_df.shape)
print("Unique Apps in Data:", reviews_df["App"].nunique())
#print the categories of the apps
print("Unique Categories in Data:", apps_df["Category"].unique())

# %%
health_fitness_apps = apps_df[apps_df['Category'] == 'HEALTH_AND_FITNESS']
print("Health & Fitness Apps Shape:", health_fitness_apps.shape)
print("Unique Apps in Health & Fitness Data:", health_fitness_apps["App"].nunique())

# %%
merged_df = pd.merge(apps_df, reviews_df, on='App')
print(merged_df.head())




# %%
five_star_reviews = merged_df[merged_df['Rating'] == 5]
print("Five Star Reviews Shape:", five_star_reviews.shape)

# %% [markdown]
# So there are no 5 Rated apps
# So taking  between 4.8 and 4.9 Rated apps

# %%
top_rated_apps = merged_df[merged_df['Rating'] >= 4.8]
top_rated_apps.shape
#unique apps in the top rated apps
print("Unique Apps in Top Rated Apps Data:", top_rated_apps["App"].nunique())

# %%
#generate word cloud
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Assuming top_rated_apps is your merged DataFrame
# Combine all reviews into a single string
text = ' '.join(top_rated_apps['Translated_Review'].dropna())

# Create a set of stopwords
stop_words = set(stopwords.words('english'))
stop_words.update(STOPWORDS)
stop_words.update(top_rated_apps['App'].tolist())  # Add app names to stopwords

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, 
                      background_color='white', 
                      stopwords=stop_words, 
                      min_font_size=10).generate(text)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Word Cloud: Most Frequent Keywords in Top-Rated App Reviews')
plt.axis('off')
plt.show()


# %% [markdown]
# Now word cloud for health and fitness category

# %%
health_top_rated = top_rated_apps[top_rated_apps['Category'] == 'HEALTH_AND_FITNESS']
print(health_top_rated['App'].nunique())


# %%
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from io import BytesIO
import base64

# Combine all reviews into a single string
text = ' '.join(health_top_rated['Translated_Review'].dropna())

# Create a set of stopwords
stop_words = set(stopwords.words('english'))
stop_words.update(STOPWORDS)
stop_words.update(health_top_rated['App'].tolist())  # Add app names to stopwords

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, 
                      background_color='white', 
                      stopwords=stop_words, 
                      min_font_size=10).generate(text)

# Save the word cloud to a BytesIO object (in memory)
img_buffer = BytesIO()
wordcloud.to_image().save(img_buffer, format='PNG')
img_buffer.seek(0)

# Convert the image to a base64 string
img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

# HTML img tag with base64-encoded image
wordcloud_img_tag = f'<img src="data:image/png;base64,{img_base64}" alt="Word Cloud Image" style="width: 100%; max-width: 600px; margin: 0 auto; display: block;">'

# Optionally, print it out to verify
wordcloud_img_tag


# %% [markdown]
# Task 1 is completed

# %%



# Clean and preprocess the data (as you have it in your notebook)
apps_df = apps_df.dropna(subset=['Rating', 'Type', 'Content Rating', 'Current Ver', 'Android Ver','Installs'])
reviews_df = reviews_df.dropna(subset=['Translated_Review', 'Sentiment', 'Sentiment_Polarity', 'Sentiment_Subjectivity'])

apps_df = apps_df[apps_df['Android Ver'].str.contains('Varies with device') == False]
apps_df = apps_df[apps_df['Current Ver'].str.contains('Varies with device') == False]

# Convert 'Installs' to string type
apps_df['Installs'] = apps_df['Installs'].astype(str)

# Convert Installs to numeric, removing '+' and ',' characters
apps_df['Installs_Numeric'] = apps_df['Installs'].str.replace('+', '').str.replace(',', '').astype(float)

#Group by Category and sum the numeric installs, then get the top 10
top_10_categories = apps_df.groupby('Category')['Installs_Numeric'].sum().nlargest(10)

print(top_10_categories)




# %%
top_categories_df = apps_df[apps_df['Category'].isin(top_10_categories.index)]
top_categories_df.shape

# %%
category_stats = top_categories_df.groupby('Category').agg({
    'Rating': 'mean',
    'Reviews': 'sum'
}).reset_index()

# %%
category_stats = category_stats.sort_values('Reviews', ascending=False)


# %%
fig, ax1 = plt.subplots(figsize=(12, 6))

x = np.arange(len(category_stats))
width = 0.35

# Plot average rating bars
rects1 = ax1.bar(x - width/2, category_stats['Rating'], width, label='Avg Rating', color='skyblue')
ax1.set_ylabel('Average Rating')
ax1.set_ylim(0, 5)


ax2 = ax1.twinx()
rects2 = ax2.bar(x + width/2, category_stats['Reviews'], width, label='Total Reviews', color='lightgreen')
ax2.set_ylabel('Total Review Count')
ax1.set_xticks(x)
ax1.set_xticklabels(category_stats['Category'], rotation=45, ha='right')


lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.title('Average Rating and Total Reviews for Top 10 App Categories by Installs')
plt.tight_layout()
plt.show()

# %% [markdown]
# Grouped bar chart to compare the average rating and total review count for the top 10 app categories by number of installs

# %%
print(apps_df['Size'].unique())

# %%
def convert_size(size):
    if pd.isna(size) or size == 'Varies with device':
        return np.nan
    if isinstance(size, str):
        if 'M' in size:
            return float(size.replace('M', ''))
        elif 'k' in size:
            return float(size.replace('k', '')) / 1024
        elif 'G' in size:
            return float(size.replace('G', '')) * 1024
    elif isinstance(size, (int, float)):
        return float(size)
    return np.nan

# Apply the conversion
apps_df['Size_MB'] = apps_df['Size'].apply(convert_size)


# %%
apps_df['Last Updated'] = pd.to_datetime(apps_df['Last Updated'])


# %%
print(apps_df.columns)

# %%
filtered_df = apps_df[
    (apps_df['Rating'] >= 4.0) &
    (apps_df['Size_MB'] >= 10) &
    (apps_df['Last Updated'].dt.month == 1) &
    (~apps_df['Rating'].isna()) &
    (~apps_df['Size_MB'].isna())
]
filtered_df.shape

# %%
# Convert 'Installs' to numeric
filtered_df['Installs_Numeric'] = filtered_df['Installs'].str.replace('+', '').str.replace(',', '').astype(float)

# Get top 10 categories
top_10_categories = filtered_df.groupby('Category')['Installs_Numeric'].sum().nlargest(10).index

top_categories_df = filtered_df[filtered_df['Category'].isin(top_10_categories)]

category_stats = top_categories_df.groupby('Category').agg({
    'Rating': 'mean',
    'Reviews': 'sum',
    'Installs_Numeric': 'sum'  # Use Installs_Numeric for consistency
}).reset_index()

category_stats = category_stats.sort_values('Reviews', ascending=False)

fig, ax1 = plt.subplots(figsize=(12, 6))

x = np.arange(len(category_stats))
width = 0.35

# Plot average rating bars
rects1 = ax1.bar(x - width/2, category_stats['Rating'], width, label='Avg Rating', color='skyblue')
ax1.set_ylabel('Average Rating')
ax1.set_ylim(4, 5)  # Adjusted to start from 4.0

# Create a second y-axis for review count
ax2 = ax1.twinx()
rects2 = ax2.bar(x + width/2, category_stats['Reviews'], width, label='Total Reviews', color='lightgreen')
ax2.set_ylabel('Total Review Count')

# Set x-axis ticks and labels
ax1.set_xticks(x)
ax1.set_xticklabels(category_stats['Category'], rotation=45, ha='right')

# Add legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.title('Average Rating and Total Reviews for Top 10 Categories')
plt.tight_layout()
plt.show()


# %%
import datetime
import os
import plotly.io as pio

# Get current time
current_time = datetime.datetime.now().time()


# Define the time window for each figure
start_time_figurea = datetime.time(15, 0)  # 3:00 PM for figurea
end_time_figurea = datetime.time(17, 0)    # 5:00 PM for figurea

start_time_figureb = datetime.time(16, 0)  # 4:00 PM for figureb
end_time_figureb = datetime.time(19, 00)    # 7:00 PM for figureb

# Function to save Plotly figure as HTML
def save_plot_ab_as_html(fig, filename, insight):
    global plot_containers
    # global plot_containers_1
    filepath = os.path.join(html_files_path, filename)
    html_content = pio.to_html(fig, full_html=False, include_plotlyjs='inline')  # pio

    # Check if the figure is figurea or figureb and handle their time windows
    if fig == figa:
        if start_time_figurea <= current_time <= end_time_figurea:
            plot_containers+= f"""
            <div class="plot-container" id="{filename}" onclick="openPlot('{filename}')">
                <div class="plot">{html_content}</div>
                <div class="insights">{insight}</div>
            </div>
            """
    elif filename == "bubble.html":
        if start_time_figureb <= current_time <= end_time_figureb:
            plot_containers += f"""
            <div class="plot-container" id="{filename}" onclick="openPlot('{filename}')">
                <div class="plot">{html_content}</div>
                <div class="insights">{insight}</div>
            </div>
            """
    
    fig.write_html(filepath, full_html=False, include_plotlyjs='inline')


# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create subplot with secondary y-axis
figa = make_subplots(specs=[[{"secondary_y": True}]])

# Add bars for average rating
figa.add_trace(
    go.Bar(
        x=category_stats['Category'],
        y=category_stats['Rating'],
        name="Avg Rating",
        marker_color='skyblue',
        offsetgroup=0
    ),
    secondary_y=False,
)

# Add bars for total reviews
figa.add_trace(
    go.Bar(
        x=category_stats['Category'],
        y=category_stats['Reviews'],
        name="Total Reviews",
        marker_color='lightgreen',
        offsetgroup=1
    ),
    secondary_y=True,
)

# Update layout
figa.update_layout(
    title_text='Average Rating and Total Reviews for average rating is below 4.0 and size below 10 M and last update should be Jan month',
    barmode='group',
    xaxis_tickangle=-45,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=300,
    width=400
)
#reduce the font size of all labels
figa.update_layout(font=dict(size=10))
#increase the area of the plot
figa.update_layout(margin=dict(l=10, r=10, t=30, b=10))



# Update axes
figa.update_xaxes(title_text="Category")
figa.update_yaxes(title_text="Average Rating", secondary_y=False, range=[4, 5])
figa.update_yaxes(title_text="Total Review Count", secondary_y=True)

# Show the plot
figa.show()

save_plot_ab_as_html(figa, "Bar.html", "Updates have been increasing over the years, Family is the best category")


# %% [markdown]
# Filter applied Filter out any categories where the average rating is below 4.0 and size below 10 M and last update should be Jan month

# %% [markdown]
# Third question
# 3. Plot a bubble chart to analyze the relationship between app size (in MB) and average rating, with the bubble size representing the number of installs. Include a filter to show only apps with a rating higher than 3.5 and that belong to the "Games" category and installs should be more than 50k as well as this graph should work only between 5 PM IST to 7 PM IST apart from that time we should not show this graph in dashboard itself.

# %%
# Convert 'Installs' to string type first
apps_df['Installs'] = apps_df['Installs'].astype(str)

# Now apply string operations and convert to float
apps_df['Installs_Num'] = apps_df['Installs'].str.replace('+', '').str.replace(',', '').astype(float)

# Create the bubble chart
fig = px.scatter(apps_df, 
                 x='Size_MB', 
                 y='Rating', 
                 size='Installs_Num', 
                 color='Category',
                 hover_name='App',
                 log_x=True,
                 size_max=60,
                 title='App Size vs Rating (Bubble size represents number of installs)')

# Update layout
fig.update_layout(
    xaxis_title='App Size (MB)',
    yaxis_title='Average Rating',
    legend_title='Category',
    height=600,
    width=1000
)

# Show the plot
fig.show()


# %%
# Convert 'Installs' to numeric, removing '+' and ',' characters
apps_df['Installs_Num'] = apps_df['Installs'].str.replace('+', '').str.replace(',', '').astype(float)

# Now filter using the numeric 'Installs_Num' column
filtered_df2 = apps_df[(apps_df['Rating'] > 3.5) & 
                       (apps_df['Category'] == 'GAME') & 
                       (apps_df['Installs_Num'] > 50000)]

figb = px.scatter(filtered_df2, 
                 x='Size_MB', 
                 y='Rating', 
                 size='Installs_Num',  # Use 'Installs_Num' here as well
                 color='Category',
                 hover_name='App',
                 log_x=True,
                 size_max=60,
                 title='App Size vs Rating for Games')

figb.update_layout(
    xaxis_title='App Size (MB)',
    yaxis_title='Rating',
    legend_title='Category',
    height=300,
    width=400
)

figb.show()

save_plot_ab_as_html(figb, "bubble.html", "Updates have been increasing over the years, Family is the best category")


# %%
plot_containers_split=plot_containers.split('</div>')
if len(plot_containers_split) > 1:
    final_plot=plot_containers_split[-2]+'</div>'
else:
    final_plot=plot_containers


# %%
dashboard_html= """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name=viewport" content="width=device-width,initial-scale-1.0">
    <title> Google Play Store Review Analytics</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            background-color: #333;
            color: #fff;
            margin: 0;
            padding: 0;
        }}
        .header {{
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
            background-color: #444
        }}
        .header img {{
            margin: 0 10px;
            height: 50px;
        }}
        .container {{
            display: flex;
            flex-wrap: wrap;
            justify_content: center;
            padding: 20px;
        }}
        .plot-container {{
            border: 2px solid #555
            margin: 10px;
            padding: 10px;
            width: {plot_width}px;
            height: {plot_height}px;
            overflow: hidden;
            position: relative;
            cursor: pointer;
        }}
        .insights {{
            display: none;
            position: absolute;
            right: 10px;
            top: 10px;
            background-color: rgba(0,0,0,0.7);
            padding: 5px;
            border-radius: 5px;
            color: #fff;
        }}
        .plot-container: hover .insights {{
            display: block;
        }}
        </style>
        <script>
            function openPlot(filename) {{
                window.open(filename, '_blank');
                }}
        </script>
    </head>
    <body>
        <div class= "header">
            <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/4a/Logo_2013_Google.png/800px-Logo_2013_Google.png" alt="Google Logo">
            <h1>Google Play Store Reviews Analytics</h1>
            <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/78/Google_Play_Store_badge_EN.svg/1024px-Google_Play_Store_badge_EN.svg.png" alt="Google Play Store Logo">
        </div>
        <div>
            {wordcloud_img_tag}
        </div>
        <div class="container">
            {plots}
        </div>
    </body>
    </html>
    """




# %%

# Now, update the plot containers based on the current time
# if start_time_figurea <= current_time <= end_time_figurea:
#     plot_containers += plot_containers_1  # Include plot_containers_1 if in time window

final_html = dashboard_html.format(plots=plot_containers, plot_width=plot_width, plot_height=plot_height,wordcloud_img_tag=wordcloud_img_tag)

# Save and display the dashboard
dashboard_path = os.path.join(html_files_path, "web_page.html")
with open(dashboard_path, "w", encoding="utf-8") as f:
    f.write(final_html)

import webbrowser
webbrowser.open('file://' + os.path.realpath(dashboard_path))



