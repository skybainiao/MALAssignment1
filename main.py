import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
import nltk
import numpy as np
import seaborn as sns




filename = 'C:/Users/45527/Downloads/listings.csv'
data = pd.read_csv('C:/Users/45527/Downloads/listings.csv')
print(data.head())


columns_to_keep = [
    'id', 'name', 'host_id', 'host_name', 'neighbourhood_cleansed',
    'latitude', 'longitude', 'room_type', 'price', 'minimum_nights',
    'number_of_reviews', 'last_review', 'review_scores_rating',
    'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin',
    'review_scores_communication', 'review_scores_location', 'review_scores_value',
    'reviews_per_month', 'calculated_host_listings_count', 'availability_365'
]
data = data[columns_to_keep]
print(data.head())


data = data[data['number_of_reviews'] != 0]
data = data.dropna()
print(data.isnull().sum())



text = ' '.join(data['name'].astype(str))
wordcloud = WordCloud(width=800, height=400, background_color ='white', stopwords={'Copenhagen'}).generate(text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
text = ' '.join(data['name'].astype(str).tolist())
wordcloud = WordCloud(stopwords=stop_words, background_color='white', width=800, height=400).generate(text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')



names_text = ' '.join(data['host_name'].astype(str).tolist())
name_wordcloud = WordCloud(background_color='white', width=800, height=400).generate(names_text)
plt.figure(figsize=(10, 5))
plt.imshow(name_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()



data['price'] = pd.to_numeric(data['price'], errors='coerce')
data = data.dropna(subset=['price'])
bins = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, np.inf]
labels = ['0-1000', '1000-2000', '2000-3000', '3000-4000', '4000-5000', '5000-6000', '6000-7000', '7000-8000', '8000-9000', '9000-10000', '10000+']
data['price_bin'] = pd.cut(data['price'], bins=bins, labels=labels)
print(data.head())





plt.figure(figsize=(10, 10))
sns.scatterplot(x='longitude', y='latitude', hue='price_bin', data=data, palette='viridis')
plt.title('Listings on Map colored by Price Bin')
plt.xlabel('Longitude')
plt.ylabel('Latitude')


neighbourhood_counts = data['neighbourhood_cleansed'].value_counts()
plt.figure(figsize=(12, 6))
sns.barplot(x=neighbourhood_counts.index, y=neighbourhood_counts.values, palette='viridis')
plt.title('Number of Listings in each Neighbourhood')
plt.xlabel('Neighbourhood')
plt.ylabel('Number of Listings')
plt.xticks(rotation=45)



plt.figure(figsize=(12, 8))
palette = sns.color_palette("husl", n_colors=data['neighbourhood_cleansed'].nunique())
sns.scatterplot(x='longitude', y='latitude', hue='neighbourhood_cleansed', data=data, palette=palette, s=50)
plt.title('Airbnb Listings Grouped by Neighbourhood')
plt.xlabel('Longitude')
plt.ylabel('Latitude')


plt.figure(figsize=(16, 8))
sns.boxplot(x='neighbourhood_cleansed', y='price', data=data)
plt.xticks(rotation=45)
plt.title('Boxplot of Price by Neighbourhood')


top_hosts = data.groupby('host_id').count().sort_values(by='id', ascending=False).head(10).reset_index()
plt.figure(figsize=(12, 6))
sns.barplot(x='host_id', y='id', data=top_hosts, palette='viridis')
plt.title('Top 10 Hosts with Most Listings')
plt.xlabel('Host ID')
plt.ylabel('Number of Listings')
plt.xticks(rotation=45)
plt.show()



neighbourhood_analysis = data.groupby(['neighbourhood_cleansed', 'room_type']).describe(include='all')
print(neighbourhood_analysis)
