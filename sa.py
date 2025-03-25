from textblob import TextBlob
import pandas as pd
import sys
import io
import matplotlib.pyplot as plt

# Set the encoding to UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')



# Check if 'FBPost' column exists
if 'self_text' not in df_quotes.columns:
    raise ValueError("The column 'self_text' does not exist in the CSV file.")

# Initialize a list to hold sentiment analysis results
reddit = []

# Analyze quotes
for self_text in df_quotes['self_text']:
    blob = TextBlob(str(self_text))
    score = {
        'FBPost': str(self_text),
        'polarity': blob.sentiment.polarity,
        'subjectivity': blob.sentiment.subjectivity
    }
    reddit.append(score)

# Create DataFrame from analyzed quotes
df_analysis = pd.DataFrame(reddit)

# Classify polarity
df_analysis['Sentiment'] = df_analysis['polarity'].apply(
    lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral')
)

# Filter out posts with zero polarity
df_analysis = df_analysis[df_analysis.polarity != 0.0]

# Set pandas option to display all rows
pd.set_option('display.max_rows', None)

# Print the analyzed DataFrame
print(df_analysis)

# Save the results to a new CSV file
df_analysis.to_csv('analyzed_fb_posts.csv', index=False)

# Visualization - Pie Chart of Sentiment Distribution
sentiment_counts = df_analysis['Sentiment'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=['green', 'red', 'gray'])
plt.title('Sentiment Distribution')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()
