import pandas as pd
import matplotlib.colors as mcolors
import random
from datetime import datetime, time
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from collections import Counter
import string
import html

nltk.download('stopwords')
nltk.download('omw-1.4')

df = pd.read_csv("reports.csv")

df["country"] = df["country"].fillna("unknown")
df["state"] = df["state"].fillna("unknown")
df["shape"] = df["shape"].fillna("unknown")
df = df.drop(["city","duration (hours/min)", "date posted"], axis=1)
df = df.rename(columns={"duration (seconds)": "seconds"})
df = df.iloc[:, :-1]
df = df.dropna(how = "any")
df["latitude"] = pd.to_numeric(df["latitude"], errors='coerce')
df = df.dropna(subset=['latitude'])
df['seconds'] = pd.to_numeric(df['seconds'], errors='coerce')
df.dropna(inplace=True)
df = df.query('latitude != 0 or longitude != 0')

countries = df['country'].unique()
country_mapping = {
    "us" : "USA",
    "ca" : "Canada",
    "gb" : "Great Britain",
    "de" : "Germany",
    "au" : "Australia",
    "unknown"   : "unknown"
}
df["country"] = df["country"].map(country_mapping)



state_mapping = {
    "unknown" : "unknown",
    'al': 'Alabama',
    'ak': 'Alaska',
    'az': 'Arizona',
    'ar': 'Arkansas',
    'ca': 'California',
    'co': 'Colorado',
    'ct': 'Connecticut',
    'de': 'Delaware',
    'fl': 'Florida',
    'ga': 'Georgia',
    'hi': 'Hawaii',
    'id': 'Idaho',
    'il': 'Illinois',
    'in': 'Indiana',
    'ia': 'Iowa',
    'ks': 'Kansas',
    'ky': 'Kentucky',
    'la': 'Louisiana',
    'me': 'Maine',
    'md': 'Maryland',
    'ma': 'Massachusetts',
    'mi': 'Michigan',
    'mn': 'Minnesota',
    'ms': 'Mississippi',
    'mo': 'Missouri',
    'mt': 'Montana',
    'ne': 'Nebraska',
    'nv': 'Nevada',
    'nh': 'New Hampshire',
    'nj': 'New Jersey',
    'nm': 'New Mexico',
    'ny': 'New York',
    'nc': 'North Carolina',
    'nd': 'North Dakota',
    'oh': 'Ohio',
    'ok': 'Oklahoma',
    'or': 'Oregon',
    'pa': 'Pennsylvania',
    'ri': 'Rhode Island',
    'sc': 'South Carolina',
    'sd': 'South Dakota',
    'tn': 'Tennessee',
    'tx': 'Texas',
    'ut': 'Utah',
    'vt': 'Vermont',
    'va': 'Virginia',
    'wa': 'Washington',
    'wv': 'West Virginia',
    'wi': 'Wisconsin',
    'wy': 'Wyoming',
}
df["state"] = df["state"].map(state_mapping).fillna("other")

df['datetime'] = pd.to_datetime(df['datetime'], format='%m/%d/%Y %H:%M', errors='coerce')
df = df.dropna(subset=['datetime'])
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.strftime('%b')
df['weekday'] = df['datetime'].dt.day_name()
df['hour'] = df['datetime'].dt.hour



years = list(set(df["year"])) ; years.sort()
first = years[0] ; last = years[-1]
year_colormap = mcolors.LinearSegmentedColormap.from_list('my_colormap', ['blue', 'red'])
year_norm = mcolors.Normalize(vmin=first, vmax=last)
df["year_color"] = [year_colormap(year_norm(value)) for value in (df["year"])]
df['year_color'] = df['year_color'].apply(lambda x: '#%02x%02x%02x%02x' % tuple(int(c * 255) for c in x) if isinstance(x, tuple) and len(x) == 4 else None)

months       = ['Jan',   'Feb',   'Mar',   'Apr',   'May',   'Jun',   'Jul',   'Aug',   'Sep',   'Oct',   'Nov',   'Dec']
month_colors = [(0,0,1), (0,0,1), (0,1,1), (0,1,1), (0,1,1), (1,0,0), (1,0,0), (1,0,0), (1,0,1), (1,0,1), (1,0,1), (0,0,1)]
month_colors = {month : color for month, color in zip(months, month_colors)}
df["month_color"] = [month_colors[month] for month in df["month"]]
df['month_color'] = df['month_color'].apply(lambda x: '#%02x%02x%02x' % tuple(int(c * 255) for c in x))

weekdays       = ["Monday", "Tuesday",   "Wednesday", "Thursday",   "Friday", "Saturday", "Sunday"]
weekday_colors = [(1,0,0),  (.75,0,.25), (.5,0,.5),   (.25,0,.75),  (0,0,1),  (0,.5,.5),  (0,1,0)]
weekday_colors = {weekday : color for weekday, color in zip(weekdays, weekday_colors)}
df["weekday_color"] = [weekday_colors[weekday] for weekday in df["weekday"]]
df['weekday_color'] = df['weekday_color'].apply(lambda x: '#%02x%02x%02x' % tuple(int(c * 255) for c in x))

hours       = [i for i in range(24)]
hour_colors = [
    (0, 0, 0),         # 0 (midnight)
    (0, 0.1, 0.1),
    (0, 0.2, 0.2),
    (0, 0.3, 0.3),
    (0, 0.4, 0.4),
    (0, 0.5, 0.5),
    (0, 0.6, 0.6),
    (0, 0.7, 0.7),
    (0, 0.8, 0.8),
    (0, 0.9, 0.9),
    (0, 1, 1),         # 10
    (0, 1, 1),         # 11
    (0, 1, 1),         # 12 (noon)
    (0, 1, 1),         # 13
    (0, 0.9, 0.9),
    (0, 0.8, 0.8),
    (0, 0.7, 0.7),
    (0, 0.6, 0.6),
    (0, 0.5, 0.5),
    (0, 0.4, 0.4),
    (0, 0.3, 0.3),     # 20
    (0, 0.2, 0.2),
    (0, 0.1, 0.1),
    (0, 0, 0)          # 23
]
hour_colors = {hour : color for hour, color in zip(hours, hour_colors)}
df["hour_color"] = [hour_colors[hour] for hour in df["hour"]]
df['hour_color'] = df['hour_color'].apply(lambda x: '#%02x%02x%02x' % tuple(int(c * 255) for c in x))

seconds = list(set(df["seconds"])) ; seconds.sort()
first = seconds[0] ; last = seconds[-1]
seconds_colormap = mcolors.LinearSegmentedColormap.from_list('my_colormap', ['blue', 'red'])
seconds_norm = mcolors.Normalize(vmin=first, vmax=last)
df["seconds_color"] = [seconds_colormap(seconds_norm(value)) for value in (df["seconds"])]
df['seconds_color'] = df['seconds_color'].apply(lambda x: '#%02x%02x%02x%02x' % tuple(int(c * 255) for c in x) if isinstance(x, tuple) and len(x) == 4 else None)

shapes = ['circle', 'disk', 'oval', 'round', 'teardrop', 'crescent', 'cigar', 'egg', 'sphere', 'fireball', 'cylinder', 'cone', 'pyramid', 'diamond', 'triangle', 'hexagon', 'delta', 'chevron', 'cross', 'rectangle', 'formation', 'changed', 'changing', 'light', 'flash', 'flare', 'unknown', 'other']
shape_colors = [
    (1, 0, 0),         # circle
    (1, 0.1, 0),       # disk
    (0.9, 0.2, 0),     # oval
    (1, 0.3, 0),       # round
    (1, 0, 0.1),       # teardrop
    (0.9, 0.4, 0),     # crescent
    (0.8, 0.5, 0),     # cigar
    (0.7, 0.6, 0),     # egg
    (1, 0.7, 0),       # sphere
    (0.9, 0.8, 0),     # fireball
    (0, 1, 0),         # cylinder
    (0, 1, 0.1),       # cone
    (0, 0.9, 0.2),     # pyramid
    (0, 1, 0.3),       # diamond
    (0, 0.9, 0.4),     # triangle
    (0, 1, 0.5),       # hexagon
    (0, 0.9, 0.6),     # delta
    (0, 1, 0.7),       # chevron
    (0, 0.9, 0.8),     # cross
    (0, 0, 1),         # rectangle
    (0.1, 0.2, 1),     # formation
    (0.2, 0.3, 0.9),   # changed
    (0.3, 0.4, 1),     # changing
    (0.4, 0.5, 0.9),   # light
    (0.5, 0.6, 1),     # flash
    (0.6, 0.7, 0.9),   # flare
    (0.7, 0.8, 1),     # unknown
    (0.8, 0.9, 0.9)    # other
]
shape_colors = {shape : color for shape, color in zip(shapes, shape_colors)}
df["shape_color"] = [shape_colors[shape] for shape in df["shape"]]
df['shape_color'] = df['shape_color'].apply(lambda x: '#%02x%02x%02x' % tuple(int(c * 255) for c in x))
df["shape"] = df["shape"].fillna("unknown")

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

df['comments'] = df['comments'].astype(str).apply(html.unescape)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
df['processed_comments'] = df['comments'].str.lower().apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
df['tokenized_comments'] = df['processed_comments'].apply(lambda x: nltk.word_tokenize(x) if isinstance(x, str) else [])
df['tokenized_comments'] = df['tokenized_comments'].apply(lambda x: [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in x])
df['tokenized_comments'] = df['tokenized_comments'].apply(lambda x: ' '.join(word for word in x if word not in stop_words))

df['word_frequency'] = df['tokenized_comments'].apply(lambda x: dict(Counter(x)))
vectorizer = CountVectorizer(min_df=100)
X = vectorizer.fit_transform(df['tokenized_comments'])
features = vectorizer.get_feature_names_out()
existing_cols = set(df.columns)
features = [f'"{feature}"' if feature in existing_cols else feature for feature in features]
df_wordcounts = pd.DataFrame(X.toarray(), columns=features)

df.reset_index(drop=True, inplace=True)
df_wordcounts.reset_index(drop=True, inplace=True)

df = df.reindex(columns=["datetime", "year", "month", "weekday", "hour", "seconds", "shape", "latitude", "longitude", "country", "state", "comments", "year_color", "month_color", "weekday_color", "hour_color", "seconds_color", "shape_color"])
df = pd.concat([df, df_wordcounts], axis=1)
df.to_csv(path_or_buf = "processed_reports.csv", sep=',')

print("{} Columns, {} Rows.\n".format(df.shape[1], df.shape[0]))
print(df.head(), "\n")



import pickle 

with open("assets.pickle", "wb") as file:
    pickle.dump([year_colormap, year_norm, 
                 month_colors, months,
                 weekday_colors, weekdays, hours,
                 shape_colors,
                 seconds_colormap, seconds_norm], file = file)