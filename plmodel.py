#import kagglehub
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv('premier-league-matches.csv')

# Get to Know the Data
# print(df.shape)
# print(df.head())
# print(df.dtypes)
# print(df.isnull().sum())

#Step 1: understand target variable
#decoding result
df['result'] = df['FTR'].map({'H':'Home Win', 'A':'Away Win', 'D': 'Draw'})
print(df['result'].value_counts())
print(df['result'].value_counts(normalize=True).round(3))

# #Step 2: Features
# print(df.describe())

#Step 2.1: Encode
le_temp = LabelEncoder()
df['result_encoded'] = le_temp.fit_transform(df['result'])
df.corr(numeric_only=True)['result_encoded'].sort_values()

#Step 3: Visualize
df['HomeGoals'].hist(bins=10)
# plt.title('Distribution of HomeGoals')
# plt.show()

#Step 4: Encode Features to pass numbers to models
df['result'] = df['FTR'].map({'H':'Home Win', 'A':'Away Win', 'D': 'Draw'})
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

df['home_goals_avg'] = df.groupby('Home')['HomeGoals'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean()
)

df['away_goals_avg'] = df.groupby('Away')['AwayGoals'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean()
)

df['home_conceded_avg'] = df.groupby('Home')['AwayGoals'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean()
)

df['away_conceded_avg'] = df.groupby('Away')['HomeGoals'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean()
)
df= df.dropna()

le_home = LabelEncoder()
le_away = LabelEncoder()
le_result = LabelEncoder()

df['home_encoded'] = le_home.fit_transform(df['Home'])
df['away_encoded'] = le_away.fit_transform(df['Away'])
df['result_encoded'] = le_result.fit_transform(df['result'])


#Step 5:Split data into train and test sets
features = ['home_encoded', 'away_encoded', 'home_goals_avg', 'away_goals_avg', 'home_conceded_avg', 'away_conceded_avg']
X = df[features]
y = df['result_encoded']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

#Step 6: Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

#Step 7: Predict
predictions = model.predict(X_test)

#Step 8: Decode predictions into labels
pred_lables = le_result.inverse_transform(predictions[:10])
print(pred_lables)

#Step 9: Evaluate the model
print(accuracy_score(y_test, predictions))

#Step 10: Debug low accuracy
print(classification_report(y_test, predictions, target_names= le_result.classes_))

cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=['Away Win', 'Draw', 'Home Win'],
            yticklabels=['Away Win', 'Draw', 'Home Win'])