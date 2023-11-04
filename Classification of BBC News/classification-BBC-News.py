'''*****************************************************************************
Purpose: To analyze the 
Implementation:

Limitations:

Author: github:https://github.com/DivyaBiradar54
-------------------------------------------------------------------
****************************************************************************'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer, roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
dataset = pd.read_csv("BBC News Train.csv")
dataset.head()
dataset.shape
dataset.info()
dataset['Category'].value_counts()
target_category = dataset['Category'].unique()
print(target_category)
dataset['CategoryId'] = dataset['Category'].factorize()[0]
dataset.head()
category = dataset[['Category', 'CategoryId']].drop_duplicates().sort_values('CategoryId')
category
dataset.groupby('Category').CategoryId.value_counts().plot(kind = "bar", color = ["pink", 
"orange", "red", "yellow", "blue"])
plt.xlabel("Category of data")
plt.title("Visulaize numbers of Category of data")
fig = plt.figure(figsize = (5,5))
colors = ["skyblue"]
business = dataset[dataset['CategoryId'] == 0 ]
tech = dataset[dataset['CategoryId'] == 1 ]
politics = dataset[dataset['CategoryId'] == 2]
sport = dataset[dataset['CategoryId'] == 3]
entertainment = dataset[dataset['CategoryId'] == 4]
count = [business['CategoryId'].count(), tech['CategoryId'].count(), 
politics['CategoryId'].count(), sport['CategoryId'].count(), entertainment['CategoryId'].count()]
pie = plt.pie(count, labels = ['business', 'tech', 'politics', 'sport', 'entertainment'],
autopct = "%1.1f%%",
shadow = True,
colors = colors,
startangle = 45,
explode = (0.05, 0.05, 0.05, 0.05,0.05))
from wordcloud import WordCloud
stop = set(stopwords.words('english'))
business = dataset[dataset['CategoryId'] == 0]
business = business['Text']
tech = dataset[dataset['CategoryId'] == 1]
tech = tech['Text']
politics = dataset[dataset['CategoryId'] == 2]
politics = politics['Text']
sport = dataset[dataset['CategoryId'] == 3]
sport = sport['Text']
entertainment = dataset[dataset['CategoryId'] == 4]
entertainment = entertainment['Text']
def wordcloud_draw(dataset, color = 'white'):
words = ' '.join(dataset)
cleaned_word = ' '.join([word for word in words.split()
if (word != 'news' and word != 'text')])
wordcloud = WordCloud(stopwords = stop,background_color = color,width = 2500, height = 
2500).generate(cleaned_word)
plt.figure(1, figsize = (10,7))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
print("business related words:")
wordcloud_draw(business, 'white')
print("tech related words:")
wordcloud_draw(tech, 'white')
print("politics related words:")
wordcloud_draw(politics, 'white')
print("sport related words:")
wordcloud_draw(sport, 'white')
print("entertainment related words:")
wordcloud_draw(entertainment, 'white')
text = dataset["Text"]
text.head(10)
category = dataset['Category']
category.head(10)
def remove_tags(text):
remove = re.compile(r'')
return re.sub(remove, '', text)
dataset['Text'] = dataset['Text'].apply(remove_tags)
def special_char(text):
reviews = ''
for x in text:
if x.isalnum():
reviews = reviews + x
else:
reviews = reviews + ' '
return reviews
dataset['Text'] = dataset['Text'].apply(special_char)
def convert_lower(text):
return text.lower()
dataset['Text'] = dataset['Text'].apply(convert_lower)
dataset['Text'][1]
def remove_stopwords(text):
stop_words = set(stopwords.words('english'))
words = word_tokenize(text)
return [x for x in words if x not in stop_words]
dataset['Text'] = dataset['Text'].apply(remove_stopwords)
dataset['Text'][1]
def lemmatize_word(text):
wordnet = WordNetLemmatizer()
return " ".join([wordnet.lemmatize(word) for word in text])
dataset['Text'] = dataset['Text'].apply(lemmatize_word)
dataset['Text'][1]
dataset
x = dataset['Text']
y = dataset['CategoryId']
from sklearn.feature_extraction.text import CountVectorizer
x = np.array(dataset.iloc[:,0].values)
y = np.array(dataset.CategoryId.values)
cv = CountVectorizer(max_features = 5000)
x = cv.fit_transform(dataset.Text).toarray()
print("X.shape = ",x.shape)
print("y.shape = ",y.shape)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0, shuffle 
= True)
print(len(x_train))
print(len(x_test))
perform_list = [ ]
def run_model(model_name, est_c, est_pnlty):
mdl=’’
if model_name == 'Logistic Regression':
mdl = LogisticRegression()
elif model_name == 'Random Forest':
mdl = RandomForestClassifier(n_estimators=100 ,criterion='entropy' , random_state=0)
elif model_name == 'Multinomial Naive Bayes':
mdl = MultinomialNB(alpha=1.0,fit_prior=True)
elif model_name == 'Support Vector Classifer':
mdl = SVC()
elif model_name == 'Decision Tree Classifier':
mdl = DecisionTreeClassifier()
elif model_name == 'K Nearest Neighbour':
mdl = KNeighborsClassifier(n_neighbors=10 , metric= 'minkowski' , p = 4)
elif model_name == 'Gaussian Naive Bayes':
mdl = GaussianNB()
oneVsRest = OneVsRestClassifier(mdl)
oneVsRest.fit(x_train, y_train)
y_pred = oneVsRest.predict(x_test)
accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
precision, recall, f1score, support = score(y_test, y_pred, average='micro')
print(f'Test Accuracy Score of Basic {model_name}: % {accuracy}')
print(f'Precision : {precision}')
print(f'Recall : {recall}')
print(f'F1-score : {f1score}')
perform_list.append(dict([('Model', model_name),('Test Accuracy', round(accuracy, 
2)),('Precision', round(precision, 2)),('Recall', round(recall, 2)),('F1', round(f1score, 2))]))
run_model('Logistic Regression', est_c=None, est_pnlty=None)
run_model('Random Forest', est_c=None, est_pnlty=None)
run_model('Multinomial Naive Bayes', est_c=None, est_pnlty=None)
run_model('Support Vector Classifer', est_c=None, est_pnlty=None)
run_model('Decision Tree Classifier', est_c=None, est_pnlty=None)
run_model('K Nearest Neighbour', est_c=None, est_pnlty=None)
run_model('Gaussian Naive Bayes', est_c=None, est_pnlty=None)
model_performance = pd.DataFrame(data=perform_list)
model_performance = model_performance[['Model', 'Test Accuracy', 'Precision', 'Recall', 
'F1']]
model_performance
model = model_performance["Model"]
max_value = model_performance["Test Accuracy"].max()
print("The best accuracy of model is", max_value,"from Random")
classifier = RandomForestClassifier(n_estimators=100 ,criterion='entropy' , 
random_state=0).fit(x_train, y_train)
classifier
y_pred = classifier.predict(x_test)
y_pred1 = cv.transform(['Hour ago, I contemplated retirement for a lot of reasons. I felt like 
people were not sensitive enough to my injuries. I felt like a lot of people were backed, why 
not me? I have done no less. I have won a lot of games for the team, and I am not feeling 
backed, said Ashwin'])
yy = classifier.predict(y_pred1)
result = ""
if yy == [0]:
result = "Business News"
elif yy == [1]:
result = "Tech News"
elif yy == [2]:
result = "Politics News"
elif yy == [3]:
result = "Sports News"
elif yy == [1]:
result = "Entertainment News"
print(result)
