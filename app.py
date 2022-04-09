from flask import Flask, render_template, url_for, request
from flask_bootstrap import Bootstrap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from wordcloud import WordCloud
import bnlp
from bnlp.corpus import stopwords, punctuations
from wordcloud import WordCloud, STOPWORDS , ImageColorGenerator
from bnlp import BasicTokenizer,NLTKTokenizer
from bangla_stemmer.stemmer.stemmer import BanglaStemmer
import warnings
warnings.filterwarnings("ignore")
from sklearn.multiclass import OneVsOneClassifier
from sklearn import preprocessing
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import cross_val_score,train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error,confusion_matrix,accuracy_score,f1_score,classification_report, precision_score, recall_score, auc,roc_curve

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib
from nltk import pos_tag
from nltk.corpus import wordnet
from sklearn.svm import LinearSVC,SVC
from sklearn.pipeline import Pipeline
from bnlp import POS
from collections import Counter
from functools import reduce

spec_chars = ["!",'"',"।","#","%","&","'","(",")",
              "*","+",",","-",".","/",":",";","<",
              "=",">","?","@","[","\\","]","^","_",
              "`","{","|","}","~","–"]


custom_stop_word_list=['আমার ','অথচ ','অথবা ','অনুযায়ী ','অনেক ','অনেকে ','অনেকেই ','অন্তত ','অন্য ','অবধি ','অবশ্য ','অর্থাত ','আই ','আগামী ','আগে ','আগেই ','আছে ','আজ ','আদ্যভাগে ',
                       'আপনার ','আপনারা ','আপনি ','আবার ','আসবে ','আমরা ',' আমাকে ','আমাদের ','আমার ','আমি ','আর ','আরও ','ইত্যাদি ','ইহা ','উচিত ','উত্তর ','উনি ','উপর ','উপরে ','এ ','এঁদের ','এঁরা ','এরা ',
                       'এই ','একই ','একটি ','একবার ','একে ','এক্ ','এখন ','এখনও ','এখানে ','এখানেই ','এটা ','এটাই ','এটি ','এত ','এতটাই ','এতে ','এদের ','এব ','এবং ','এবার ','এমন ','এমনকী ',
                       'এমনি ','এর ','এরা ','এল ','এস ','এসে ','ঐ ','ওঁদের ','ওঁর ','ওঁরা ','ওই ','ওকে ','ওখানে ','ওদের ','ওর ','ওরা ','কখনও ','কত ','কবে ','কমনে ','কয়েক ','কয়েকটি ','করছে ',
                       'করছেন ','করতে ',' করবে',' করবেন',' করলে ',' করলেন',' করা',' করাই',' করায়',' করার',' করি','করতে ','করিতে ','করিয়া ','করিয়ে ','করে ','করেই ','করেছিলেন ','করেছে ','করেছেন ','করেন ',
                       'কাউকে ','কাছ ','কাছে ','কাজ ','কাজে ','কারও ','কারণ ','কি ','কিংবা ','কিছু ','কিছুই ','হেতি ','কিন্তু ','ন্তু ','কী ','কে ','কেউ ','কেউই ','কেখা ','কেন ','কোটি ','কোন ','কোনও ',
                       'কোনো ','ক্ষেত্রে ','কয়েক ','খুব ','গিয়ে ','গিয়েছে ','গেছেন ','গিয়ে ','গুলি ','গেছে ','গেল ','গেলে ','গোটা ','চলে ','চান ','চায় ','চার ','চালু ','চেয়ে ','চেষ্টা ','ছাড়া ','ছাড়াও ','ছিল ','ছিলেন ','জন ',
                       'জনকে ','জনের ','জন্য ','জন্যওজে ','জানতে ','জানা ','জানানো ','জানায় ','জানিয়ে ','জানিয়েছে ','জ্নজন ','জন ','টা ','টি ','ঠিক ','তখন ','তত ','তথা ','তবু ','তবে ','তা ','তাঁকে ','তাঁদের ',
                       'তাঁর ','তোর ','তাঁরা ','তাঁহারা ','তাই ','যে ''তাও ','তাকে ','তাতে ','তাদের ','তার ','তারপর ','তারা ','তারৈ ','তাহলে ','তাহা ','তাহাতে ' ,'তাহার ','তিনঐ ','তিনি ','তিনিও ','তুমি ','তুলে ','তেমন ','তো ','তোমার ',
                       'থাকবে ','থাকবেন ','থাকা ','থাকায় ','থাকে ','থাকেন ','থেকে ','থেকেই ','থেকেও ','দিকে ','দিতে ','দিতাম','দিন ','দিয়ে ','দিয়েছে ','দিয়েছেন ','দিলেন ', 'দু ','দুই ','দুটি ','দুটো ','দেওয়া ','দেওয়ার ','দেওয়া ',
                       'দেখতে ','দেখা ','দেখে ','দেন ','দেয়া ','দেয় ','দ্বারা ','ধরা ','ধরে ','ধামার ','নতুন ','নাই ','নাকি ','নাগাদ ','নানা ','নিজে ','নিজেই ','নিজেদের ','নিজের ','নিতে ','নিয়ে ','নিয়ে ','নেই ','নেওয়া ','নেওয়ার ',
                       'নেওয়া ','নয় ','পক্ষে ','পর ','পরে ','পরেই ','পরেও ','পর্যন্ত ','পাওয়া ','পাচ ','পারি ','পারে ','পারেন ','পেয়ে ','পেয়্র্ ','প্রতি ','প্রথম ','প্রভৃতি ','প্রযন্ত ','প্রাথমিক ','প্রায় ','প্রায় ','ফলে ','ফিরে ','ফের ',
                       'বক্তব্য ','বদলে ','বন ','বরং ','বলতে ','বলছি ','বলল ','বললেন ','বলা ','বলে ','বলেছেন ','বলেন ','বসে ','বহু' ,'বাদে ','বার ','বিনা ','বিভিন্ন ','বিশেষ ','বিষয়টি ','বেশ ','বেশি ','ব্যবহার ','ব্যাপারে ','ভাবে ', 'ভাবেই ',
                       'মতো ','মতোই ','মধ্যভাগে ','মধ্যে ','মধ্যেই ','মধ্যেও ','মনে ','মাত্র ','মাধ্যমে ','মোট ','মোটেই ','যখন ','যত ','যতটা ','যথেষ্ট ','যদি ','যদিও ','যা ','যাঁর ','যাঁরা ','যাওয়া ','যাওয়ার ','যাওয়া ','যাকে ','যাচ্ছে ',
                       'যাতে ','যাদের ','যান ','যাবে ','যায় ','যার ','যারা ','যিনি ','অতএব ','যেখানে ','যেতে ','যেন ','যেমন ','রকম ','রয়েছে ','রাখা ','রেখে ','লক্ষ ','শুধু ','শুরু ','সঙ্গে ','সঙ্গেও ','সব ','সবার ','সবাইর ','সমস্ত ',
                       'সম্প্রতি ','সহ ','সহিত ','সবই ','সাধারণ ','সামনে ','সুতরাং ','সবাইর ','সে ','সেই ','সেখান ','সেখানে ','সেটা ', 'সেটাই ','সেটাও ','সেটি ','স্পষ্ট ','স্বয়ং ','হইতে ','হইবে ','হইয়া ','হওয়া ','হওয়ায় ','হওয়ার ','হচ্ছে ','হত ','হতে ',
                       'লেগেছে ','হতেই ','হন ','হইত ','হবে ','তিনি ','হবেন ','হয় ','হয়তো ','হয়নি ','হয়ে ','হয়েই ','হয়েছিল ','হয়েছে ','হয়েছেন ','হল ','হলে ','হলেই ','হলেও ','হলো ','হাজার ','হিসাবে ','হৈলে ','হোক ','হয় ']

digits=['০ ','১ ','২ ','৩ ','৪ ','৫ ','৬ ','৭ ','৮ ','৯ ']


df=pd.read_csv('data/bangla_comments_tokenized.csv')
df = df.fillna(0)
df['label'] = df['label'].replace({'not bully':'acceptable'})
sample_data = [2000,5000,10000,20000,30000,40000]

def label_encoding(category,bool):
  le = preprocessing.LabelEncoder()
  le.fit(category)
  encoded_labels = le.transform(category)
  labels = np.array(encoded_labels) # Converting into numpy array
  class_names =le.classes_ ## Define the class names again
  if bool == True:
    for i in sample_data:
        return labels

df.labels = label_encoding(df.label,True)

def demoji(text):
	emoji_pattern = re.compile("["
		u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U00010000-\U0010ffff"
	                           "]+", flags=re.UNICODE)
	return(emoji_pattern.sub(r'', text)) 


def clean(text):
    text = re.sub('[%s]' % re.escape(punctuations), ' ', text)     #escape punctuation
    text = re.sub('\n', ' ', text)                                 #replace line break with space
    text = re.sub('\w*\d\w*', ' ', text)                           #ignore digits
    #text = re.sub('\xa0', ' ', text)                              
    return text

def remove_punct(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[\u09E6-\u09FF]+', ' ', text)                  #remove bangla punctuations
    return text

def stemming_text(corpus):
    stm = BanglaStemmer()
    return [' '.join([stm.stem(word) for word in review.split()]) for review in corpus]

def pos_tagging(doc):
  bn_pos = POS()
  model_path = "data/bn_pos.pkl"
  doc = bn_pos.tag(model_path,doc)
  return doc

app = Flask(__name__)
Bootstrap(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    df.text=df.text.apply(str)
    X = df.text.values
    y = df.labels
    if request.method == 'POST':
        comment = request.form['namequery']
        comment = [comment]
        comment_df = pd.DataFrame (comment, columns = ['text'])
        comment_df[u'text'] = comment_df[u'text'].astype(str)
        comment_df[u'text'] = comment_df[u'text'].apply(lambda x:demoji(x))
        comment_df['text'] = comment_df['text'].apply(lambda x: re.split('http:\/\/.*', str(x))) #remove urls
        comment_df["text"] = comment_df['text'].apply(lambda x: clean(str(x)))                      
        comment_df['text'] = comment_df['text'].apply(lambda x: remove_punct(x))
        #remove special characters
        for char in spec_chars:
            comment_df['text'] = comment_df['text'].str.replace(char, ' ') 
            comment_df['text'] = comment_df['text'].str.split().str.join(' ')  
        
        final_stopword_list = custom_stop_word_list 
        pat = r'\b(?:{})\b'.format('|'.join(final_stopword_list))
        comment_df['text'] = comment_df['text'].str.replace(pat, ' ')
        comment_df['text'] = comment_df['text'].str.replace(r'\s+', ' ')
        comment_df['text'] = stemming_text(comment_df['text'])
        b_token = BasicTokenizer()
        comment_df['tokenized_stem_text'] = comment_df.apply(lambda row: b_token.tokenize(row['text']), axis=1)
        comment_df['token_length'] = comment_df.apply(lambda row: len(row['tokenized_stem_text']), axis=1)
        actual = ' '.join(comment)
        cleanText = ' '.join(comment_df.text.values.tolist())
        length = ' '.join(str(e) for e in comment_df.apply(lambda row: len(row['tokenized_stem_text']), axis=1)) 
        tokens =' '.join(str(e) for e in comment_df['tokenized_stem_text'].values)
        comment_df['tokenized_stem_text'] = comment_df.tokenized_stem_text.astype(str)
        def get_postags(row):
            postags = pos_tagging(row["tokenized_stem_text"])
            list_classes = list()
            for  word in postags:
                list_classes.append(word[1])
            return list_classes
        comment_df["postags_list"] = comment_df.apply(lambda row: get_postags(row), axis = 1)
        comment_df['postags_list'] = comment_df.postags_list.apply(lambda x: [i for i in x if i != 'PU'])
        comment_df['postags_list'] = comment_df.postags_list.apply(lambda x: [i for i in x if i != 'RDS'])
       

        def find_no_class(count, class_name = ""):
            total = 0
            for key in count.keys():
                if key.startswith(class_name):
                    total += count[key]        
            return total


        def get_classes(row, grammatical_class = ""):
            count = Counter(row["postags_list"])
            try:
                return find_no_class(count, class_name = grammatical_class)/len(row["postags_list"])
            except ZeroDivisionError:
                return find_no_class(count, class_name = grammatical_class)
    
        comment_df["freqAdverbs"] = comment_df.apply(lambda row: get_classes(row, "AMN"), axis = 1)
        comment_df["freqPreposition"] = comment_df.apply(lambda row: get_classes(row, "PP"), axis = 1)
        comment_df["freqPronoun"] = comment_df.apply(lambda row: get_classes(row, "PPR"), axis = 1)
        comment_df["freqVerbs"] = comment_df.apply(lambda row: get_classes(row, "VM"), axis = 1)
        comment_df["freqAdjectives"] = comment_df.apply(lambda row: get_classes(row, "JJ"), axis = 1)
        comment_df["freqNouns"] = comment_df.apply(lambda row: get_classes(row, ("NC")), axis = 1)
        comment_df["freqEnglish"] = comment_df.apply(lambda row: get_classes(row, ("RDF")), axis = 1)
        pos =' '.join(str(e) for e in comment_df['postags_list'].values)
        if request.form['ML'] =='ML' :
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state = 42)
            TFIDF_SGD_pipeline = Pipeline([
                               
                ('tfidf', TfidfVectorizer(tokenizer=lambda x: x.split(),lowercase=False,max_features=13000,min_df=1,ngram_range=(1,2))),
                ('clf', OneVsRestClassifier(SGDClassifier(loss='hinge', penalty='l2', alpha=1e-4, random_state=42, max_iter=200, tol=None)))
            ])
            TFIDF_SGD_pipeline.fit(X_train, y_train)
            prediction = TFIDF_SGD_pipeline.predict(comment_df['text'])

    return render_template('index.html',pos= pos, actual = actual, tokens = tokens, length=length, prediction = prediction, cleanText = cleanText )


@app.route('/overview')
def overview():
    return render_template('overview.html')


if __name__ == '__main__':
    app.run(debug=True)