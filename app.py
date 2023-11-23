# importing necessary packages
import pandas as pd
import numpy as np
import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import argparse
import pickle
import time

# Verifying and downloading necessary ntlk corpus
if not nltk.corpus.stopwords.words('english'):
    nltk.download('stopwords')

try:
    nltk.corpus.wordnet.ensure_loaded()
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# creating necessary class objects
xgb = XGBClassifier()
lemmatizer = WordNetLemmatizer()
vec = TfidfVectorizer()

# list of columns for data validation
colList = ['Text','IsBold','IsItalic','IsUnderlined','Left',
            'Right','Top','Bottom','FontType','Label']

def dataValidate(df):
    '''
    This method validates if the provided dataset 
    has all the required columns
    
    Input: DataFrame
    
    Output: Independent & Target features as X and y
    '''
    df = df.filter(colList, axis=1)
    if list(df.columns) != colList:
        return 0
    
    X = df.drop("Label",axis = 1)
    y = df["Label"]
    return X,y

def textClean(text):
    '''
    This method implements text preprocessing steps
    
    Input: text string
    
    Output: Processed string
    '''
    text = str(text)
    text = re.sub(r'[^a-zA-Z0-9]',' ',text)
    text = text.lower()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text if not word in set(stopwords.words('english'))]
    text = ' '.join(text)
    return text

def updateDF(X,train):
    '''
    This method vectorises the text and
    onehotencode the FontType columns to 
    prepare the data for model fitting
    
    Input: DataFrame of independent features, flag (train or test data)
    
    Output: Processed DataFrame
    '''
    print('Preprocessing Data')
    X["Text"] = X["Text"].apply(lambda x: textClean(x))
    if train:
        tfidf = vec.fit_transform(X["Text"])
        pickle.dump(vec, open('vector.pkl', 'wb'))
        tfidf = tfidf.toarray()
        vecNames = vec.get_feature_names_out()
    else:
        tfidfVec = pickle.load(open('vector.pkl', 'rb'))
        tfidf = tfidfVec.transform(X["Text"]).toarray()
        vecNames = tfidfVec.get_feature_names_out()
            
    tfidf = pd.DataFrame(tfidf,columns=vecNames)
    X = X.drop("Text",axis = 1)
    X = pd.concat([X,tfidf],axis = 1)
    X = pd.get_dummies(X, columns=['FontType'], prefix='FontType')
    return X

def modelTraining(trainData):
    '''
    This method traines the model
    
    Input: Training Data Path
    
    Output: Model
    '''
    df = pd.read_csv(trainData,encoding='latin')
    print('Validating Data')
    if not dataValidate(df):
        parser.error(f"Invalid data")
    X,y = dataValidate(df)
    X = updateDF(X,train = 1)
    print('Fitting the model')
    xgb.fit(X, y)
    print('Saving the model')
    pickle.dump(xgb, open('model.pkl', 'wb'))

def predict(testData):
    '''
    This method implements classification using saved model
    
    Input: Test Dataset Path
    
    Output: Accuracy Report
    '''
    model = pickle.load(open('model.pkl', 'rb'))
    print('Validating Data')
    df = pd.read_csv(testData,encoding='latin')
    if not dataValidate(df):
        parser.error(f"Invalid data")
    X,y = dataValidate(df)
    X = updateDF(X,train = 0)
    print('Making Predictions')
    preds = model.predict(X)
    print('Classification Report')
    print(classification_report(y, preds))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Title Classification')

    # input for retraining
    parser.add_argument('--train', action='store_true', help='Model re-training')
    parser.add_argument('--data', help='Path to the training data')

    # input for prediction
    parser.add_argument('--predict', action='store_true', help='Making predictions using the trained model')
    parser.add_argument('--test-data', help='Path to the test data')

    args = parser.parse_args()

    # for model re-training
    if args.train:
        if not args.data:
            parser.error('--train requires --data')
        if not args.data.endswith('.csv'):
            parser.error(f"Invalid file type.")
        modelTraining(args.data)

    # for making predictions
    elif args.predict:
        if not args.test_data:
            parser.error('--predict requires --test-data')
        predict(args.test_data)

    else:
        parser.print_help()