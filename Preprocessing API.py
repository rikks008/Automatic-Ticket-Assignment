#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import re
import unicodedata
import nltk
import spacy
from nltk.tokenize.toktok import ToktokTokenizer
stopword_list = nltk.corpus.stopwords.words('english')
tokenizer = ToktokTokenizer()
nlp = spacy.load('en_core_web_sm', parse = False, tag=False, entity=False)
from contractions import CONTRACTION_MAP
import nltk
nltk.download('words')
words = set(nltk.corpus.words.words())



# In[2]:


stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.extend((['please','receiv','dear','company','from','sent','to','subject',
                          'unabl','need','pleas','issu','compani','widia',
                          'regards','see','phone','thanks',
                          'thankyou','yes','no','na','mii','hii','hello','hi','help',
                          'ewew','hui','amssm','via','dly','plz','pls','gmail','name','received','sid_','name']))


# In[3]:


# Remove emails 
def remove_emails(text):
    text = re.sub(r'\b[^\s]+@[^\s]+[.][^\s]+\b', ' ', text)
    return text

def remove_hyperlink(text):
    text=re.sub(r'(http|https)://[^\s]*',' ',text)
    return text

# Removing Digits
def remove_digits(text):
    #text= re.sub(r"\b\d+\b", "", text)
    text= re.sub(r"(\s\d+)", " ", text)
    return text
    
#Validate special charater count and remove if more than a specific count
def removeSpecialCharCount(text,count):
    length = len(re.findall(r'[^a-zA-Z0-9\s]',text))
    if (count >= length):
        text = re.sub('[^a-zA-Z0-9\s]', ' ', text)
        return text
    else:
        return text

# Removing Special Characters
def remove_special_characters(text):
    text = re.sub('[^a-zA-Z0-9_\s]', ' ', text)
    return text


# removing accented charactors
def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

def remove_received_from(text):
    testString = 'received from:'
    if re.search(testString,text):
        text=text.replace(testString,'')
        return text
    else:
        return text
    
 # Removing Stopwords
def remove_stopwords(wordList,text,is_lower_case):
    #For removing the given list of words from stopword list
    for i in wordList:
        if i in stopword_list:
            stopword_list.remove(i)   
    
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)   
    return filtered_text

# Lemmetization
def lemmatize_text(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text


# # Expanding Contractions

def expand_contractions(s, contractions_dict=CONTRACTION_MAP):
        contractions_re = re.compile('(%s)' % '|'.join(CONTRACTION_MAP.keys()))
        def replace(match):
            return contractions_dict[match.group(0)]
        return contractions_re.sub(replace, s)

#removing non english words
def remove_non_english_words(doc):
    doc = " ".join(w for w in nltk.wordpunct_tokenize(doc) if w.lower() in words or not w.isalpha())
    return doc

def remove_username(text, is_lower_case=False):
    userList = ['username','user.name','user-name','user name']
    tokens_user_list = tokenizer.tokenize(userList)
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in tokens_user_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in tokens_user_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

def remove_SID(text):
    text = re.sub(r'(sid|SID)+(_*\d*)*', ' ', text)
    return text

def removeCaller(doc,caller_name):
    tokens_caller = tokenizer.tokenize(caller_name)
    tokens_doc = tokenizer.tokenize(doc)
    filtered_tokens = [token for token in tokens_doc if token not in tokens_caller]
    filtered_text = ' '.join(filtered_tokens)   
    return filtered_text


# In[4]:


# # Text preprocessing
def text_preprocessing(corpus,isRemoveEmail,isRemoveDigits,isRemoveHyperLink,isSpecialCharVal, 
                     specCharCount,isRemoveSpecialCharac,isRemoveAccentChar, 
                     isRemoveReceived,isRemoveUsername,text_lower_case,contraction_expansion,
                     isRemoveNonEnglish,text_lemmatization, stopword_removal,wordList,isSID,caller_corpus):
    
    normalized_corpus = []
    
    for doc,caller_name in zip(corpus,caller_corpus):
        
        if text_lower_case:
            doc = doc.lower()
        
        if isRemoveEmail:
            doc = remove_emails(doc)
        
        if isRemoveHyperLink:
            doc=remove_hyperlink(doc)
        
        if isRemoveSpecialCharac:
            doc=remove_special_characters(doc)
       
        if contraction_expansion:
            doc = expand_contractions(doc.lower())
        
        if isSpecialCharVal:
            doc=removeSpecialCharCount(doc,specCharCount)
        
        if isRemoveReceived:
            doc=remove_received_from(doc)
        
        if isRemoveUsername:
            doc = remove_username(doc)
             
        if isRemoveAccentChar:
            doc = remove_accented_chars(doc)
            
        if isSID:
            doc= remove_SID(doc)
       
        if isRemoveDigits:
            doc = remove_digits(doc)
        
        # remove extra newlines
        doc = re.sub(r'[\r|\n|\r\n]+', ' ',doc)
        # insert spaces between special characters to isolate them    
        special_char_pattern = re.compile(r'([{.(-)!}])')
        doc = special_char_pattern.sub(" \\1 ", doc)
        
        if text_lemmatization:
            doc = lemmatize_text(doc)
        
        removeuser=True 
        if removeuser:
            doc = removeCaller(doc,caller_name)
        
        # remove extra whitespace
        doc = re.sub(' +', ' ', doc)
        
        if stopword_removal:
            doc = remove_stopwords(wordList,doc,is_lower_case=text_lower_case)
                
        normalized_corpus.append(doc)
        
    return normalized_corpus

