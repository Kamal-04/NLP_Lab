!pip install nltk

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
sample_text = "this is a text ready to tokenize"
tokens = word_tokenize(sample_text)
print(tokens)


from nltk.tokenize import TweetTokenizer
tweet_tokenizer = TweetTokenizer()
sample_text = "This is a tweet @jack #NLP"
tokens = tweet_tokenizer.tokenize(sample_text)
print(tokens)


from nltk.tokenize import sent_tokenize
sample_text = "This is a sentence. This is another one!\nAnd this is the last one."
sentences = sent_tokenize(sample_text)
print(sentences)

import nltk
from nltk.corpus import stopwords
stopwords_ = set(stopwords.words("english"))
sample_text = "this is a sample text"
tokens = sample_text.split()
clean_tokens = [t for t in tokens if not t in stopwords_]
clean_text = " ".join(clean_tokens)
print(sample_text, clean_text)


#pip install spacy
#python -m spacy download en_core_web_sm
import spacy
nlp = spacy.load("en_core_web_sm")

sample_text = "this is a text ready to tokenize"
doc = nlp(sample_text)
tokens = [token.text for token in doc]
print(sample_text, tokens)


import spacy
nlp = spacy.load("en_core_web_sm")
sample_text = "This is a sentence. This is another one!\nAnd this is the last one"
doc = nlp(sample_text)
sentences = [sentence.text for sentence in doc.sents]
print(sample_text, sentences)
