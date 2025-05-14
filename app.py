import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.tokenize import WhitespaceTokenizer
from nltk.tokenize import wordpunct_tokenize
from nltk.tokenize import blankline_tokenize
from nltk.util import bigrams,trigrams,ngrams
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.title("NLP APPLICATION")
st.header("Word Tokenization")

user_input=st.text_area("Enter your text here:")
if st.button("Tokenize"):
    tokens=word_tokenize(user_input)
    st.write("Tokens:",tokens)

st.header("Sentence Tokenization")

user_input2=st.text_area("Enter your text here: ",key="sentence tokenization")
if st.button("Tokenize Sentences"):
    sentences=sent_tokenize(user_input2)
    st.write("Sentences:",sentences)

st.header("Blankline Tokenization")
user_input3=st.text_area("Enter your text here:",key="blankline tokenization")
if st.button("Tokenize Blanklines"):
    blankline_tokens=blankline_tokenize(user_input3)
    st.write("Blankline Tokens:",blankline_tokens)

st.header("Whitespace Tokenization")
user_input4=st.text_area("Enter your text here:",key="whitespace tokenization")
if st.button("Tokenize Whitespace"):
    whitespace_tokens=WhitespaceTokenizer().tokenize(user_input4)
    st.write("Whitespace Tokens:",whitespace_tokens)

st.header("Word Punct Tokenization")
user_input5=st.text_area("Enter your text here:",key="word punct tokenization")
if st.button("Tokenize Word Punct"):
    word_punct_tokens=wordpunct_tokenize(user_input5)
    st.write("Word Punct Tokens:",word_punct_tokens)

utils=st.selectbox("Select a utility",('bigram','trigram','ngrams'))
user_input6=st.text_area("Enter your text here:",key="bigram/trigram/ngram")
tokens=nltk.word_tokenize(user_input6)
if utils=='bigram':
    st.header("Bigram Tokenization")
    if st.button("Tokenize Bigrams"):
        biagrams=list(nltk.bigrams(tokens))
        st.write("Bigrams:",biagrams)
    
    
if utils=='trigram':
    st.header("Trigram Tokenization")
    if st.button("Tokenize Trigrams"):
        trigrams=list(nltk.trigrams(tokens))
        st.write("Trigrams:",trigrams)

if utils=='ngrams':
    st.header("Ngram Tokenization")
    input_ngram=st.number_input("Enter the value of n",min_value=4)
    if st.button("Ngram Tokenize"):
        ngrams=list(nltk.ngrams(tokens,input_ngram))   
        st.write("Ngrams:",ngrams)
        
st.header("Word Cloud")
user_input7=st.text_area("Enter your text here:",key="word cloud")
if st.button("Generate Word Cloud"):
    wordcloud=WordCloud(height=400,width=300,background_color='black',colormap='plasma',mode='RGBA').generate(user_input7)
    plt.imshow(wordcloud,interpolation='bilinear',)
    plt.axis('off')
    plt.margins(x=0,y=0)
    plt.show()
    st.pyplot(plt)

