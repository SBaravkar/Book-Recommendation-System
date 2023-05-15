import requests
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from keybert import KeyBERT
import pandas as pd
from flask import Flask, request, render_template

book_input = ""
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


keyword_strings = []
# Get the data in the desired format
data = pd.read_csv('Sample.csv', usecols=['title', 'description', 'ratings', 'ISBN'])
data = pd.DataFrame({
    'title': data['title'].astype(str),  # str.lower()
    'description': data['description'].astype(str),
    'ratings': data['ratings'].astype(str),
    'ISBN': data['ISBN'].astype(str)
})
data = data.reset_index(drop=True)
book_indices = pd.Series(data.index, index=data['title'])


# print(book_indices)


# Filter the description by getting all tokens and the eliminating the stop words
def token():
    # Tokenize the descriptions
    tokens = []
    for i, row in data.iterrows():
        desc = row['description']
        desc_tokens = word_tokenize(desc)
        tokens.append(desc_tokens)

    # Remove stop words and lemmatize the tokens
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    filtered_tokens = []
    for desc_tokens in tokens:
        filtered_tokens.append(
            [lemmatizer.lemmatize(word.lower()) for word in desc_tokens if word.lower() not in stop_words])
    return filtered_tokens


import re


def keyword():
    # Extract keywords
    model = KeyBERT('distilbert-base-nli-mean-tokens')
    keywords = []
    filtered_tokens = token()
    for i, tokens in enumerate(filtered_tokens):
        doc = ' '.join(tokens)
        doc_keywords = model.extract_keywords(doc, keyphrase_ngram_range=(1, 1), stop_words='english', use_maxsum=True,
                                              nr_candidates=20, top_n=2)
        keyword_list = [key for key, _ in doc_keywords]
        synonyms = []
        for keyword in keyword_list:
            # Split keywords based on underscores
            split_keywords = re.split('_+', keyword)
            for split_keyword in split_keywords:
                synsets = wordnet.synsets(split_keyword)
                for synset in synsets:
                    for lemma in synset.lemmas():
                        synonyms.append(lemma.name())
            synonyms = list(set(synonyms))
            split_keywords.extend(synonyms)
            keywords.append(split_keywords)
    # print("keywords::",keywords)
    return keywords


# Check if book is available as book name or similar to the entered keyword
def check_book_availability(book_input):
    keywords = keyword()
    print(keywords)
    if book_input in data['title'].values:
        book_available = True
        book_title = book_input
        print(book_input, 'is available')
        return vectorization(book_available)
    elif any(book_input.lower() in ' '.join(keywords[i]).lower() for i in range(len(keywords))):
        related_book_available = True
        print('Books related to [', book_input, '] are available')
        return vectorization(related_book_available)
    else:
        print('Book not available')
        return False


# Vectorization and cosine similarities
def vectorization(flag):
    if flag:
        keywords = keyword()
        for keywords_list in keywords:
            keyword_string = ' '.join(keywords_list)
            keyword_strings.append(keyword_string)
        # Vectorize descriptions and calculate cosine similarity
        tfidf = TfidfVectorizer()
        desc_tfidf = tfidf.fit_transform(keyword_strings)
        cosine_sim = cosine_similarity(desc_tfidf)

        # Create dictionary with book titles as keys and keywords, ratings as values
        keyword_data = {}
        for i in range(len(data['title'])):
            keyword_data[data['title'][i]] = keywords[i], [data['ratings'][i]], data['ISBN'][i]
        print("keyword_data", keyword_data)
        return recommend_book(cosine_sim, keyword_data)


# Recommend top 3 Similar books and sort by highest ratings
def recommend_book(cosine_sim, keyword_data):
    # Get the index of the input book or keyword
    if book_input not in book_indices:
        keys = [key for key in keyword_data.keys() if book_input.lower() in keyword_data[key][0]]
        book_detail = {}
        for i in keys:
            for key, value in keyword_data.items():
                # print("Value:::",value)
                if i == key:
                    ratings = [int(r) for r in value[1]]
                    isbn = ''.join(value[2])
                    book_detail[i] = max(ratings), int(isbn)
        # print("book_detail::", book_detail)
        sorted_books = list(sorted(book_detail.items(), key=lambda x: x[1], reverse=True))
        # print("Sorted books", sorted_books)

        # sorted_books_list = [{'title': book[0], 'ratings': str(book[1][0]), 'ISBN': str(book[0][1])} for book in sorted_books][:3]
        sorted_books_list = [{'title': book[0], 'ratings': str(book[1][0]), 'ISBN': str(book[1][1])} for book in
                             sorted_books][:3]
        print(sorted_books_list)
        return sorted_books_list

    else:
        book_idx = book_indices[book_input]  # gets the ID of book to find cosine similarity
        print("book_idx", book_idx)

        # Get the cosine similarity scores for the input book
        sim_scores = list(enumerate(cosine_sim[book_idx]))

        # Sort the book indices based on the cosine similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # print("sim_scores:",sim_scores)

        top_books_indices = [i[0] for i in sim_scores if i[1] > 0.5]
        print("top_books_indices", top_books_indices)
        valid_indices = data.index.isin(top_books_indices)
        top_books = data.loc[valid_indices, ['title', 'ratings', 'ISBN']]
        top_three_books = top_books.sort_values('ratings', ascending=False).head(3)
        # print("top_three_books",top_three_books)
        recommended_books = top_three_books.loc[
            top_three_books['title'] != book_input, ['title', 'ratings', 'ISBN']].to_dict('records')
        user_book = data.loc[data['title'] == book_input, ['title', 'ratings', 'ISBN']].to_dict('records')
        print("::::::::::", user_book)
        if len(recommended_books) != 0:
            print("Books similar to", book_input, "are:", recommended_books)
            # print(recommended_books)
        if len(user_book) != 0:
            user_book.extend(recommended_books)
            return user_book
        print("user_book:", user_book)
        return recommended_books


@app.route('/recommend')
def recommended_books():
    global book_input
    book_input = request.args.get('book_input')     #.lower()
    if len(book_input) != 0:
        books = check_book_availability(book_input)
        # if (books == False):
        #     book_input = book_input + " is not available"
        # else:
        #     book_input = book_input # + " is available"
    else:
        book_input = "Please enter the book name or the keyword."
        books = 0
    return render_template('result.html', book_input=book_input, books=books)


if __name__ == '__main__':
    app.run(port=5000, )
