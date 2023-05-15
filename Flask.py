from flask import Flask, request, jsonify, render_template
import main

app = Flask(__name__, template_folder='templates')

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/recommend', methods=['GET'])
def recommend():
    # Get the book input from the query parameters
    book_input = request.args.get('book_input')

    # Check if the book is available
    result = main.check_book_availability(book_input)

    # If a book or related books are available, recommend the top 3 similar books
    if result:
        cosine_sim, keyword_data = result
        recommended_books = main.recommend_book(cosine_sim, keyword_data)
        print("These are: "+recommended_books)
        return jsonify({'recommended_books': recommended_books})
    else:
        return jsonify({'error': 'Book not available'})


if __name__ == '__main__':
    app.run(debug=True)