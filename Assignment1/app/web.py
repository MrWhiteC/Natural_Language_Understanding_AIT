from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/similarity', methods=['POST'])
def similarity():
    try:
        data = request.json
        input_word = data.get('word')

        if not input_word:
            return jsonify({"error": "Word is required"}), 400

        # Replace the following line with your model's similarity computation logic
        similar_word, similarity_score = "example_word", 0.85

        return jsonify({
            "input_word": input_word,
            "similar_word": similar_word,
            "similarity_score": similarity_score
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
