from flask import Flask, request, jsonify, render_template
import torch
import similarity_cal
from skipgram import Skipgram,get_embed

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/similarity', methods=['POST'])
def similarity():
    model = torch.load("model_skipgram", weights_only=False)
    # try:
    data = request.json
    input_word = data.get('word')
    get_embed(model,[5163])
    sim_score = similarity_cal.cosine_similarity(model.embedding_center(torch.LongTensor([5163])),model.embedding_center(torch.LongTensor([5163])))

    if not input_word:
        return jsonify({"error": "Word is required"}), 400

    # Replace the following line with your model's similarity computation logic
    similar_word,  = "example_word"

    similarity_score = sim_score

    return jsonify({
        "input_word": input_word,
        "similar_word": sim_score,
        "similarity_score": similarity_score
    })
    # except Exception as e:
    #     return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)



