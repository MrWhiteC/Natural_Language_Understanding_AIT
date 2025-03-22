from flask import Flask, request, jsonify, render_template
import torch
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import datasets
import numpy as np 

device = "mps"

teacher_id = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(teacher_id)
model_use = AutoModelForSequenceClassification.from_pretrained("bert_model_odd")
model_use = model_use.to(device)
model_use.eval()

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/similarity', methods=['POST'])
def similarity():
    # try:
    data = request.json
    input_word = data.get('word')
    inputs = tokenizer(input_word, return_tensors="pt")
    with torch.no_grad():
        logits = model_use(**inputs.to(device)).logits

    # Convert logits to class probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probs).item()

    if predicted_class == 1:
        classify = "Toxic"
    else:
        classify = "Non-Toxic"

    if not input_word:
        return jsonify({"error": "Word is required"}), 400

    # Replace the following line with your model's similarity computation logic

    return jsonify({
        "input_word": input_word,
        "classify": classify
    })
    # except Exception as e:
    #     return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)



