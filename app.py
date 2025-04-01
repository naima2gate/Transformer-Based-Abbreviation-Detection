import logging
from logging.handlers import RotatingFileHandler
import torch
from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Set up logging
log_formatter = logging.Formatter('%(asctime)s - %(message)s')
log_handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=1)
log_handler.setFormatter(log_formatter)
log_handler.setLevel(logging.INFO)

app = Flask(__name__)
app.logger.setLevel(logging.INFO)
app.logger.addHandler(log_handler)

path = "model"
tokenizer = AutoTokenizer.from_pretrained('roberta-base', add_prefix_space=True)
model = AutoModelForTokenClassification.from_pretrained(path, local_files_only=True)

model.eval()


def align_predictions_with_words(words, token_predictions, offsets):
    word_predictions = []
    current_labels = []
    word_idx = 0
    for idx, (label, offset) in enumerate(zip(token_predictions, offsets)):
        if offset[0] == 0 and offset[1] == 0:
            continue
        if len(current_labels) == 0:
            current_labels.append(label)
        elif offset[0] == offsets[idx - 1][1]:
            current_labels.append(label)
        else:
            if word_idx < len(words):
                word_predictions.append((words[word_idx], max(set(current_labels), key=current_labels.count)))
                current_labels = [label]
                word_idx += 1

    if word_idx < len(words):
        word_predictions.append((words[word_idx], max(set(current_labels), key=current_labels.count)))

    return word_predictions


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input text from the request
        input_data = request.json
        text = input_data['text']

        # Tokenize the input text
        inputs = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        offsets = inputs["offset_mapping"].squeeze().tolist()

        # Perform prediction
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Get the prediction for each token
        predictions = torch.argmax(outputs.logits, dim=2).squeeze().tolist()

        # Convert prediction indices to labels
        id2label = {0: "B-O", 1: "B-AC", 2: "I-AC", 3: "B-LF", 4: "I-LF"}
        label_predictions = [id2label[pred] for pred in predictions]

        # Remove special tokens (CLS, SEP, etc.)
        special_tokens_mask = tokenizer.get_special_tokens_mask(input_ids.squeeze().tolist(),
                                                                already_has_special_tokens=True)
        filtered_predictions = [label for label, mask in zip(label_predictions, special_tokens_mask) if mask == 0]
        filtered_tokens = [token for token, mask in
                           zip(tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist()), special_tokens_mask) if
                           mask == 0]
        filtered_offsets = [offset for offset, mask in zip(offsets, special_tokens_mask) if mask == 0]

        # Align predictions with words
        words = text.split()
        word_predictions = align_predictions_with_words(words, filtered_predictions, filtered_offsets)

        # Log the input and predictions
        log_message = f"Input: {text} | Predictions: {word_predictions}"
        app.logger.info(log_message)

        # Return the result as JSON
        return jsonify({'predictions': [{word: label} for word, label in word_predictions]})
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
