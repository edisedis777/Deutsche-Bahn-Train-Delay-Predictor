from flask import Flask, request, jsonify, render_template
import torch
import numpy as np
import os

app = Flask(__name__)

# Load model
model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'train_delay_model.pt')
model = torch.load(model_path, map_location=torch.device('cpu'))
model.eval()

# Define the same input features used during training
FEATURES = [
    "weekday", "hour",
    "station_id_BER", "station_id_MUC", "station_id_FRA",
    "train_type_ICE", "train_type_IC", "train_type_RE"
]

@app.route('/', methods=['GET'])
def home():
    return render_template('form.html')  # ✅ make sure to return form.html

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()

        # Initialize feature vector
        x_input = np.zeros(len(FEATURES))
        for i, feature in enumerate(FEATURES):
            if feature in input_data:
                x_input[i] = input_data[feature]

        x_tensor = torch.tensor([x_input], dtype=torch.float32)
        with torch.no_grad():
            output = model(x_tensor).item()

        return jsonify({'predicted_delay_minutes': round(output, 2)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
