from flask import Flask, current_app, request, jsonify
import io
from model import predict_class
import base64
import logging

app = Flask(__name__)
@app.route('/', methods=['POST'])
def predict():
    data = {}
    try:
        data = request.get_json()['data']
    except Exception:
        return jsonify(status_code='400', msg='Bad Request'), 400

    data = base64.b64decode(data)

    image = io.BytesIO(data)
    predictions = predict_class(image)
    current_app.logger.info('Predictions: %s', predictions)
    return jsonify(predictions=predictions)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)

