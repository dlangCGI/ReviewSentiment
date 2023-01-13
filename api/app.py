from flask import Flask, render_template, request, jsonify, make_response
from review_classification.predict import make_prediction_raw
from flask_cors import CORS


app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})


@app.route('/', methods=['GET', 'POST'])
def predict():
    """View that handles the default get request and the post request when
    a form is submitted."""

    # when the form is submitted this route gets a POST request
    if request.method == 'POST':
        review_text = request.form["text_input"]

        # if no input provided
        if not review_text.strip():
            return render_template('index.html', error="No text provided.")

        # catch any error during prediction
        try:
            predictions = make_prediction_raw([review_text])
        except Exception as e:
            return render_template('index.html', error=e)

        prediction = 'positive' if predictions[0] == 1 else 'negative'
        # finally render template with correct sentiment
        return render_template('index.html', prediction=prediction)

    # this is the 'home' route with a get request (no form submitted)
    else:
        return render_template('index.html')


@app.route('/api/sentiment/v1', methods=['POST'])
def predict_api():
    """
    JSON Response for requests over api
    waiting:
        '{"input": "text"}'
    returning:
        '{"prediction": "positive", "error" : ""}'
    """

    # when the form is submitted this route gets a POST request
    if request.method == 'POST':
        review_text = request.json["input"]

        # if no input provided
        if not review_text.strip():
            return jsonify({'prediction': "", 'error': "No input text."}), 400

        # catch any error during prediction
        try:
            predictions = make_prediction_raw([review_text])
        except Exception as _:
            return jsonify({'prediction': "", 'error': "Something wrong in server."}), 500

        prediction = 'positive' if predictions[0] == 1 else 'negative'
        # finally render template with correct sentiment
        return jsonify({'prediction': prediction})
    elif request.method == 'OPTIONS':
        # temporary solution for cross site resource sharing
        # later, a library can be used.
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response


if __name__ == '__main__':
    app.run()
