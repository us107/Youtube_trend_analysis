from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import os
from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)

# Define base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load models with proper paths
try:
    linear_regression_model = joblib.load(os.path.join(BASE_DIR, 'models', 'linear_regression_model.pkl'))
    kmeans_model = joblib.load(os.path.join(BASE_DIR, 'models', 'kmeans_model.pkl'))
except FileNotFoundError as e:
    print(f"Error loading models: {e}")

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Analysis route
@app.route('/analysis')
def analysis():
    try:
        data_path = os.path.join(BASE_DIR, 'data', 'youtube_trending_cleaned.csv')
        print(f"Looking for file at: {data_path}")  # Debug print
        data = pd.read_csv(data_path)

        # Bar chart of views by channel
        channel_data = data.groupby('Channel')['Views'].sum().reset_index()
        plt.clf()  # Clear any existing plots
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=channel_data, x='Channel', y='Views', ax=ax)
        ax.set_title('Total Views by Channel', pad=20, fontsize=14)
        ax.set_xlabel('Channel', fontsize=12)
        ax.set_ylabel('Total Views', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Convert plot to PNG image
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight', dpi=300)
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        plt.close()

        return render_template('results.html', plot_url=plot_url)
    except FileNotFoundError:
        error_message = f"Error: Could not find data file at {data_path}"
        print(error_message)
        return render_template('error.html', message=error_message)
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print(error_message)
        return render_template('error.html', message=error_message)

# Predictions form
@app.route('/predictions', methods=['GET'])
def predictions_form():
    # Render a form for the user to input data
    return render_template('predictions.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Handle form or JSON submission
        if request.is_json:
            data = request.get_json()
            features = pd.DataFrame([data['features']], columns=['feature1', 'feature2', 'feature3'])
        else:
            features = pd.DataFrame([[
                float(request.form.get('feature1', 0)),
                float(request.form.get('feature2', 0)),
                float(request.form.get('feature3', 0))
            ]], columns=['feature1', 'feature2', 'feature3'])

        # Make predictions
        lr_prediction = linear_regression_model.predict(features)
        kmeans_cluster = kmeans_model.predict(features)

        # Return results
        return render_template('predict_results.html', 
                               lr_prediction=lr_prediction[0], 
                               kmeans_cluster=kmeans_cluster[0])
    except Exception as e:
        return render_template('error.html', message=str(e)), 500



# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html', message='Page not found'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', message='Internal server error'), 500

if __name__ == '__main__':
    app.run(debug=True)
