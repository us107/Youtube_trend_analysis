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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analysis')
def analysis():
    try:
        # Read CSV with proper path
        data_path = os.path.join(BASE_DIR, 'data', 'youtube_trending_cleaned.csv')
        print(f"Looking for file at: {data_path}")  # Debug print
        
        data = pd.read_csv(data_path)
        
        # Create figure with larger size and adjusted layout
        plt.figure(figsize=(12, 6))
        
        # Generate a bar chart of views by channel
        channel_data = data.groupby('Channel')['Views'].sum().reset_index()
        
        # Create the bar plot
        plt.clf()  # Clear any existing plots
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=channel_data, x='Channel', y='Views', ax=ax)
        
        # Customize the plot
        ax.set_title('Total Views by Channel', pad=20, fontsize=14)
        ax.set_xlabel('Channel', fontsize=12)
        ax.set_ylabel('Total Views', fontsize=12)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Convert plot to PNG image
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight', dpi=300)
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        
        # Close the figure to free memory
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

@app.route('/predictions', methods=['GET'])
def predict_form():
    # Render a template with a form to submit prediction requests
    return render_template('predictions.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # For JSON data (API request)
        if request.is_json:
            data = request.get_json()
            # Validate input data
            if not data or 'features' not in data:
                return jsonify({'error': 'Invalid input data'}), 400
            
            # Extract features from the incoming request
            features = np.array(data['features']).reshape(1, -1)
        else:
            # For form submission
            features = [
                float(request.form.get(f'feature{i}', 0))  # Correct feature names as "feature1", "feature2", etc.
                for i in range(1, 4)  # Update range based on your number of features
            ]
            features = np.array(features).reshape(1, -1)
        
        # Make predictions using the models
        lr_prediction = linear_regression_model.predict(features)
        kmeans_cluster = kmeans_model.predict(features)
        
        # Return the predictions to the user
        if request.is_json:
            return jsonify({
                'linear_regression_prediction': float(lr_prediction[0]),
                'kmeans_cluster': int(kmeans_cluster[0])
            })
        else:
            return render_template(
                'predict_results.html',
                lr_prediction=lr_prediction[0],
                kmeans_cluster=kmeans_cluster[0]
            )
    
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print(error_message)
        if request.is_json:
            return jsonify({'error': str(e)}), 500
        else:
            return render_template('error.html', message=error_message)

# Add error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html', message='Page not found'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', message='Internal server error'), 500

if __name__ == '__main__':
    app.run(debug=True)
