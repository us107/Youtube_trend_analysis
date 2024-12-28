from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

app = Flask(__name__)

# Load models
linear_regression_model = joblib.load('models/linear_regression_model.pkl')
kmeans_model = joblib.load('models/kmeans_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analysis')
def analysis():
    # Example: Generate visualizations from the analysis
    data = pd.read_csv('your_trending_video_data.csv')  # Replace with your actual data

    # Generate a bar chart of views by category
    category_data = data.groupby('category')['views'].sum().reset_index()
    fig, ax = plt.subplots()
    sns.barplot(data=category_data, x='category', y='views', ax=ax)
    ax.set_title('Total Views by Category')

    # Convert plot to PNG image to display on webpage
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    return render_template('results.html', plot_url=plot_url)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Extract features from the incoming request
    features = np.array(data['features']).reshape(1, -1)

    # Make predictions using the models
    lr_prediction = linear_regression_model.predict(features)
    kmeans_cluster = kmeans_model.predict(features)

    # Return the predictions as JSON response
    return jsonify({
        'linear_regression_prediction': lr_prediction[0],
        'kmeans_cluster': int(kmeans_cluster[0])
    })

if __name__ == '__main__':
    app.run(debug=True)
