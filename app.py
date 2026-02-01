"""
SmartNews Analyzer - Web Application
=====================================
Flask web app that provides an interactive interface for the NLP pipeline.

Features:
- Upload or paste article text
- Real-time analysis with loading indicators
- Beautiful visualizations
- Download results as JSON/PDF
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import json
from datetime import datetime
import os
import nltk
from flask import Flask, render_template # and your other imports...

# ADD THIS LINE HERE:
nltk.download('punkt_tab')

# These are likely already there:
nltk.download('stopwords')
nltk.download('wordnet')
from nlp_pipeline import NewsAnalyzerPipeline

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable cross-origin requests

# Initialize NLP pipeline (global to avoid reloading models on each request)
print("🚀 Initializing NLP Pipeline...")
analyzer = NewsAnalyzerPipeline(use_transformers=False)
print("✅ Server ready!\n")

# Store analysis history (in production, use a database)
analysis_history = []


@app.route('/')
def home():
    """Render the main page"""
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Main analysis endpoint.
    
    Accepts JSON with 'text' field and returns complete NLP analysis.
    """
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text or len(text.strip()) < 50:
            return jsonify({
                'error': 'Please provide text with at least 50 characters'
            }), 400
        
        # Perform analysis
        results = analyzer.analyze_article(text)
        
        # Add metadata
        results['timestamp'] = datetime.now().isoformat()
        results['id'] = len(analysis_history) + 1
        
        # Store in history
        analysis_history.append(results)
        
        # Keep only last 50 analyses
        if len(analysis_history) > 50:
            analysis_history.pop(0)
        
        return jsonify({
            'success': True,
            'results': results
        })
    
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        return jsonify({
            'error': f'Analysis failed: {str(e)}'
        }), 500


@app.route('/history', methods=['GET'])
def get_history():
    """Get analysis history"""
    return jsonify({
        'history': analysis_history,
        'count': len(analysis_history)
    })


@app.route('/stats', methods=['GET'])
def get_stats():
    """
    Get aggregate statistics from analysis history.
    Useful for dashboard visualizations.
    """
    if not analysis_history:
        return jsonify({
            'message': 'No analyses yet'
        })
    
    # Calculate statistics
    stats = {
        'total_analyses': len(analysis_history),
        'category_distribution': {},
        'sentiment_distribution': {},
        'avg_word_count': 0,
        'total_entities': 0
    }
    
    for analysis in analysis_history:
        # Category distribution
        category = analysis['classification']['category']
        stats['category_distribution'][category] = \
            stats['category_distribution'].get(category, 0) + 1
        
        # Sentiment distribution
        sentiment = analysis['sentiment']['sentiment']
        stats['sentiment_distribution'][sentiment] = \
            stats['sentiment_distribution'].get(sentiment, 0) + 1
        
        # Average word count
        stats['avg_word_count'] += analysis['word_count']
        
        # Total entities
        for entities in analysis['entities'].values():
            stats['total_entities'] += len(entities)
    
    stats['avg_word_count'] = round(stats['avg_word_count'] / len(analysis_history))
    
    return jsonify(stats)


@app.route('/health', methods=['GET'])
def health_check():
    """Check if the service is running"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'analyses_performed': len(analysis_history)
    })


# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    # Run the app
    print("\n" + "="*60)
    print("🌐 SmartNews Analyzer Server Starting...")
    print("="*60)
    print("\n📍 Access the dashboard at: http://localhost:5001")
    print("📍 API endpoint: http://localhost:5001/analyze")
    print("\n💡 Tip: Use Ctrl+C to stop the server\n")
    
    app.run(
        debug=True,  # Enable debug mode for development
        host='0.0.0.0',  # Allow external connections
        port=5001
    )
