"""
SmartNews NLP Pipeline
"""

import re
import string
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Machine Learning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

# Deep Learning
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    pipeline
)

# Spacy for NER
import spacy

# For summarization
from heapq import nlargest


class NewsAnalyzerPipeline:
    """
    Main NLP pipeline that handles all analysis tasks.
    
    This class demonstrates:
    - Object-oriented programming in Python
    - Multiple NLP techniques
    - Model management
    - Error handling
    """
    
    def __init__(self, use_transformers=True):
        """
        Initialize the NLP pipeline with all necessary models.
        
        Args:
            use_transformers (bool): If True, use transformer models (slower but better)
                                     If False, use classical ML (faster but less accurate)
        """
        print("🚀 Initializing SmartNews Analyzer...")
        
        self.use_transformers = use_transformers
        self.lemmatizer = WordNetLemmatizer()
        
        # Download required NLTK data
        self._download_nltk_data()
        
        # Initialize stopwords
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize sentiment analyzers
        self.vader = SentimentIntensityAnalyzer()
        
        # Load spaCy model for NER
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("✅ spaCy model loaded")
        except:
            print("⚠️  Downloading spaCy model (one-time setup)...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize classification models
        if use_transformers:
            self._load_transformer_models()
        else:
            self.classifier = None  # Will be trained on first use
            self.vectorizer = TfidfVectorizer(max_features=5000)
        
        print("✅ Pipeline initialized successfully!\n")
    
    def _download_nltk_data(self):
        """Download required NLTK datasets"""
        required_data = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
        for data in required_data:
            try:
                nltk.data.find(f'tokenizers/{data}')
            except LookupError:
                print(f"Downloading {data}...")
                nltk.download(data, quiet=True)
    
    def _load_transformer_models(self):
        """
        Load pre-trained transformer models from HuggingFace.
        
        These models are already trained on millions of examples.
        We use them directly (zero-shot) or with minimal fine-tuning.
        """
        print("📥 Loading transformer models (this may take a minute)...")
        
        # For classification - using a model fine-tuned for news classification
        self.classifier_pipeline = pipeline(
            "text-classification",
            model="fabriceyhc/bert-base-uncased-ag_news",
            device=-1  # Use CPU (-1), change to 0 for GPU
        )
        
        # For sentiment analysis - using a sentiment-specific model
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1
        )
        
        # For summarization
        self.summarizer_pipeline = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=-1
        )
        
        print("✅ Transformer models loaded")
    
    # ==================== TEXT PREPROCESSING ====================
    
    def preprocess_text(self, text: str, remove_stopwords=True) -> str:
        """
        Clean and normalize text for NLP processing.
        
        Steps:
        1. Convert to lowercase
        2. Remove URLs, emails, special characters
        3. Tokenize into words
        4. Remove stopwords (optional)
        5. Lemmatize (convert words to base form)
        
        Args:
            text (str): Raw text to process
            remove_stopwords (bool): Whether to remove common words
            
        Returns:
            str: Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove mentions and hashtags (for social media text)
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords if specified
        if remove_stopwords:
            tokens = [word for word in tokens if word not in self.stop_words]
        
        # Lemmatize (e.g., "running" -> "run", "better" -> "good")
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        
        # Remove very short words (likely not meaningful)
        tokens = [word for word in tokens if len(word) > 2]
        
        return ' '.join(tokens)
    
    # ==================== TEXT CLASSIFICATION ====================
    
    def classify_article(self, text: str) -> Dict[str, any]:
        """
        Classify news article into categories.
        
        Categories: World, Sports, Business, Sci/Tech
        
        Args:
            text (str): Article text
            
        Returns:
            dict: Classification results with category and confidence
        """
        if not text:
            return {"category": "Unknown", "confidence": 0.0}
        
        if self.use_transformers:
            # Use pre-trained transformer model
            result = self.classifier_pipeline(text[:512])[0]  # Limit to 512 tokens
            
            # Map AG News labels to readable categories
            label_map = {
                "World": "World News",
                "Sports": "Sports",
                "Business": "Business",
                "Sci/Tech": "Technology"
            }
            
            category = label_map.get(result['label'], result['label'])
            
            return {
                "category": category,
                "confidence": round(result['score'], 3),
                "method": "BERT Transformer"
            }
        else:
            # Use classical ML (TF-IDF + Naive Bayes)
            # This requires training data - simplified version shown
            cleaned_text = self.preprocess_text(text)
            
            # For demo, use simple keyword matching
            keywords = {
                "Technology": ["tech", "software", "computer", "ai", "digital", "app"],
                "Sports": ["game", "player", "team", "win", "score", "championship"],
                "Business": ["market", "stock", "company", "business", "economy"],
                "Politics": ["government", "election", "president", "policy", "vote"],
                "Health": ["health", "medical", "doctor", "disease", "hospital"],
                "Entertainment": ["movie", "music", "celebrity", "show", "film"]
            }
            
            scores = {}
            for category, words in keywords.items():
                score = sum(1 for word in words if word in cleaned_text)
                scores[category] = score
            
            category = max(scores, key=scores.get)
            confidence = scores[category] / max(sum(scores.values()), 1)
            
            return {
                "category": category,
                "confidence": round(confidence, 3),
                "method": "Keyword Matching"
            }
    
    # ==================== SENTIMENT ANALYSIS ====================
    
    def analyze_sentiment(self, text: str) -> Dict[str, any]:
        """
        Analyze sentiment of the text using multiple methods.
        
        Combines:
        1. VADER (rule-based, good for social media)
        2. TextBlob (pattern-based)
        3. Transformer model (context-aware)
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Sentiment scores and overall sentiment
        """
        if not text:
            return {"sentiment": "neutral", "scores": {}}
        
        results = {}
        
        # 1. VADER Sentiment (fast, rule-based)
        vader_scores = self.vader.polarity_scores(text)
        results['vader'] = {
            'compound': vader_scores['compound'],
            'positive': vader_scores['pos'],
            'negative': vader_scores['neg'],
            'neutral': vader_scores['neu']
        }
        
        # 2. TextBlob Sentiment (pattern-based)
        blob = TextBlob(text)
        results['textblob'] = {
            'polarity': blob.sentiment.polarity,  # -1 to 1
            'subjectivity': blob.sentiment.subjectivity  # 0 to 1
        }
        
        # 3. Transformer-based (if enabled)
        if self.use_transformers:
            try:
                transformer_result = self.sentiment_pipeline(text[:512])[0]
                results['transformer'] = {
                    'label': transformer_result['label'],
                    'score': transformer_result['score']
                }
            except:
                results['transformer'] = None
        
        # Determine overall sentiment (ensemble approach)
        vader_sentiment = (
            'positive' if vader_scores['compound'] > 0.05
            else 'negative' if vader_scores['compound'] < -0.05
            else 'neutral'
        )
        
        textblob_sentiment = (
            'positive' if blob.sentiment.polarity > 0.1
            else 'negative' if blob.sentiment.polarity < -0.1
            else 'neutral'
        )
        
        # Majority vote
        sentiments = [vader_sentiment, textblob_sentiment]
        if self.use_transformers and results.get('transformer') and results['transformer'] is not None:
            transformer_label = results['transformer']['label'].lower()
            if transformer_label in ['positive', 'negative', 'neutral']:
                sentiments.append(transformer_label)
        
        overall = max(set(sentiments), key=sentiments.count)
        
        return {
            "sentiment": overall,
            "confidence": round(abs(vader_scores['compound']), 3),
            "detailed_scores": results
        }
    
    # ==================== NAMED ENTITY RECOGNITION ====================
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities (people, places, organizations) from text.
        
        Uses spaCy's pre-trained NER model which can identify:
        - PERSON: People's names
        - ORG: Organizations, companies
        - GPE: Countries, cities, states
        - DATE: Dates and time expressions
        - MONEY: Monetary values
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Entities grouped by type
        """
        if not text:
            return {}
        
        # Process with spaCy
        doc = self.nlp(text)
        
        # Group entities by type
        entities = {
            'persons': [],
            'organizations': [],
            'locations': [],
            'dates': [],
            'money': [],
            'other': []
        }
        
        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                entities['persons'].append(ent.text)
            elif ent.label_ == 'ORG':
                entities['organizations'].append(ent.text)
            elif ent.label_ == 'GPE':
                entities['locations'].append(ent.text)
            elif ent.label_ == 'DATE':
                entities['dates'].append(ent.text)
            elif ent.label_ == 'MONEY':
                entities['money'].append(ent.text)
            else:
                entities['other'].append(f"{ent.text} ({ent.label_})")
        
        # Remove duplicates while preserving order
        for key in entities:
            entities[key] = list(dict.fromkeys(entities[key]))
        
        return entities
    
    # ==================== TEXT SUMMARIZATION ====================
    
    def summarize_text(self, text: str, num_sentences: int = 3) -> Dict[str, str]:
        """
        Generate a summary of the text using two methods.
        
        1. Extractive: Select most important sentences from original text
        2. Abstractive: Generate new sentences (if transformers enabled)
        
        Args:
            text (str): Text to summarize
            num_sentences (int): Number of sentences in summary
            
        Returns:
            dict: Both extractive and abstractive summaries
        """
        if not text:
            return {"extractive": "", "abstractive": ""}
        
        # EXTRACTIVE SUMMARIZATION (TF-IDF based)
        sentences = sent_tokenize(text)
        
        if len(sentences) <= num_sentences:
            extractive_summary = text
        else:
            # Calculate sentence scores using TF-IDF
            sentence_scores = {}
            words = word_tokenize(text.lower())
            word_freq = {}
            
            for word in words:
                if word not in self.stop_words and word not in string.punctuation:
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Normalize
            max_freq = max(word_freq.values()) if word_freq else 1
            for word in word_freq:
                word_freq[word] = word_freq[word] / max_freq
            
            # Score sentences
            for sent in sentences:
                for word in word_tokenize(sent.lower()):
                    if word in word_freq:
                        sentence_scores[sent] = sentence_scores.get(sent, 0) + word_freq[word]
            
            # Get top sentences
            summary_sentences = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
            extractive_summary = ' '.join(summary_sentences)
        
        # ABSTRACTIVE SUMMARIZATION (Transformer-based)
        abstractive_summary = ""
        if self.use_transformers and len(text) > 100:
            try:
                # BART model works best with 1024 tokens max
                result = self.summarizer_pipeline(
                    text[:1024],
                    max_length=130,
                    min_length=30,
                    do_sample=False
                )
                abstractive_summary = result[0]['summary_text']
            except Exception as e:
                abstractive_summary = f"Summarization failed: {str(e)}"
        
        return {
            "extractive": extractive_summary,
            "abstractive": abstractive_summary if abstractive_summary else ""
        }
    
    # ==================== COMPLETE ANALYSIS ====================
    
    def analyze_article(self, text: str) -> Dict:
        """
        Perform complete NLP analysis on an article.
        
        This is the main function that combines all capabilities.
        
        Args:
            text (str): Article text
            
        Returns:
            dict: Complete analysis results
        """
        print(f"\n📊 Analyzing article ({len(text)} characters)...")
        
        results = {
            "original_text": text,
            "word_count": len(text.split()),
            "char_count": len(text)
        }
        
        # 1. Preprocess
        results["cleaned_text"] = self.preprocess_text(text, remove_stopwords=False)
        
        # 2. Classification
        results["classification"] = self.classify_article(text)
        print(f"   ✓ Category: {results['classification']['category']}")
        
        # 3. Sentiment Analysis
        results["sentiment"] = self.analyze_sentiment(text)
        print(f"   ✓ Sentiment: {results['sentiment']['sentiment']}")
        
        # 4. Named Entity Recognition
        results["entities"] = self.extract_entities(text)
        entity_count = sum(len(v) for v in results["entities"].values())
        print(f"   ✓ Entities found: {entity_count}")
        
        # 5. Summarization
        results["summary"] = self.summarize_text(text)
        print(f"   ✓ Summary generated")
        
        print("✅ Analysis complete!\n")
        
        return results


def download_models():
    """
    Download all required models for first-time setup.
    Run this once before using the pipeline.
    """
    print("📦 Downloading required models...")
    
    # spaCy model
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    
    # NLTK data
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    
    # Pre-cache transformer models
    from transformers import pipeline
    
    print("Downloading BERT classifier...")
    pipeline("text-classification", model="fabriceyhc/bert-base-uncased-ag_news")
    
    print("Downloading sentiment analyzer...")
    pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    
    print("Downloading summarizer...")
    pipeline("summarization", model="facebook/bart-large-cnn")
    
    print("✅ All models downloaded!")


# Example usage
if __name__ == "__main__":
    # Demo the pipeline
    sample_text = """
    Apple Inc. announced record quarterly earnings today, with CEO Tim Cook 
    praising the success of the iPhone 15 launch. The Cupertino-based tech 
    giant reported revenue of $119.6 billion, exceeding Wall Street expectations. 
    The strong performance was driven by robust sales in China and India, 
    despite ongoing supply chain challenges. Investors reacted positively, 
    pushing the stock up 5% in after-hours trading.
    """
    
    # Initialize pipeline
    analyzer = NewsAnalyzerPipeline(use_transformers=True)
    
    # Analyze
    results = analyzer.analyze_article(sample_text)
    
    # Display results
    print("=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)
    print(f"\nCategory: {results['classification']['category']}")
    print(f"Confidence: {results['classification']['confidence']}")
    print(f"\nSentiment: {results['sentiment']['sentiment']}")
    print(f"\nKey Entities:")
    for entity_type, entities in results['entities'].items():
        if entities:
            print(f"  {entity_type.title()}: {', '.join(entities)}")
    print(f"\nSummary:\n{results['summary']['extractive']}")
