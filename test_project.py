
##Test Suite for SmartNews Analyzer


import sys
import time
from colorama import init, Fore, Style

# Initialize colorama for colored terminal output
init()

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"{Fore.CYAN}{Style.BRIGHT}{text}{Style.RESET_ALL}")
    print("="*60)

def print_success(text):
    """Print success message"""
    print(f"{Fore.GREEN}✓ {text}{Style.RESET_ALL}")

def print_error(text):
    """Print error message"""
    print(f"{Fore.RED}✗ {text}{Style.RESET_ALL}")

def print_info(text):
    """Print info message"""
    print(f"{Fore.YELLOW}ℹ {text}{Style.RESET_ALL}")


def test_imports():
    """Test if all required packages are installed"""
    print_header("Testing Package Imports")
    
    packages = [
        ('transformers', 'Hugging Face Transformers'),
        ('torch', 'PyTorch'),
        ('spacy', 'spaCy'),
        ('nltk', 'NLTK'),
        ('sklearn', 'scikit-learn'),
        ('flask', 'Flask'),
        ('pandas', 'Pandas'),
        ('plotly', 'Plotly')
    ]
    
    all_passed = True
    
    for package, name in packages:
        try:
            __import__(package)
            print_success(f"{name} imported successfully")
        except ImportError as e:
            print_error(f"{name} import failed: {e}")
            all_passed = False
    
    return all_passed


def test_nltk_data():
    """Test NLTK data downloads"""
    print_header("Testing NLTK Data")
    
    import nltk
    
    datasets = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
    all_passed = True
    
    for dataset in datasets:
        try:
            nltk.data.find(f'tokenizers/{dataset}')
            print_success(f"NLTK {dataset} data found")
        except LookupError:
            print_info(f"Downloading {dataset}...")
            nltk.download(dataset, quiet=True)
            print_success(f"NLTK {dataset} downloaded")
    
    return all_passed


def test_spacy_model():
    """Test spaCy model"""
    print_header("Testing spaCy Model")
    
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        
        # Test with sample text
        doc = nlp("Apple Inc. is located in Cupertino, California.")
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        if len(entities) > 0:
            print_success(f"spaCy model working - Found {len(entities)} entities")
            for text, label in entities:
                print(f"   - {text} ({label})")
            return True
        else:
            print_error("spaCy model loaded but found no entities")
            return False
            
    except OSError:
        print_error("spaCy model not found")
        print_info("Downloading en_core_web_sm...")
        import subprocess
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print_success("spaCy model downloaded")
        return True
    except Exception as e:
        print_error(f"spaCy test failed: {e}")
        return False


def test_nlp_pipeline():
    """Test the NLP pipeline"""
    print_header("Testing NLP Pipeline")
    
    try:
        from nlp_pipeline import NewsAnalyzerPipeline
        
        # Initialize with simple models (faster for testing)
        print_info("Initializing pipeline (this may take a minute)...")
        analyzer = NewsAnalyzerPipeline(use_transformers=False)
        print_success("Pipeline initialized")
        
        # Test text
        sample_text = """
        Apple Inc. announced strong quarterly results today. CEO Tim Cook 
        expressed optimism about future growth. The tech giant's stock rose 
        5% on the positive news from Cupertino headquarters.
        """
        
        # Test preprocessing
        print_info("Testing text preprocessing...")
        cleaned = analyzer.preprocess_text(sample_text)
        if len(cleaned) > 0:
            print_success(f"Preprocessing works - Output: {len(cleaned)} chars")
        else:
            print_error("Preprocessing failed")
            return False
        
        # Test classification
        print_info("Testing classification...")
        classification = analyzer.classify_article(sample_text)
        if 'category' in classification:
            print_success(f"Classification works - Category: {classification['category']}")
        else:
            print_error("Classification failed")
            return False
        
        # Test sentiment
        print_info("Testing sentiment analysis...")
        sentiment = analyzer.analyze_sentiment(sample_text)
        if 'sentiment' in sentiment:
            print_success(f"Sentiment analysis works - Sentiment: {sentiment['sentiment']}")
        else:
            print_error("Sentiment analysis failed")
            return False
        
        # Test NER
        print_info("Testing named entity recognition...")
        entities = analyzer.extract_entities(sample_text)
        entity_count = sum(len(v) for v in entities.values())
        print_success(f"NER works - Found {entity_count} entities")
        
        # Test summarization
        print_info("Testing summarization...")
        summary = analyzer.summarize_text(sample_text)
        if 'extractive' in summary:
            print_success(f"Summarization works")
        else:
            print_error("Summarization failed")
            return False
        
        # Full analysis test
        print_info("Testing complete analysis...")
        results = analyzer.analyze_article(sample_text)
        if all(key in results for key in ['classification', 'sentiment', 'entities', 'summary']):
            print_success("Complete analysis works!")
            return True
        else:
            print_error("Complete analysis missing some components")
            return False
            
    except Exception as e:
        print_error(f"Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_transformer_models():
    """Test transformer models (optional - can be slow)"""
    print_header("Testing Transformer Models (Optional)")
    
    print_info("This test downloads large models and may take 5-10 minutes")
    response = input("Do you want to test transformers? (y/n): ").lower()
    
    if response != 'y':
        print_info("Skipping transformer tests")
        return True
    
    try:
        from nlp_pipeline import NewsAnalyzerPipeline
        
        print_info("Initializing with transformers...")
        analyzer = NewsAnalyzerPipeline(use_transformers=True)
        
        sample_text = "The company announced record profits despite economic challenges."
        
        print_info("Testing transformer classification...")
        result = analyzer.classify_article(sample_text)
        print_success(f"Transformer classification works - {result['method']}")
        
        print_info("Testing transformer sentiment...")
        result = analyzer.analyze_sentiment(sample_text)
        if result.get('detailed_scores', {}).get('transformer'):
            print_success("Transformer sentiment works")
        else:
            print_info("Transformer sentiment not available")
        
        return True
        
    except Exception as e:
        print_error(f"Transformer test failed: {e}")
        return False


def test_sample_data():
    """Test sample data loading"""
    print_header("Testing Sample Data")
    
    try:
        from sample_data import get_sample_dataset, get_random_article
        
        df = get_sample_dataset()
        print_success(f"Sample dataset loaded - {len(df)} articles")
        
        article = get_random_article()
        print_success(f"Random article retrieved - Category: {article['category']}")
        
        return True
        
    except Exception as e:
        print_error(f"Sample data test failed: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print_header("🧪 SmartNews Analyzer - Test Suite")
    print("This will verify your installation and test all components\n")
    
    start_time = time.time()
    
    tests = [
        ("Package Imports", test_imports),
        ("NLTK Data", test_nltk_data),
        ("spaCy Model", test_spacy_model),
        ("NLP Pipeline", test_nlp_pipeline),
        ("Sample Data", test_sample_data),
        ("Transformer Models", test_transformer_models)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except KeyboardInterrupt:
            print("\n\nTests interrupted by user")
            sys.exit(0)
        except Exception as e:
            print_error(f"Unexpected error in {test_name}: {e}")
            results[test_name] = False
    
    # Summary
    print_header("Test Summary")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        if result:
            print_success(f"{test_name}")
        else:
            print_error(f"{test_name}")
    
    elapsed_time = time.time() - start_time
    
    print(f"\n{Fore.CYAN}Results: {passed}/{total} tests passed{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Time: {elapsed_time:.1f} seconds{Style.RESET_ALL}")
    
    if passed == total:
        print(f"\n{Fore.GREEN}{Style.BRIGHT}🎉 All tests passed! Your setup is ready.{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Run 'python app.py' to start the web application{Style.RESET_ALL}")
    else:
        print(f"\n{Fore.YELLOW}⚠️  Some tests failed. Check the errors above.{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}You may still be able to run the project with limited functionality.{Style.RESET_ALL}")


if __name__ == "__main__":
    try:
        run_all_tests()
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        sys.exit(0)
