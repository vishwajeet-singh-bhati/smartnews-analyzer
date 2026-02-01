# 🍎 Mac Installation Guide - SmartNews Analyzer

## Step-by-Step Installation for Mac

### Step 1: Verify Python Installation

```bash
# Check Python version
python3 --version

# Should show Python 3.8 or higher
# If not installed, download from python.org
```

### Step 2: Navigate to Project Directory

```bash
cd /path/to/smartnews-analyzer
# Or wherever you saved the files
```

### Step 3: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# You should see (venv) in your terminal prompt
```

### Step 4: Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all packages (this will take 5-10 minutes)
pip install -r requirements.txt

# If you get any errors, try installing packages individually:
pip install transformers
pip install torch
pip install spacy
pip install flask
pip install nltk scikit-learn pandas numpy
pip install plotly matplotlib seaborn
pip install textblob vaderSentiment
```

### Step 5: Download NLP Models

```bash
# Download spaCy model
python3 -m spacy download en_core_web_sm

# Download NLTK data
python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
```

### Step 6: Test Installation

```bash
# Run test suite
python3 test_project.py

# This will verify everything is working
```

### Step 7: Run the Application

```bash
# Start the web server
python3 app.py

# Open browser to: http://localhost:5000
```

## Common Mac Issues & Solutions

### Issue 1: "Command not found: python"

**Solution:** Use `python3` instead of `python`

```bash
python3 app.py
```

### Issue 2: "Permission denied"

**Solution:** Use user installation

```bash
pip install --user -r requirements.txt
```

### Issue 3: PyTorch Installation on Apple Silicon (M1/M2/M3)

**Solution:** Install Apple Silicon optimized version

```bash
# For M1/M2/M3 Macs
pip3 install torch torchvision torchaudio
```

### Issue 4: "xcrun: error"

**Solution:** Install Xcode Command Line Tools

```bash
xcode-select --install
```

### Issue 5: Port 5000 Already in Use

**Solution:** macOS AirPlay uses port 5000

```bash
# Option 1: Disable AirPlay Receiver
# System Settings > General > AirDrop & Handoff > AirPlay Receiver (turn off)

# Option 2: Use different port
# In app.py, change last line to:
# app.run(debug=True, host='0.0.0.0', port=5001)
```

### Issue 6: "SSL Certificate Error"

**Solution:** Update certificates

```bash
# Install certifi
pip install --upgrade certifi

# Run certificate installation
/Applications/Python\ 3.*/Install\ Certificates.command
```

## Quick Install Script (Copy-Paste)

Save this as `install_mac.sh` and run `bash install_mac.sh`:

```bash
#!/bin/bash

echo "🍎 Installing SmartNews Analyzer on Mac..."

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install packages
echo "Installing Python packages (this may take 5-10 minutes)..."
pip install transformers torch spacy flask flask-cors
pip install nltk scikit-learn pandas numpy
pip install plotly matplotlib seaborn wordcloud
pip install textblob vaderSentiment gensim
pip install python-dotenv tqdm joblib colorama
pip install beautifulsoup4 requests newsapi-python

# Download models
echo "Downloading NLP models..."
python3 -m spacy download en_core_web_sm

# Download NLTK data
python3 << EOF
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
EOF

echo "✅ Installation complete!"
echo "Run: python3 app.py"
```

## Performance Optimization for Mac

### For M1/M2/M3 Macs (Apple Silicon)

The project will automatically use CPU, which is fine. If you want to use Metal Performance Shaders:

```python
# In nlp_pipeline.py, you can try:
import torch
device = "mps" if torch.backends.mps.is_available() else "cpu"

# However, for this project, CPU mode works perfectly fine
```

### Memory Management

If you experience slowdowns:

```python
# In nlp_pipeline.py, change to:
self.use_transformers = False  # Uses faster classical ML
```

## Recommended Mac Setup

```bash
# 1. Use iTerm2 or Terminal
# 2. Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 3. Install Python via Homebrew (optional)
brew install python@3.11

# 4. Create project directory
mkdir -p ~/Projects/smartnews-analyzer
cd ~/Projects/smartnews-analyzer

# 5. Download project files here
# 6. Follow installation steps above
```

## Running on Mac

```bash
# Always activate virtual environment first
cd ~/Projects/smartnews-analyzer
source venv/bin/activate

# Run the app
python3 app.py

# Open browser
open http://localhost:5000
```

## Deactivating Virtual Environment

```bash
# When done
deactivate
```

## Updating Dependencies

```bash
# Activate environment
source venv/bin/activate

# Update all packages
pip install --upgrade -r requirements.txt
```

## Uninstalling

```bash
# Remove virtual environment
rm -rf venv

# Remove downloaded models
rm -rf ~/Library/Caches/huggingface
```

## Troubleshooting Checklist

- [ ] Python 3.8+ installed (`python3 --version`)
- [ ] Virtual environment activated (`source venv/bin/activate`)
- [ ] All packages installed (`pip list`)
- [ ] spaCy model downloaded (`python3 -m spacy validate`)
- [ ] Port 5000 available (check AirPlay settings)
- [ ] No firewall blocking localhost

## Getting Help

If you encounter issues:

1. Check error message carefully
2. Search error on Stack Overflow
3. Try installing packages individually
4. Use Python 3.11 (most stable)

---

**Mac-Specific Note:** If you have an M1/M2/M3 Mac, everything will work but might be slightly slower than on Intel Macs due to model compatibility. The app will automatically use CPU mode which is perfectly fine for this project.

Good luck! 🚀
