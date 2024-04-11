# Scaling Implicit Bias Analysis across Transformer-based Language Models

## Abstract

In the evolving field of machine learning, deploying fair and transparent models remains a formidable challenge. This study builds on earlier research, demonstrating that neural architectures exhibit inherent biases by analyzing a broad spectrum of transformer-based language models from base to extra-large configurations. This article investigates movie reviews for genre-based bias, which leverages the Word Embedding Association Test (WEAT), revealing that scaling models up tends to mitigate bias, with larger models showing up to a 29% reduction in prejudice. Additionally, this study underscores the effectiveness of prompt-based learning, a facet of prompt engineering, as a practical approach to bias mitigation, reducing genre bias in reviews by more than 37% on average. This suggests that the refinement of development practices should include the strategic use of prompts in shaping model outputs, highlighting the crucial role of ethical AI integration to weave fairness seamlessly into the core functionality of transformer models.

## Keywords

AI, Model Scaling, Generative Pretrained Transformer (GPT), K-Means, Bidirectional Encoder Representations from Transformers (BERT), Transformer, Prompt Engineering, Language Models, Word Embedding Association Test (WEAT), Natural Language Processing (NLP)

## Dataset

This project utilizes the *IMDB Movie Reviews Dataset* to analyze bias across different language models. 

## Installation

To set up your environment to run these notebooks and scripts, follow the steps below:

```bash
# Clone the repository
git clone https://github.com/Deep6Lab/Bias-Analysis
cd Bias-Analysis

# Set up a Python virtual environment (recommended)
python -m venv env
source env/bin/activate  # On Windows use env\Scripts\activate

# Install required packages
pip install numpy pandas matplotlib seaborn scikit-learn nltk tensorflow keras pytorch transformers beautifulsoup4 wordcloud colorsys

# Additional dependency for transformers
pip install -U transformers

# Install PyTorch (visit the official website https://pytorch.org/ for custom installation options)
pip install torch torchvision torchaudio

# Run the script
python main.py 