# Classification in Viva Interactions: Distinguishing Questions, Answers, and Statements

## Overview
This repository contains a comprehensive machine learning and natural language processing (NLP) project aimed at classifying viva (oral) interactions. The primary objective is to accurately distinguish between three distinct types of conversational turns:
- **Questions**
- **Answers**
- **Statements**

Understanding and categorizing these interactions is crucial for analyzing educational dialogue, automated assessment systems, and conversational AI.

**Key Result:** The system achieves its best results using Support Vector Machines (SVM) combined with BERT embeddings, yielding an impressive **F1-score of 0.81**, ensuring accurate segmentation and reliable performance in classifying these interactions.

## Techniques & Models Explored
This project explores a wide range of text representation and classification techniques, ranging from traditional machine learning approaches to state-of-the-art transformer-based models. 

### Text Embeddings & Representations
- **TF-IDF (Term Frequency-Inverse Document Frequency):** Traditional statistical measure used to evaluate how important a word is to a document in a corpus.
- **FastText:** Word embedding technique that treats each word as composed of character n-grams, allowing for better handling of out-of-vocabulary words.
- **Byte-Pair Encoding (BPE):** Subword tokenization method that iteratively replaces the most frequent pair of bytes in a sequence.

### Transformer Models
- **BERT (Bidirectional Encoder Representations from Transformers):** Pre-trained transformer model for deep contextualized sequence representations.
- **RoBERTa (Robustly Optimized BERT Pretraining Approach):** An optimized method for pretraining NLP systems that improves on BERT's language masking strategy.
- **SBERT (Sentence-BERT):** Modification of the pre-trained BERT network that uses siamese and triplet network structures to derive semantically meaningful sentence embeddings.
- **T5 (Text-to-Text Transfer Transformer):** An architecture that casts every NLP problem as a text-to-text task.

### Classifiers
Classical machine learning algorithms (such as SVM, Random Forest, Logistic Regression, Naive Bayes, XGBoost) and neural networks are evaluated on top of these embeddings to determine the best-performing architecture for this task.

## Repository Structure
The repository consists of several Jupyter Notebooks, each dedicated to a specific model or approach:

- `TFid.ipynb` - Implementation using TF-IDF features and traditional ML classifiers.
- `fasttext.ipynb` - Text classification leveraging FastText embeddings.
- `Byte_pair.ipynb` - Approach utilizing Byte-Pair Encoding.
- `bert.ipynb` - Deep learning classification using standard BERT.
- `Robata.ipynb` - Deep learning classification using RoBERTa.
- `sbert.ipynb` - Implementation utilizing Sentence-BERT for sentence-level embeddings.
- `t5.ipynb` - Approach utilizing the T5 transformer model.
- `best accurate.ipynb` - The culmination notebook demonstrating the model with the highest accuracy and performance metrics, leveraging SVM with BERT embeddings.

## Requirements & Dependencies
To run the notebooks in this repository, you will need the following libraries:
- `Python 3.8+`
- `pandas` & `numpy` (Data manipulation)
- `scikit-learn` (Traditional ML & evaluation)
- `torch` (PyTorch for Deep Learning)
- `transformers` (Hugging Face ecosystem)
- `sentence-transformers`
- `fasttext`
- `xgboost`

## Usage
1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install pandas numpy scikit-learn torch transformers sentence-transformers fasttext xgboost
   ```
3. Open the Jupyter Notebooks in your preferred environment (Jupyter Lab, VS Code, Google Colab) to explore the various approaches. Check `best accurate.ipynb` for the optimal solution and its evaluation metrics.
