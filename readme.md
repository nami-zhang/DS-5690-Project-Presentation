# Sentiment Analysis of Amazon Electronics Reviews Using BERT

This project focuses on analyzing customer sentiments in Amazon Electronics product reviews by leveraging the Bidirectional Encoder Representations from Transformers (BERT) model. By fine-tuning BERT for sentiment classification, we aim to accurately categorize reviews as positive, neutral, or negative, providing valuable insights into customer opinions.

## Overview

Understanding customer sentiment is crucial for businesses to assess product reception and identify areas for improvement. Traditional sentiment analysis methods often struggle with the nuances of human language, such as sarcasm and context. BERT, a transformer-based model developed by Google, captures contextual relationships between words, making it highly effective for natural language understanding tasks.

In this project, we fine-tuned a pre-trained BERT model on a labeled dataset of Amazon Electronics reviews. The dataset was preprocessed to map star ratings to sentiment labels:

- **1-2 stars**: Negative
- **3 stars**: Neutral
- **4-5 stars**: Positive

We then trained the model to classify reviews into these sentiment categories.

## Model Card

**Model Architecture**: BERT base model with a classification head

**Pretrained Model Name**: [BERT-base-uncased](https://huggingface.co/bert-base-uncased)

**Training Data**: [Amazon Review Data (2018)](https://nijianmo.github.io/amazon/index.html)
- Electronics 5-core, a subset of the complete dataset, was used in this project.

**Performance Metrics**:

- **Accuracy**: 78%
- **F1-Score**: 0.79 (negative), 0.69 (neutral), 0.86 (positive)

**Intended Use**: Classifying customer reviews into sentiment categories to gain insights into customer opinions

**Limitations**:

1. **Limited Data Source**:
   - The model is trained exclusively on the Amazon Electronics 5-core dataset, which may introduce domain-specific biases. For example, sentiment expressions in electronics reviews (e.g., "battery life is great") may not generalize well to other product categories like books or clothing.
2. **Limited Sentiment Categories**:
   - The sentiment classification is restricted to three categories: positive, neutral, and negative, oversimplifying the complexity of human emotions.
3. **Neutral Sentiment Challenges**:
   - Neutral sentiment lacks strong sentiment signals, making it harder to classify accurately.
4. **Limited Language Coverage**:
   - The dataset and model are primarily in English, limiting the model’s utility for non-English reviews.
  
**Permissions**: This project is licensed under the MIT License. 

## Critical Analysis

1. **Impact of the Project**:
   - Automating sentiment analysis saves time and resources, enabling businesses to quickly interpret customer opinions and respond to feedback.
   - By identifying common complaints or areas of excellence, companies can enhance their products and services, leading to higher customer satisfaction and loyalty.
2. **What Does It Reveal?**:
   - The results highlight the importance of domain-specific training data and the need for handling neutral sentiments more effectively.
3. **Next Steps**:
   - **Dataset Expansion**: Incorporate reviews from diverse categories and languages to enhance generalizability.
   - **Fine-Grained Sentiment Analysis**: Introduce more granular categories like “very positive” or “slightly negative.”
   - **Model Optimization**: Experiment with larger models like `RoBERTa` or lightweight transformer models like `DistilBERT`.
   - **Interactive Analysis**: Combine sentiment analysis with metadata insights (e.g., trends based on product categories).

## Resources

For further reading on BERT-based sentiment analysis, consider the following research papers:

1. **"BERT-Based Sentiment Analysis: A Software Engineering Perspective"**  
   *Authors*: Himanshu Batra, Narinder Singh Punn, Sanjay Kumar Sonbhadra, Sonali Agarwal  
   *Link*: [arXiv:2106.02581](https://arxiv.org/abs/2106.02581)

2. **"Sentiment Analysis Classification System Using Hybrid BERT Models"**  
   *Author*: Amira Samy Talaat  
   *Link*: [Journal of Big Data](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-023-00781-w)

3. **"Fine-Grained Sentiment Classification Using BERT"**  
   *Authors*: [Author Names]  
   *Link*: [IEEE Xplore](https://ieeexplore.ieee.org/document/8947435)

4. **"Research on the Application of Deep Learning-Based BERT Model in Sentiment Analysis"**  
   *Authors*: Yichao Wu, Zhengyu Jin, Chenxi Shi, Penghao Liang, Tong Zhan  
   *Link*: [arXiv:2403.08217](https://arxiv.org/abs/2403.08217)

These papers provide insights into various approaches and advancements in sentiment analysis using BERT models.

## Code Demonstration

A Jupyter notebook demonstrating the model training and evaluation process is available in the repository. It includes data preprocessing steps, model fine-tuning, and performance evaluation metrics.

## Gradio Demo

Experience the interactive demo to test the model with your own reviews:
[Gradio Demo Link](https://e7675153a9d668b668.gradio.live)

## Repository

The project repository includes:

- **README.md**: This document
- **main.ipynb**: Code demonstration in a Jupyter Notebook
- **[Electronics_5.json](https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Electronics_5.json.gz)**: Dataset is not included due to GitHub file size restrictions. Download and place in root directory.

## Video Recording

A short video overview of the project is available [here](https://drive.google.com/file/d/13t_-dMDWIgFcdYLLLMVoAfq_fA8B78Bu/view?usp=sharing).
