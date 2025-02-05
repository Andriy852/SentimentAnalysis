# Sentiment classification of tweets project

## Author: Andrii Zhurba

## Description
This project focuses on sentiment classification of posts on the Twitter platform. The dataset consists of 1.6 million tweets, and the goal is to predict whether each tweet is positive or negative. The target variable is balanced, and our primary metric for model evaluation was accuracy. Additionally, we tracked Precision, Recall, and ROC AUC scores to assess model performance comprehensively.

## Results 
We trained different models. The best ones are:
* Logistic Regression:
    * 79.4% accuracy 
* Long Short-Term Memory:
    * 79.8% accuracy
* Bert transformer:
    * 79.6% accuracy

While all models delivered decent results, Logistic Regression was the most efficient in terms of both training time and prediction speed compared to LSTM and BERT. The BERT Transformer model could potentially be enhanced by adding additional layers, though this would significantly increase computation time.

## Data
The data for this project can be found at this [link](https://www.kaggle.com/datasets/kazanova/sentiment140/data).

To run the notebooks, install the following dependencies:

## Technologies Used

- **Python**: Programming language used for the project.
- **TensorFlow**: Deep learning framework used for training the model.
- **Transformers (by Hugging Face)**: Library used for working with BERT.
- **Scikit-learn**: Machine learning library used for data processing and model evaluation.
- **Pandas**: Data manipulation library used for handling and preprocessing the dataset.
- **NumPy**: Library for numerical operations and array handling.
- **Matplotlib/Seaborn**: Visualization libraries used for plotting graphs and metrics.
- **Jupyter Notebook**: Interactive environment for running the code and documenting the project.
- **Google Colab / Kaggle**: Cloud-based platforms used for training the models.

```bash
pip install -r requirements.txt