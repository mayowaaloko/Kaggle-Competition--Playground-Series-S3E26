## Kaggle Competition: Playground Series S3E26 - Multi-Class Prediction of Cirrhosis Outcomes

### Overview
This repository contains my solution for the **Kaggle Playground Series S3E26** competition. The goal of this competition is to predict the outcomes of patients with cirrhosis using a multi-class approach. The competition evaluates submissions based on the multi-class logarithmic loss, which measures the accuracy of predicted probabilities for each of the three outcomes: `Status_C`, `Status_CL`, and `Status_D`.

### Project Structure
- **Notebook**: The main analysis and modeling work is documented in the Jupyter Notebook, `pss3e26_Cirrhosis.ipynb`.
- **Submission File**: The final predictions for the competition are saved in the submission file `submission.csv`.
- **Data**: The competition data can be accessed and downloaded from the Kaggle competition page [here](https://www.kaggle.com/competitions/playground-series-s3e26/data).

### Approach
1. **Data Exploration**: I began by exploring the dataset, understanding the features, and identifying any missing values or data issues.
2. **Feature Engineering**: Created new features based on domain knowledge and exploratory data analysis to improve model performance.
3. **Model Selection**: Experimented with various machine learning models, such as Random Forest, XGBoost etc.
4. **Model Training and Evaluation**: Trained selected models on the training dataset and used multi-class logarithmic loss to evaluate model performance.
5. **Hyperparameter Tuning**: Applied tuning techniques to optimize the model's performance and reduce overfitting.
6. **Final Prediction**: The best-performing model was used to generate predictions for the test dataset, which were saved in `submission.csv`.

### Evaluation Metrics
The competition uses the **multi-class logarithmic loss** to evaluate submissions. The formula is as follows:
\[ \text{logloss} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{M} y_{ij} \log(p_{ij}) \]
where \( N \) is the number of rows in the test set, \( M \) is the number of possible outcomes (3), \( y_{ij} \) is 1 if row \( i \) has the true label \( j \), and 0 otherwise. The predicted probabilities are adjusted to avoid extremes in the logarithm function.

### Dependencies
To run the code and reproduce the results, ensure the following libraries are installed:
- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `matplotlib`
- `seaborn`

### Usage
1. **Clone the Repository**: Use `git clone` to clone this repository to your local machine.
2. **Open the Notebook**: Open the Jupyter Notebook `pss3e26_Cirrhosis.ipynb`.
3. **Run the Code**: Execute the cells in sequence to reproduce the analysis and generate predictions.
4. **Submit Your Results**: If you're participating in the competition, submit the `submission.csv` file on Kaggle.

Feel free to experiment with different approaches and improve the solution! If you have any questions or feedback, please don't hesitate to reach out.

### Conclusion
Thanks for checking out this project. Good luck with your Kaggle competitions and happy coding! 🚀
