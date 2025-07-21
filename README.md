# 🏦 Loan Approval Prediction

This project uses machine learning (Random Forest Classifier) to predict whether a loan application will be approved based on applicant details such as income, credit history, education, and more.

---

## 📌 Features

- Handles missing values
- Encodes categorical variables
- Splits dataset into training and testing sets
- Trains a Random Forest Classifier
- Evaluates model using Accuracy, Confusion Matrix, and Classification Report
- Visualizes feature importance

---

## 📁 Project Structure

Loan-Approval-Prediction/
├── data/
│ └── train.csv ← Place dataset here (not included in repo)
├── loan_approval.py ← Main Python script
├── requirements.txt ← Required Python packages
├── .gitignore
└── README.md

## 🧪 Dataset

Please download or use your own dataset and place it in the `data/` folder.

> **File expected**: `data/train.csv`  
> This file is ignored from the repo for privacy and size reasons.

---

## 📦 Requirements

Install all dependencies using:

```bash
pip install -r requirements.txt
▶️ How to Run
Make sure you're inside the project folder, then run:

python loan_approval.py
This will:

Train the model

Output evaluation metrics

Display a feature importance bar graph

📊 Example Output
lua
Copy
Edit
Accuracy: 0.81

Confusion Matrix:
[[20  4]
 [ 5 95]]

Classification Report:
              precision    recall  f1-score   support
           0       0.80      0.83      0.81        24
           1       0.96      0.95      0.95       100
And a plot of feature importance will be shown using seaborn.

📌 Author
👤 Ayesha Khanum
🔗 GitHub: Ayesha0702
🔗 LinkedIn: Ayesha Khanum

✅ License
This project is licensed for educational and non-commercial use.

### ✅ Instructions

1. Save the above as a file named `README.md` in your project folder.
2. Then run:

```bash
git add README.md
git commit -m "Added project README"
git push
