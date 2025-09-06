# Predicting Stress and Sleep Disorders Using Health and Lifestyle Data
- This project explores **machine learning for mental health**, 
- specifically predicting **human stress levels** from physiological and behavioral signals.  
- It was developed as part of **Introduction to Machine Learning** at **IIT Palakkad**.

---

## 🚀 Project Overview  
Stress is a critical factor affecting mental and physical well-being. Our project leverages **supervised learning techniques** to classify stress levels (e.g., low, medium, high) based on features extracted from sensor and questionnaire data.  

Key Highlights:  
- Preprocessing of real-world physiological datasets (EDA, heart rate, etc.)  
- Feature engineering for stress-related markers  
- Training and evaluation of multiple ML models  
- Comparative analysis of accuracy, precision, recall, and F1-score  
- Documentation of pipeline for reproducibility  

---

## 📊 Results  
- Regression
    - Achieved **85% accuracy**  with XGBOOST (regressor).  
- Classification
    - Achieved **80% accuracy** with SVM SVC (classifier).
- Notable improvement over baseline methods.  
- Insights into feature importance (e.g., heart rate variability as strongest predictor).  

*(Optional: add a confusion matrix / accuracy table screenshot in `/results`)*  

---

## Tech Stack  
- **Languages**: Python (NumPy, Pandas, Matplotlib)  
- **ML Libraries**: scikit-learn, XGBoost  
- **Notebooks & Visualization**: Jupyter, Seaborn  
- **Collaboration & Docs**: GitHub, Google Docs  

---

## Repository Structure  
.
├── data            # Data used for training (thanks to kaggle)
│   ├── Sleep_health_and_lifestyle_dataset.csv
│   └── Sleep_health_and_lifestyle_dataset_part_2.csv
├── docs            # Documentation for the project
│   └── ML Project.pdf
├── notebooks       # Jupyter notebooks for visualization
│   ├── endsem.ipynb
│   ├── midsem.ipynb
│   └── test.ipynb
├── README.md       # Readme
├── results         # Results of the training
└── src             # Main part of the code

---

## Getting Started  

Clone the repository and install dependencies:  
```bash
git clone https://github.com/VISHNU-SHREERAM/ML-Project
cd ML-Project
pip install -r requirements.txt
```

## Run training
``` bash
python3 src/train.py
```

## View Results
```
jupyter notebook notebooks/evaluation.ipynb
```

## Documentation
- Detailed methodology and results can be found in our project [report](docs/ML Project.pdf).

## Team
- [Vishnu Shreeram M P](https://github.com/VISHNU-SHREERAM) 
- [Bhupathi Varun](https://github.com/cvbshcbad)
- [Bhogaraju Shanmukha Sri Krishna](https://github.com/wanderer3519)
