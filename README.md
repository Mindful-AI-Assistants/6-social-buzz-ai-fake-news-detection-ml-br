<br>
 
 \[[🇧🇷 Português](README.pt_BR.md)\] \[**[🇺🇸 English](README.md)**\]

<br>


# <p align="center"> 6- Social Buzz AI - Fake News Detection Using Machine Learning

<br><br>

<p align="center">
   <img src="https://github.com/user-attachments/assets/791a69e2-d09a-429f-9257-f6667fff5c04 ">
 </p>

<br><br>

[**Course:**]() Humanistic AI & Data Science (4th Semester)  
[**Institution:**]() PUC-SP  
**Professor:** [Erick Bacconi](https://www.linkedin.com/in/eric-bacconi-423137/)  

<br><br>


#### <p align="center"> [![Sponsor Mindful AI Assistants](https://img.shields.io/badge/Sponsor-%C2%B7%C2%B7%C2%B7%20Mindful%20AI%20Assistants%20%C2%B7%C2%B7%C2%B7-brightgreen?logo=GitHub)](https://github.com/sponsors/Mindful-AI-Assistants)


<br><br>

> [!TIP]
>
> This repository 2-social-buzz-ai-GBoost-and-LowDefault-Modeling is part of the main project 1-social-buzz-ai-main.
> To explore all related materials, analyses, and notebooks, visit the main repository 
>
> * [1-social-buzz-ai-main](https://github.com/Mindful-AI-Assistants/1-social-buzz-ai-main)
> *Part of the Humanistic AI Research & Data Modeling Series — where data meets human insight.*
>
>  *  [Access](https://github.com/Mindful-AI-Assistants/4-social-buzz-ai--Natural_Language_Processing-NL-Class_1):  NLP  - Class 1 Repo
>
> 
>  *  [Access]() Code
> 
>  *  [Access]()  True Dataset
> 
> *  [Access](https://github.com/Mindful-AI-Assistants/6-social-buzz-ai-fake-news-detection-ml-br/tree/0f68f4132e72c0a43ca0f71e15a39525bd07bad4/dataset_Fake)  Fake Dataset
> 
>  <br>
> 





<br><br>

## 1. [Introduction]()

<br>

- **Fake news** are false information mainly spread on social networks, which can cause serious political, social, and public health impacts.
- This study aims to apply *Machine Learning* (ML) algorithms to automatically detect fake news, offering a technological alternative to address this issue.

<br><br>

## 2. [Study Objectives]()

<br>

- Test and compare different ML algorithms for detecting fake news.
- Assess the performance of each model in accuracy, sensitivity, and specificity.
- Propose an automated, replicable, and useful solution for society.

<br><br>

## 3. [Detailed Methodology]()

<br>

### 3.1. [Dataset]()

- [Data obtained from Kaggle:]()
    - Fake: 23,481 samples
    - True: 21,417 samples
- [Main columns:]() title, text, topic, date

<br>

### 3.2. [Tools Used]()

Python + libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, NLTK

<br>

### 3.3. [Data Processing]()

<br>

```python
import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords

# Load fake and true datasets
fake = pd.read_csv('Fake.csv')
true = pd.read_csv('True.csv')
fake['target'] = 1
true['target'] = 0

# Concatenate and shuffle records
data = pd.concat([fake, true], ignore_index=True)
data = data.sample(frac=1).reset_index(drop=True)

# Remove title and date columns
data.drop(['title', 'date'], axis=1, inplace=True)

# Clean text (lowercase, no punctuation, no stopwords)
nltk.download('stopwords')
stop_words = set(stopwords.words('portuguese'))

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

data['text'] = data['text'].apply(clean_text)
```

<br>

### 3.4. [Visualization: WordCloud]()

<br>

```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt

wc = WordCloud(width=800, height=400, background_color='white').generate(' '.join(data['text']))
plt.figure(figsize=(10, 5))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()
```

<br>

### 3.5. [Vectorization and Split]()

<br>

```python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

X = data['text']
y = data['target']

vectorizer = TfidfVectorizer(max_features=5000)
X_vect = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)
```

<br><br>

## 4. [Applied Algorithms]()

<br>

Five supervised models were trained and evaluated:

<br>

| [Model]() | [Accuracy]() | [Remarks]() |
| :-- | :-- | :-- |
| [Logistic Regression]() | 98.92% | High precision, confusion matrix shows low error rate. |
| [Decision Tree]() | 99.6% | Best overall performance and lowest error. |
| [Random Forest]() | 98.74% | Good performance, consistent confusion matrix. |
| [Support Vector Machine]() | 99.5% | Excellent accuracy and precision, robust text model. |
| [K-Nearest Neighbors (KNN)]() | 60.84% | Low performance, high number of false negatives. |

<br>

**Metrics computed via confusion matrix** (including TP, TN, FP, FN) and specific precision and sensitivity values for each model.

#### Example: [Simplified Confusion Matrix (Decision Tree)]()

- [True Positives (TP)](): 4,711
- [False Positives (FP)](): 13
- [False Negatives (FN)](): 22
- [True Negatives (TN)](): 4,234

<br><br>

## 4.1. [Model Training and Evaluation]()

<br>

Below are examples of models tested and evaluated.

<br>

### [Logistic Regression]()

<br>

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

<br>

### [Decision Tree]()

<br>

```python
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

<br>

### [Random Forest]()

<br>

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

<br>

### [SVM]()

<br>

```python
from sklearn.svm import SVC

svm = SVC()
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

<br>

### [KNN]()

<br>

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

<br>

### [Confusion Matrix Visualization]()

<br>

```python
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

<br><br>

## 5. [Evaluation Metrics]()

<br>

- [**Accuracy:**]() Overall model success rate.
- [**Precision:**]() How well fake news are detected.
- [**Sensitivity:**]() Model's ability to correctly identify actual fake news.
- [**Specificity:**]() Model's ability to correctly identify actual real news.

<br>

#### [Model Performance]()

<br>

| [Model]() | [Precision]() | [Sensitivity]() | [Specificity]() |
| :-- | :-- | :-- | :-- |
| [Logistic Regression]() | 98% | 99% | 98% |
| [Decision Tree]() | 98.5% | 99% | 99% |
| [Random Forest]() | 98.5% | 99% | 98% |
| [SVM]() | 99% | 99% | 99% |
| [KNN]() | 99% | 57% | 19% |

<br><br>

## 6. [Detailed Results]()

<br>

- Four out of five models achieved accuracy above 90%.
- KNN performed poorly, mainly due to the high number of **false negatives** (57% sensitivity).
- Decision Tree and SVM excelled as the most efficient.
- Data processing and feature selection were key to the models' success.

<br><br>

## 7. [Limitations and Future Directions]()

<br>

- [**Study limitations:**]() Difficulty in finding standardized datasets (especially in Portuguese), few systems applied to the Brazilian context.
- [**Future directions:**]() Test new algorithms (Naive Bayes, Boosting, K-means, Gradient Descent), apply other validation techniques, expand Portuguese datasets, include cross-validation (K-fold, Leave-one-out), and develop web applications for public use.

<br><br>

## 8. [Conclusion]()

<br>

- Machine Learning has proven powerful for fake news detection and is crucial for protecting society from the impact of false information.
- Ongoing research is especially necessary in the Brazilian context.

<br><br>

## 9. [Our Crew:]()

<br>

- 👨🏽‍🚀 [**Andson Ribeiro**](https://github.com/andsonandreribeiro09)
- 👩🏻‍🚀 **Fabiana ⚡️ Campanari** - [Shoot me an email](mailto:fabicampanari@proton.me)
- 👨🏽‍🚀 [**Pedro Barrenco**](https://github.com/Pgbarenco)
- 🧑🏼‍🚀 [**Pedro Vyctor**](https://github.com/Pgbarenco)

<br><br>

## 10. [References]()

<br>

Monteiro Bastos & Monteiro de Lima (2023). Fake News Detection Using Decision Tree, Support Vector Machine, and K-Nearest Neighbors Algorithms. Revista de Estudos Multidisciplinares XV Encontro Científico da UNDB.

<br><br>

## 💌 [Let the data flow... Ping Me !](mailto:fabicampanari@proton.me)

#### [Contact and Support]()

- For notebook files, detailed tutorials, or enhanced visualizations, please reach out.
- Interested in Python notebooks simulating these dynamics or advanced Humanistic AI models? Just ask!

<br>

#### <p align="center">  🛸๋ My Contacts [Hub](https://linktr.ee/fabianacampanari)

<br>

### <p align="center"> <img src="https://github.com/user-attachments/assets/517fc573-7607-4c5d-82a7-38383cc0537d" />

<br>

<p align="center">  ────────────── 🔭⋆ ──────────────

<p align="center"> ➣➢➤ <a href="#top">Back to Top </a>

<b><br>

#

##### <p align="center">Copyright 2025 Mindful-AI-Assistants. Code released under the [MIT license.](https://github.com/Mindful-AI-Assistants/lacan-psychoanalysis-math-graphs/blob/28d9178584b831679dec129fb0aa040203ce0e9e/LICENSE.md)



