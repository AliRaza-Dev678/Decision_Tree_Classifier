# 🌿 Decision Tree Classifier — Likeness Dataset

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat-square&logo=jupyter)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-red?style=flat-square&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

> A supervised machine learning project implementing a **Decision Tree Classifier** on the **Likeness Dataset** — exploring tree-based decision logic, pruning strategies, feature importance, and thorough model evaluation with visual tree rendering.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [What is a Decision Tree?](#-what-is-a-decision-tree)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Key Concepts Covered](#-key-concepts-covered)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [Project Structure](#-project-structure)
- [Model Performance](#-model-performance)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## 🔍 Overview

This project applies a **Decision Tree Classifier** to predict outcomes on the Likeness Dataset — a classification task centered on understanding what features drive a user's preference or affinity. Decision Trees are one of the most interpretable machine learning algorithms, making them ideal for understanding *why* a model makes a specific prediction. The full pipeline — from EDA to tree visualization — is implemented in a Jupyter Notebook.

---

## 🧠 What is a Decision Tree?

A **Decision Tree** is a supervised learning algorithm that makes predictions by learning a series of **if-else decision rules** inferred from the data features. It mirrors human-level thinking — splitting data step by step until it reaches a final prediction at a **leaf node**.

```
Root Node → Branch (Feature Split) → Internal Nodes → Leaf (Prediction)
```

| Component | Description |
|---|---|
| **Root Node** | The topmost node; the first and most important feature split |
| **Internal Node** | Represents a test on a feature (e.g., Age > 30?) |
| **Branch** | Outcome of a split condition (True / False) |
| **Leaf Node** | Terminal node; holds the final class prediction |
| **Depth** | Number of levels from root to the deepest leaf |

**Splitting Criteria used by Decision Trees:**

| Criterion | Description |
|---|---|
| **Gini Impurity** | Measures how often a random element would be misclassified |
| **Entropy / Information Gain** | Measures the reduction in uncertainty after a split |

**Key strengths of Decision Trees:**
- Highly **interpretable** — the logic can be read like a flowchart
- Requires **no feature scaling** (unlike SVM or KNN)
- Handles both **numerical and categorical** features
- Naturally computes **feature importance**
- Easy to **visualize** with `plot_tree` or `export_graphviz`

---

## 📊 Dataset

The **Likeness Dataset** is a classification dataset used to predict user preference or affinity based on a set of input features.

| Property | Details |
|---|---|
| **Task** | Binary / Multi-class Classification |
| **Target** | Whether a user likes / prefers something |
| **Type** | Structured tabular data |
| **File** | Included in the repository or loaded directly in the notebook |

> 📖 Refer to the notebook for full feature descriptions, class distribution, and exploratory analysis outputs.

---

## ⚙️ Methodology

### 1. Data Loading & Exploration
- Load the Likeness Dataset using Pandas
- Inspect shape, column types, and class distribution
- Check for missing values and duplicate records

### 2. Exploratory Data Analysis (EDA)
- **Count plots** to visualize class balance
- **Histograms and box plots** to examine feature distributions
- **Correlation heatmap** to identify feature relationships
- **Pairplots** to visualize inter-feature patterns

### 3. Data Preprocessing
- Handle missing values (imputation or removal)
- Encode categorical variables using **Label Encoding** or **One-Hot Encoding**
- Split data into **training and test sets** (typically 80/20)
- Apply **feature scaling** where applicable

### 4. Model Training
- Train a **Decision Tree Classifier** using scikit-learn
- Configure `criterion` (Gini or Entropy), `max_depth`, and other parameters
- The tree learns decision rules by finding the splits that maximize information gain

### 5. Tree Visualization
- Render the full decision tree using `plot_tree` or `export_graphviz`
- Annotate with feature names, class labels, and impurity values
- Inspect which features appear at the top levels (most important splits)

### 6. Hyperparameter Tuning
- Tune key parameters to control tree complexity and prevent overfitting:
  - `max_depth` — limits tree depth to avoid overfitting
  - `min_samples_split` — minimum samples needed to split a node
  - `min_samples_leaf` — minimum samples required at a leaf node
  - `criterion` — splitting metric: `gini` or `entropy`

### 7. Model Evaluation
- Evaluate with **Accuracy Score**, **Classification Report**, and **Confusion Matrix**
- Visualize confusion matrix as a labeled heatmap
- Extract and plot **feature importance scores**

---

## 📚 Key Concepts Covered

| Concept | Description |
|---|---|
| **Recursive Partitioning** | How the tree repeatedly splits data based on feature thresholds |
| **Gini Impurity** | Node purity measure used to choose the best split |
| **Information Gain (Entropy)** | Reduction in uncertainty after applying a split |
| **Overfitting** | When a tree memorizes training data and fails to generalize |
| **Pruning** | Simplifying the tree by limiting depth or leaf size to reduce overfitting |
| **Feature Importance** | Reflects how much each feature contributes to reducing impurity |
| **Decision Boundary** | The regions in feature space assigned to each class |
| **White-box Model** | Decision trees are fully interpretable — every decision is traceable |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **Python 3.8+** | Core programming language |
| **Jupyter Notebook** | Interactive development environment |
| **Pandas** | Data loading and manipulation |
| **NumPy** | Numerical computations |
| **Matplotlib** | Tree visualization and plots |
| **Seaborn** | Statistical heatmaps and distribution plots |
| **scikit-learn** | Decision Tree model, preprocessing, and evaluation |

---

## 🚀 Getting Started

### Prerequisites

Make sure you have Python 3.8+ and pip installed.

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/AliRaza-Dev678/Decision_Tree_Classifier.git
cd Decision_Tree_Classifier

# 2. (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install required packages
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Run the Notebook

```bash
jupyter notebook Decision_Tree_Classifier.ipynb
```

Open the notebook in your browser and run all cells from top to bottom.

---

## 📁 Project Structure

```
Decision_Tree_Classifier/
│
├── Decision_Tree_Classifier.ipynb    # Main notebook — full pipeline
└── README.md                         # Project documentation
```

---

## 📈 Model Performance

Decision Tree performance depends heavily on the dataset characteristics and the depth allowed. With appropriate pruning, the model generalizes well without overfitting.

| Metric | Description |
|---|---|
| **Accuracy** | Overall proportion of correctly classified samples |
| **Precision** | How many predicted positives are actually positive |
| **Recall** | How many actual positives were correctly captured |
| **F1-Score** | Harmonic mean of Precision and Recall |
| **Confusion Matrix** | Breakdown of true vs predicted class counts |

> Run the notebook to see exact metric outputs, the rendered decision tree, and feature importance charts.

---

## 🔭 Future Improvements

- Apply **Cost Complexity Pruning (CCP)** (`ccp_alpha`) for more principled tree simplification
- Use **GridSearchCV** to systematically tune `max_depth`, `min_samples_split`, and `criterion`
- Apply **k-fold cross-validation** for robust performance estimates
- Visualize **decision boundaries** on 2D feature subsets
- Compare Decision Tree against **Random Forest**, **SVM**, **KNN**, and **Naive Bayes**
- Export the trained tree as a **PDF or PNG** using `export_graphviz` + Graphviz
- Package the trained model with **joblib** for reuse and deployment

---

## 👤 Author

**Ali Raza**

[![GitHub](https://img.shields.io/badge/GitHub-AliRaza--Dev678-black?style=flat-square&logo=github)](https://github.com/AliRaza-Dev678)

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute it.

---

> ⭐ If you found this project useful, consider giving it a star on GitHub!
