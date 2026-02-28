https://archive.ics.uci.edu/dataset/228/sms+spam+collection

#### 📩 SMS Spam Detection — Threshold Optimized ML Pipeline

#### 📌 Problem

Binary text classification task: **SMS Spam Detection**

Dataset size: **5,572 messages**

| Class    | Samples |
| -------- | ------- |
| Ham (0)  | 4,825   |
| Spam (1) | 747     |

Imbalance ratio ≈ **86.6% / 13.4%**

Given the imbalance, **F1-score** was used as the primary optimization metric.

---

#### 🧠 Modeling Strategy

Instead of using default classification thresholds (0.5), this project:

- Performs **manual Stratified 5-fold CV**
- Computes the **Precision–Recall curve**
- Selects the **threshold that maximizes F1 per fold**
- Averages optimal thresholds across folds
- Stores the optimal threshold inside each Optuna trial

This makes the evaluation strictly aligned with the business objective:

> Maximize detection performance on the minority class (Spam).

---

#### 🔍 Pipeline Architecture

For each model:
```Python
TfidfVectorizer
    ↓
Classifier
    ↓
Manual StratifiedKFold CV
    ↓
Precision–Recall Curve
    ↓
Optimal F1 Threshold Selection
```
---

#### ⚙️ Feature Engineering

**Vectorization:**
- <span style="background-color: lightgrey;">TfidfVectorizer</span>
- Tuned:
    - <span style="background-color: lightgrey;">ngram_range</span> (1,1) or (1,2)
    - <span style="background-color: lightgrey;">max_features</span> (5k – 30k)
    - <span style="background-color: lightgrey;">min_df</span> (1 – 5)

Sparse high-dimensional representation optimized for linear models.

---

#### 🤖 Models Evaluated
#### 1️⃣ Logistic Regression
   
- <span style="background-color: lightgrey;">solver="liblinear"</span>
- <span style="background-color: lightgrey;">class_weight="balanced"</span>
- Tuned regularization C
- Threshold optimized via CV

#### 2️⃣ Linear SVM (Best Model 🥇)

- <span style="background-color: lightgrey;">LinearSVC</span>
- <span style="background-color: lightgrey;">class_weight="balanced"</span>
- Hyperparameter <span style="background-color: lightgrey;">C</span> optimized
- Calibrated via <span style="background-color: lightgrey;">CalibratedClassifierCV</span>
- Decision function used for threshold tuning

#### 3️⃣ LightGBM

- Tuned:
    - <span style="background-color: lightgrey;">n_estimators</span>
    - <span style="background-color: lightgrey;">num_leaves</span>
    - <span style="background-color: lightgrey;">learning_rate</span>
- <span style="background-color: lightgrey;">class_weight="balanced"</span>

---

#### 🎯 Hyperparameter Optimization

Optimization performed with **Optuna (TPE Sampler):**

- 40 trials per model
- Objective: maximize mean CV F1-score
- Best threshold stored via <span style="background-color: lightgrey;">trial.set_user_attr</span>

Example:
```Python
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
)
study.optimize(objective, n_trials=40)
```
---
#### 🏆 Results
| Model               | F1 Score   | Optimal Threshold |
| ------------------- | ---------- | ----------------- |
| Logistic Regression | 0.9486     | 0.6520            |
| 🥇 Linear SVM       | **0.9542** | 0.4113            |
| LightGBM            | 0.9188     | 0.2492            |
---
#### 📈 Final Test Evaluation (Best Model: LinearSVC)

- Model retrained on full training set
- Decision scores computed on test set
- Predictions generated using optimized threshold
``` Python
scores = pipeline_svm.decision_function(X_test)
y_pred = (scores >= best_threshold_svm).astype(int)
```
Classification report printed on hold-out test set.

---

#### 🧪 Why Linear Models Won

- TF-IDF produces high-dimensional sparse features
- Linear decision boundaries perform strongly in this space
- Lower variance than boosting in small/medium text datasets
- Balanced class weights mitigate imbalance
- Threshold optimization improved minority recall without degrading precision
---

#### 📊 Key Technical Decisions

✔ Manual CV instead of cross_val_score to control threshold selection<br>
✔ PR-curve-based threshold optimization<br>
✔ Balanced class weights instead of resampling<br>
✔ Calibration for SVM to stabilize decision scores<br>
✔ Explicit separation of model selection and final training

---

#### 🚀 Reproducibility

- Fixed random seed (seed=42</span>)
- Stratified splits
- Deterministic Optuna sampler

---

#### 🏁 Conclusion

This project demonstrates that:

- Threshold optimization can significantly improve imbalanced classification performance.
- Linear SVM + TF-IDF remains a strong baseline for text classification.
- Careful evaluation design matters more than model complexity.

Best achieved performance:

> **F1-score = 0.9542 (LinearSVC)**