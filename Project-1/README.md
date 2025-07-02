# 🧠 Explainable-AI (XAI)

This project demonstrates how to apply model-agnostic explainability techniques — **SHAP** and **LIME** — to understand predictions made by machine learning models.

---

## 📦 Project Contents

- `shap_explainer.py`: Uses SHAP to explain a Random Forest classifier trained on the Breast Cancer dataset.
- `lime_explainer.py`: Uses LIME for local interpretation of the same model.
- `requirements.txt`: Contains all dependencies to run this project.

---

## 🚀 Getting Started

1. Clone the repository:

```bash
git clone https://github.com/benyaminemp/Explainable-AI-XAI.git
cd Explainable-AI
```

2. (Optional) Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🧪 Run the Examples

### SHAP

```bash
python shap_explainer.py
```

### LIME

```bash
python lime_explainer.py
```

Both scripts will train a Random Forest classifier on the Breast Cancer dataset and generate explainability plots.

---

## 📃 References

- [SHAP Documentation](https://shap.readthedocs.io/en/latest/)
- [LIME GitHub](https://github.com/marcotcr/lime)

