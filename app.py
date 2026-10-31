"""
Streamlit dashboard: Athlete Biography Classification (Decision Tree).

Run from the project folder after training:
    streamlit run app.py
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from train import (
    CONFUSION_PNG, DATASET_MIRROR, DATASET_NAME, DATASET_SOURCE, METRICS_JSON, METRICS_TXT, MODEL_PATH,
    load_model, predict_sport,
)

st.set_page_config(page_title="Athlete Biography Classification", page_icon="🏅", layout="wide")


@st.cache_resource
def get_model(modified: float):
    return load_model(MODEL_PATH)


@st.cache_data
def get_results(modified: float) -> dict:
    return json.loads(METRICS_JSON.read_text(encoding="utf-8"))


def model_and_results():
    """Return (model, results) or (None, None) if training has not been done."""
    if not MODEL_PATH.is_file() or not METRICS_JSON.is_file():
        return None, None
    try:
        return get_model(MODEL_PATH.stat().st_mtime), get_results(METRICS_JSON.stat().st_mtime)
    except Exception as error:  # corrupted or incompatible files
        st.error(f"Could not load the trained model or results ({error}). Please run `python train.py` again.")
        return None, None


def not_trained_message() -> None:
    st.warning("The model has not been trained yet.")
    st.markdown(
        "1. Put the dataset in the `data/` folder (see README.md), or let the script download it.\n"
        "2. In the VS Code terminal run:\n\n"
        "```bash\npython train.py\n```\n"
        "3. Refresh this page."
    )


# ---------------------------------------------------------------------------
# A. Home
# ---------------------------------------------------------------------------
def page_home(results) -> None:
    st.title("🏅 Athlete Biography Classification")
    st.subheader("Using a Decision Tree Classifier")
    st.markdown(
        "**Objective:** read a short biography of an athlete and predict which sport they play – "
        "Football, Cricket, Basketball, Tennis, Athletics or Other sports – using text features (TF-IDF) "
        "and a Decision Tree."
    )
    st.markdown(
        f"**Dataset:** [{DATASET_NAME}]({DATASET_SOURCE}) (Kaggle) – Wikipedia article abstracts labelled with "
        f"DBpedia classes. Only articles labelled as *Athlete* are used. Public copy: [Hugging Face mirror]({DATASET_MIRROR})."
    )
    if results is None:
        not_trained_message()
    else:
        data = results["data"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Records loaded", f"{data['rows_loaded']:,}")
        c2.metric("Athlete biographies", f"{data.get('rows_after_filter', data['rows_loaded']):,}")
        c3.metric("Used for training/testing", f"{data['rows_used']:,}")
        c4.metric("Sport categories", len(data["category_counts"]))
        counts = pd.Series(data["category_counts"], name="Biographies").sort_values(ascending=False)
        st.markdown("**Biographies per category**")
        st.bar_chart(counts)
        st.caption(f"Files used: {', '.join(results['dataset']['files'])} · trained {results['trained_at']}")
    st.markdown(
        "**Technologies:** Python · pandas · scikit-learn (TfidfVectorizer, DecisionTreeClassifier, Pipeline) · "
        "Matplotlib · Streamlit · Joblib"
    )


# ---------------------------------------------------------------------------
# B. Athlete classification
# ---------------------------------------------------------------------------
def page_classify(model) -> None:
    st.title("🔎 Athlete Classification")
    if model is None:
        not_trained_message()
        return
    st.write("Paste or type an athlete's biography (a few sentences, in English) and click **Classify Athlete**.")
    biography = st.text_area("Athlete biography", height=180, key="biography",
                             placeholder="e.g. She is a professional player who has won several titles ...")
    if st.button("Classify Athlete", type="primary"):
        try:
            sport, probabilities, known_words = predict_sport(model, biography)
        except ValueError as error:
            st.warning(str(error))
            return
        st.success(f"Predicted sport: **{sport}**")
        if known_words == 0:
            st.warning("None of the words in this text are in the model's vocabulary, "
                       "so this prediction is essentially a guess.")
        table = pd.DataFrame({"Sport": list(probabilities), "Probability": list(probabilities.values())})
        st.markdown("**Class probabilities**")
        st.bar_chart(table.set_index("Sport"))
        table["Probability"] = table["Probability"].map(lambda p: f"{p:.1%}")
        st.dataframe(table, hide_index=True)
        st.caption("A single Decision Tree gives the share of training biographies of each sport in the leaf "
                   "the text ends up in, so probabilities are often 100% for one sport. "
                   "Predictions are estimates and can be wrong.")


# ---------------------------------------------------------------------------
# C. Model evaluation
# ---------------------------------------------------------------------------
def page_evaluation(results) -> None:
    st.title("📈 Model Evaluation")
    if results is None:
        not_trained_message()
        return
    m = results["metrics"]
    split = results["split"]
    st.write(f"Results on the held-out test set of **{split['test_rows']:,}** biographies "
             f"(trained on {split['train_rows']:,}; stratified split, random_state={split['random_state']}).")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{m['accuracy']:.3f}")
    c2.metric("Precision (macro)", f"{m['precision_macro']:.3f}")
    c3.metric("Recall (macro)", f"{m['recall_macro']:.3f}")
    c4.metric("F1-score (macro)", f"{m['f1_macro']:.3f}")
    st.caption(f"Weighted averages – precision {m['precision_weighted']:.3f}, recall {m['recall_weighted']:.3f}, "
               f"F1 {m['f1_weighted']:.3f}. Macro averages treat every sport equally; weighted averages follow "
               "the number of biographies per sport.")

    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Confusion matrix")
        if CONFUSION_PNG.is_file():
            st.image(str(CONFUSION_PNG))
        else:
            st.info("reports/confusion_matrix.png not found – run python train.py again.")
    with right:
        st.subheader("Classification report")
        report = pd.DataFrame(m["classification_report"]).T
        report = report.drop(index=["accuracy"], errors="ignore")
        report["support"] = report["support"].astype(int)
        st.dataframe(report.round(3))
        st.caption("Precision: of the biographies predicted as a sport, how many were right. "
                   "Recall: of the biographies of a sport, how many were found. F1: balance of both.")

    with st.expander("Full text report (reports/model_metrics.txt)"):
        if METRICS_TXT.is_file():
            st.code(METRICS_TXT.read_text(encoding="utf-8"), language=None)
    model_info = results["model"]
    st.caption(f"Tree depth {model_info['tree_depth']}, {model_info['tree_leaves']:,} leaves, "
               f"vocabulary of {model_info['vocabulary_size']:,} words.")


def main() -> None:
    model, results = model_and_results()
    st.sidebar.title("🏅 Athlete Classifier")
    page = st.sidebar.radio("Section", ["Home", "Athlete Classification", "Model Evaluation"], key="page")
    st.sidebar.caption("Decision Tree + TF-IDF on real Wikipedia athlete biographies")
    if page == "Home":
        page_home(results)
    elif page == "Athlete Classification":
        page_classify(model)
    else:
        page_evaluation(results)


main()
