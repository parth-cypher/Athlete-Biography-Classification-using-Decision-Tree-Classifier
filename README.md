# Athlete Biography Classification Using Decision Tree Classifier

A beginner-friendly machine learning project that reads a short biography of an athlete and predicts their sport:
**Football, Cricket, Basketball, Tennis, Athletics** or **Other sports**. The text is turned into numbers with
**TF-IDF**, a **Decision Tree** learns the patterns, and a **Streamlit** dashboard lets you classify new biographies
and inspect the model's results.

## Project structure

```
athlete-biography-classification/
├── app.py            Streamlit dashboard (Home, Athlete Classification, Model Evaluation)
├── train.py          Loads the data, trains and evaluates the model, saves results
├── requirements.txt
├── README.md
├── .gitignore
├── data/             Dataset CSV files go here (not modified by the code)
├── models/           athlete_classifier.joblib (created by train.py)
└── reports/          model_metrics.txt, model_metrics.json, confusion_matrix.png (created by train.py)
```

## Dataset

| | |
|---|---|
| **Name** | DBPedia Classes |
| **Original source** | Kaggle, uploaded by Dan Ofer (danofer): <https://www.kaggle.com/datasets/danofer/dbpedia-classes> |
| **Public copy (no login)** | Hugging Face mirror `DeveloperOats/DBPedia_Classes`, licence CC0-1.0: <https://huggingface.co/datasets/DeveloperOats/DBPedia_Classes> |
| **Content** | About 338,000 Wikipedia article abstracts, each labelled with three levels of DBpedia classes (9, 70 and 219 classes) |
| **Files used** | `DBPEDIA_train.csv` (~241,000 rows, ~160 MB), optionally `DBPEDIA_val.csv` (~36,000) and `DBPEDIA_test.csv` (~61,000) |

**Columns**

| Column | Meaning | Example |
|---|---|---|
| `text` | The opening paragraph of the person's Wikipedia article (their biography) | "Nicolas Coutelot (born 9 February 1977) is a retired professional male tennis player from France…" |
| `l1` | Top-level class | `Agent` |
| `l2` | Second-level class | `Athlete` |
| `l3` | Detailed class – for athletes this is the sport | `TennisPlayer` |

The dataset has no separate name column; the athlete's name is the first words of `text`.

**How it is used here.** Only rows with `l2 == "Athlete"` are kept (about 13 % of the data). Their `l3` class is
mapped to the project's categories:

| DBpedia class (`l3`) | Category |
|---|---|
| `SoccerPlayer` | Football |
| `Cricketer` | Cricket |
| `BasketballPlayer` | Basketball |
| `TennisPlayer` | Tennis |
| `AthleticsPlayer` (track and field) | Athletics – only if this class exists in the files |
| every other athlete class (golf, rugby, swimming, …) | Other sports |

`train.py` prints the classes it actually finds and saves the full mapping in `reports/model_metrics.json`, so you
can see exactly which sports ended up in "Other sports". The mapping is the `SPORT_MAP` dictionary at the top of
`train.py` and is easy to change.

### Downloading the dataset

`python train.py` first looks for `DBPEDIA_train.csv`, `DBPEDIA_val.csv` or `DBPEDIA_test.csv` in `data/`. If none is
there, it downloads `DBPEDIA_train.csv` (~160 MB) from the public Hugging Face mirror – no account or password needed.

To download by hand instead:
- **Kaggle** (free account needed): open the Kaggle link above, click **Download**, unzip, and copy
  `DBPEDIA_train.csv` (optionally also `DBPEDIA_val.csv` and `DBPEDIA_test.csv`) into `data/`.
- **Hugging Face** (no account): open <https://huggingface.co/datasets/DeveloperOats/DBPedia_Classes/tree/main>,
  download `DBPEDIA_train.csv` and put it in `data/`.

If the automatic download fails with `CERTIFICATE_VERIFY_FAILED` (common with the python.org installer on macOS),
run **Install Certificates.command** from *Applications → Python 3.x*, or download the file by hand.

The CSV files are never changed; all cleaning happens in memory.

## Machine learning workflow

1. **Load** the CSV file(s) from `data/` and check that the `text`, `l2` and `l3` columns exist.
2. **Inspect** the columns and count missing values (printed and saved).
3. **Select** athlete rows (`l2 == "Athlete"`) and map `l3` to a sport category.
4. **Clean** the text: remove links, `[1]`-style references, punctuation and digits, and extra spaces.
5. **Remove** rows with missing text or label, biographies shorter than 5 words, and duplicate texts.
6. **Split** 80 % training / 20 % testing, *stratified* by sport so both parts have the same mix, with
   `random_state = 42`.
7. **Pipeline**: `TfidfVectorizer` → `DecisionTreeClassifier`, fitted on the training part only (no data leakage).
8. **Evaluate** on the held-out test part.
9. **Save** the whole pipeline to `models/athlete_classifier.joblib`, metrics to `reports/model_metrics.txt` (and
   `.json` for the dashboard), and the confusion matrix to `reports/confusion_matrix.png`.

### What TF-IDF does
A computer cannot read words, so each biography becomes a long list of numbers – one per word in the vocabulary.
**TF-IDF** (term frequency × inverse document frequency) gives a word a high score when it appears often in *this*
biography but rarely across all biographies. Words like "cricket", "wicket" or "dunk" get high scores; common words
like "the" or "born" get low scores (English stop words are removed entirely). Settings used: lower-case, words of 2+
letters only, words must appear in at least 3 biographies (this removes most names), at most 5,000 words.

### What the Decision Tree does
A Decision Tree learns a flowchart of yes/no questions such as *"Is the TF-IDF score of 'cricket' above 0.05?"*.
Each question splits the biographies into purer groups until each group is mostly one sport. To classify a new
biography it simply follows the questions down to a leaf. Settings: Gini impurity, at least 2 biographies per
leaf, and `class_weight="balanced"` so the smaller sports count as much as the large "Other sports" group.

## Evaluation metrics

All numbers come from real predictions on the held-out test set (nothing is hard-coded).

| Metric | Meaning |
|---|---|
| Accuracy | Share of test biographies classified correctly |
| Precision | For each sport: of the biographies predicted as that sport, how many really were |
| Recall | For each sport: of the biographies of that sport, how many were found |
| F1-score | Balance of precision and recall |
| Macro average | Average over sports, every sport counts equally (main score, because the classes are imbalanced) |
| Weighted average | Average weighted by the number of biographies per sport |
| Confusion matrix | Which sports get mixed up with which |

## Installation (macOS, VS Code terminal)

```bash
cd athlete-biography-classification
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Training

```bash
python train.py
```

This downloads the dataset if needed, trains the model, prints the results and saves the model and reports.
Useful options:

```bash
python train.py --sample 5000      # quick test on a stratified sample of 5,000 biographies
python train.py --no-download      # never download; use only files in data/
python train.py --help             # all options
```

To train on a different dataset, use `--data`, `--text-column`, `--label-column`, `--filter-column` (use `''` to keep
all rows), `--filter-value` and `--no-mapping` (use the label column as-is).

## Dashboard

```bash
streamlit run app.py
```

It opens at <http://localhost:8501> (press Enter if Streamlit asks for an email). Sections:
- **Home** – title, objective, dataset source, number of records and categories, technologies.
- **Athlete Classification** – paste a biography and click **Classify Athlete** to see the predicted sport and the
  class probabilities. Empty or very short text and a missing model are handled with clear messages.
- **Model Evaluation** – accuracy, precision, recall, F1-score, confusion matrix and classification report, loaded
  from the files written by `train.py`.

Run `python train.py` before using the classification and evaluation sections. Use these terminal commands rather
than VS Code's ▶ Run button.

## Dataset limitations

- **Class imbalance:** "Other sports" groups many sports together and is much larger than the named categories;
  `class_weight="balanced"` and macro-averaged metrics are used because of this.
- **Athletics may be missing or small:** the dataset only contains the DBpedia classes chosen by its author. If
  `AthleticsPlayer` is not in the files, the model has no Athletics category; `train.py` prints a note when this
  happens.
- **"Football" means association football (soccer).** American, Australian rules and Gaelic football and rugby are
  separate DBpedia classes and fall under "Other sports".
- **Biographies often name the sport directly** ("…is an Indian cricketer…"), so the task is easier than it sounds;
  biographies that never mention the sport are much harder.
- **Wikipedia snapshot:** the text reflects the articles when DBpedia extracted them, so recent careers may be missing,
  and DBpedia labels can occasionally be wrong.
- **Single Decision Tree:** easy to explain but sensitive to small data changes; its probabilities come from one leaf
  and are often 100 % for one class.

## Future improvements

- Compare with Random Forest, Logistic Regression or Linear SVM, which usually do better on text.
- Tune `max_depth` / `min_samples_leaf` with cross-validation.
- Add more sports as separate categories (golf, rugby, swimming, …).
- Use word n-grams or sentence embeddings for richer text features.
- Show which words led to a prediction (the path through the tree).

---
Academic project for a B.Tech CSE Machine Learning course. Wikipedia text is © Wikipedia contributors; the
dataset is provided by its original authors on Kaggle.
