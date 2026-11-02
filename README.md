# NLP Labs – Solutions

Solutions for the 11 Natural Language Processing labs. Each folder holds one lab notebook with the
original lab content and the completed tasks, already executed so all outputs are visible.

| Lab | Topic | Tasks solved |
|---|---|---|
| [01](Lab01_Intro_to_NLP) | Introduction to NLP | Choose a dataset type (CSV), download & explore the IMDB 50K reviews dataset |
| [02](Lab02_Text_Preprocessing_Regex) | Text pre-processing & regular expressions | Hashtag counting / top-10 on Apple tweets, `re.compile()`, `re.split()`, spaCy vs NLTK tokenization |
| [03](Lab03_N_Grams) | N-grams | Clean Pakistan tweets, bigram MLE language model, tweet generation, perplexity, P(is \| pakistan) |
| [04](Lab04_Classification_Evaluation) | Classification & evaluation | Amazon phone reviews: 5 pre-processing steps, split, TF-IDF, Naive Bayes, report, confusion matrix |
| [05](Lab05_Text_Representation) | Text representation | Cosine similarity, TF-IDF, Word2Vec skip-gram on the Simpsons scripts, `most_similar`, `doesnt_match` |
| [06](Lab06_Deep_Learning_for_NLP) | Deep learning for NLP (PyTorch) | Word2Vec review vectors + 3-hidden-layer network on Yelp reviews |
| [07](Lab07_LLMs) | Large Language Models | Model families, DistilBERT sentiment, FLAN-T5 summarization, reflection, sampling challenge |
| [08](Lab08_POS_Tagging) | Part-of-speech tagging | spaCy POS / fine-grained tags, *google* & *table* in context, displaCy visualization |
| [09](Lab09_NER) | Named Entity Recognition | NER on user input, adding missed entities + counts + visualization, training a custom NER model |
| [10](Lab10_Machine_Translation) | Machine translation (Keras LSTM) | Layer questions, English → French encoder–decoder LSTM with teacher forcing on the Kaggle dataset (BLEU ≈ 0.16) |
| [11](Lab11_Attention_RNN) | Attention with RNN | Attention weights for 3 questions, attention comparison for two similar questions |

## Running the notebooks

```bash
pip install nltk spacy gensim scikit-learn pandas matplotlib emoji kagglehub torch tensorflow transformers sentencepiece
python -m spacy download en_core_web_sm
```

Kaggle datasets are downloaded automatically with `kagglehub`. The small datasets provided with the
labs are in each lab's `dataset/` folder.

## Changes made to the provided code

A few cells were adjusted so the notebooks run outside Google Colab and train correctly:

- `/content/...` Colab paths now point to `kagglehub` downloads or the local `dataset/` folder.
- Lab 2: `import nltk` added before `nltk.download(...)`.
- Lab 6: the pre-processing `apply()` was indented after `return` inside the function, so it never ran; it was moved out.
  `optimizer.zero_grad()` was added to the training loop.
- Lab 11: `nn.RNN(..., batch_first=True)`, because the input is `(batch, words)`. Without it, the RNN read across
  questions instead of across words. `optimizer.zero_grad()` was also added to the training loop.
