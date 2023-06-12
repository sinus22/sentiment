from django.http import HttpRequest
from django.shortcuts import render
import pandas as pd

pd.options.mode.chained_assignment = None
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from home.forms import TestForm
import pickle

# Create your views here.

WORD_MODEL = 'word_model.sav'
WORD_TRANSFORM = 'word_transform.sav'

CHAR_MODEL = 'char_model.sav'
CHAR_TRANSFORM = 'char_transform.sav'

DATA_DIR = "data"
POSITIVE_WORDS = 'UZ_positive.txt'
NEGATIVE_WORDS = 'UZ_negative.txt'


def test_models(input_sentence: str):
    lr_word = pickle.load(open(WORD_MODEL, 'rb'))
    vectorizer_word = pickle.load(open(WORD_TRANSFORM, 'rb'))
    input_text_vectorized = vectorizer_word.transform([input_sentence])
    predicted_polarity = lr_word.predict(input_text_vectorized)
    predicted_scores = lr_word.predict_proba(input_text_vectorized)[0]

    lr_char = pickle.load(open(CHAR_MODEL, 'rb'))
    vectorizer_char = pickle.load(open(CHAR_TRANSFORM, 'rb'))
    input_text_vectorized = vectorizer_char.transform([input_sentence])
    predicted_polarity_char = lr_char.predict(input_text_vectorized)
    predicted_scores_char = lr_char.predict_proba(input_text_vectorized)[0]

    polarity_names = ['Negativ sentiment', 'Pozitiv sentiment']
    print("Matndagi sentiment:", polarity_names[predicted_polarity[0]])
    print("Predicted Scores:")
    print("Positive:", predicted_scores[1])
    print("Negative:", predicted_scores[0])
    return {
        'text_word': polarity_names[predicted_polarity[0]],
        'text_char': polarity_names[predicted_polarity_char[0]],
        'positive_word': int(predicted_scores[1] * 1000) / 10.,
        'negative_word': int(predicted_scores[0] * 1000) / 10.,
        'positive_char': int(predicted_scores_char[1] * 1000) / 10.,
        'negative_char': int(predicted_scores_char[0] * 1000) / 10.,
    }


def add_text(text: str, check: bool):
    if check:
        with open(os.path.join(DATA_DIR, POSITIVE_WORDS), 'a') as f:
            f.write(text + '\n')

    else:
        with open(os.path.join(DATA_DIR, NEGATIVE_WORDS), 'a') as f:
            f.write(text + '\n')
    return {
        'message': True
    }


def home_index(req: HttpRequest):
    form = TestForm(req.POST or None)
    result = {}
    if req.method == 'POST':
        print(req.POST)
        data = req.POST
        if data.get('test'):
            result = test_models(form.data.get('text'))
        if data.get('positive'):
            result = add_text(form.data.get('text'), True)
        if data.get('negative'):
            result = add_text(form.data.get('text'), False)

    return render(req, 'home/index.html', {
        'form': form,
        'result': result
    })


def model_reinstall(req: HttpRequest):
    if req.method == 'POST':
        print("post bosilib ketdi")
        # name of the folder that input data contains:

        # names of classes to analyse:
        classes = ['pos', 'neg']
        polarity_names = ['Negativ sentiment', 'Pozitiv sentiment']
        filenames = [POSITIVE_WORDS, NEGATIVE_WORDS]
        train_data = []
        train_labels = []
        # Read the data
        print("Reading the data...")
        sum_count_train = 0
        for polarity in range(2):
            print("Polarity class: " + classes[polarity] + ", Opening file: " + filenames[polarity] + ":\n")
            with open(os.path.join(DATA_DIR, filenames[polarity]), 'r', encoding='utf-8') as infile:
                count = 0
                for line in infile.readlines():
                    count += 1
                    review = line.strip('\n')
                    train_data.append(review)
                    label = 0
                    if (classes[polarity] == 'pos'):
                        label = 1
                    train_labels.append(label)
                print("\tNumber of reveiws: " + str(count))
                sum_count_train += count
        # reporting the read data:
        print("Total number of reveiws: " + str(sum_count_train))
        # .to_csv('./data/cleaned_text.csv')

        # --- build the model

        x_train, x_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.1, random_state=42,
                                                            stratify=train_labels)

        print(len(x_train), len(x_test), len(y_train), len(y_test))
        # Saving the test labels for future use
        pd.DataFrame(y_test).to_csv('predictions/y_true.csv', index=False, encoding='utf-8')

        # 1 - Bag of word model based on word ngrams
        print("### 1 - Bag of word model based on word ngrams\n")

        vectorizer_word = TfidfVectorizer(max_features=None,
                                          min_df=5,
                                          max_df=0.5,
                                          analyzer='word',
                                          ngram_range=(1, 2))

        vectorizer_word.fit(x_train)

        tfidf_matrix_word_train = vectorizer_word.transform(x_train)
        tfidf_matrix_word_test = vectorizer_word.transform(x_test)

        lr_word = LogisticRegression(solver='sag', verbose=2)
        lr_word.fit(tfidf_matrix_word_train, y_train)
        pickle.dump(lr_word, open(WORD_MODEL, 'wb'))
        pickle.dump(vectorizer_word, open(WORD_TRANSFORM, 'wb'))

        print("---------------------------------------------------------")

        vectorizer_char = TfidfVectorizer(max_features=None,
                                          min_df=5,
                                          max_df=0.5,
                                          analyzer='char',
                                          ngram_range=(1, 4))

        vectorizer_char.fit(x_train)

        tfidf_matrix_char_train = vectorizer_char.transform(x_train)
        tfidf_matrix_char_test = vectorizer_char.transform(x_test)

        lr_char = LogisticRegression(solver='sag', verbose=2)
        lr_char.fit(tfidf_matrix_char_train, y_train)

        pickle.dump(lr_char, open(CHAR_MODEL, 'wb'))
        pickle.dump(vectorizer_char, open(CHAR_TRANSFORM, 'wb'))

    return render(req, 'home/reinstall.html')


def about_project(req: HttpRequest):
    return render(req, 'home/about.html')
