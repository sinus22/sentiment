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


def home_index(req: HttpRequest):
    form = TestForm(req.POST or None)
    result = {}
    if req.method == 'POST':
        print(form.data.get('text'))

        print("isop")
        result['data'] = '100%'
        input_sentence = form.data.get('text')
        filename = 'sentiment_model.sav'
        filename1 = 'transform_model.sav'
        lr_word = pickle.load(open(filename, 'rb'))
        vectorizer_word = pickle.load(open(filename1, 'rb'))
        input_text_vectorized = vectorizer_word.transform([input_sentence])
        predicted_polarity = lr_word.predict(input_text_vectorized)
        predicted_scores = lr_word.predict_proba(input_text_vectorized)[0]
        polarity_names = ['Negativ sentiment', 'Pozitiv sentiment']
        print("Matndagi sentiment:", polarity_names[predicted_polarity[0]])
        print("Predicted Scores:")
        print("Positive:", predicted_scores[1])
        print("Negative:", predicted_scores[0])
        result = {
            'text': polarity_names[predicted_polarity[0]],
            'positive': int(predicted_scores[1] * 100),
            'negative': int(predicted_scores[0] * 100)
        }
    return render(req, 'home/index.html', {
        'form': form,
        'result': result
    })


def model_reinstall(req: HttpRequest):
    if req.method == 'POST':
        print("post bosilib ketdi")
        # name of the folder that input data contains:
        data_dir = "data"
        # names of classes to analyse:
        classes = ['pos', 'neg']
        polarity_names = ['Negativ sentiment', 'Pozitiv sentiment']
        filenames = ["UZ_positive.txt", "UZ_negative.txt"]
        train_data = []
        train_labels = []
        # Read the data
        print("Reading the data...")
        sum_count_train = 0;
        for polarity in range(2):
            print("Polarity class: " + classes[polarity] + ", Opening file: " + filenames[polarity] + ":\n")
            with open(os.path.join(data_dir, filenames[polarity]), 'r', encoding='utf-8') as infile:
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
        filename = 'sentiment_model.sav'
        filename1 = 'transform_model.sav'
        pickle.dump(lr_word, open(filename, 'wb'))
        pickle.dump(vectorizer_word, open(filename1, 'wb'))
        # print("---------------------------------------------------------")

    return render(req, 'home/reinstall.html')
