import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def find_similar_question(input_question):
    with open("questions.json", "r") as f:
        questions_json = json.load(f)

    question_keys = list(questions_json.keys())
    question_descriptions = []
    for key in question_keys:
        desc = questions_json[key]["description"]
        if isinstance(desc, list):
            desc = " ".join(desc)
        question_descriptions.append(desc)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(question_descriptions)

    input_question_vector = vectorizer.transform([input_question])
    cosine_similarities = cosine_similarity(input_question_vector, tfidf_matrix).flatten()
    most_similar_question_index = np.argmax(cosine_similarities)

    most_similar_question = question_keys[most_similar_question_index]
    most_similar_question_description = question_descriptions[most_similar_question_index]

    return (most_similar_question, most_similar_question_description)

