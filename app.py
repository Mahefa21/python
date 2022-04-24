from flask import Flask

import json
import string
import random 
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
import tensorflow as tf 
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense, Dropout
nltk.download("punkt")
nltk.download("wordnet")
nltk.download('omw-1.4')
import requests

STRAPI_CHATBOT_ENDPOINT = 'http://localhost:1337/api/chatbot'
def fetch_chatbot():
    response = requests.get(STRAPI_CHATBOT_ENDPOINT)
    json_response = json.loads(response.text)
    attributes = json_response['data']['attributes']
    intents = []
    for intent in attributes['intents']:
        tag = intent['tag']
        patterns = [p.get('pattern') for p in intent['patterns']]
        responses = [p.get('content') for p in intent['answers']]
        intents.append({'tag': tag, 'patterns': patterns, 'responses': responses})
    res = {'intents': intents, 'default_answers': [p.get('content') for p in attributes['defaultAnswers']], 'init_messages':  [p.get('content') for p in attributes['initMessages']]}
    return res
def clean_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens]
    return tokens
def bag_of_words(text, vocab):
    tokens = clean_text(text)
    bow = [0] * len(vocab)
    contains = False
    for w in tokens:
        for idx, word in enumerate(vocab):
            if word == w:
                contains = True
                bow[idx] = 1
    return {"bow": np.array(bow), "contains": contains}
def pred_class(text, vocab, labels):
    raw_bow = bag_of_words(text, vocab)
    if(not raw_bow['contains']):
        return ['default_answers']
    bow = raw_bow['bow']
    result = model.predict(np.array([bow]))[0]
    thresh = 0.2
    y_pred = [[idx, res] for idx, res in enumerate(result)if res > thresh]
    y_pred.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in y_pred:
        return_list.append(labels[r[0]])
    return return_list
def get_response(intents_list, intents_json):
    tag = intents_list[0]
    list_of_intents = intents_json["intents"]
    if tag == 'default_answers':
        return random.choice(intents_json["default_answers"])
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result

app = Flask(__name__)
@app.route('/')
# def helloIndex():
#     return 'Hello World from Python Flask!'

def messages_bot():
    message = request.args.get("message", default="", type=str)   
    print(message)
    intents = pred_class(message, words, classes)
    result = get_response(intents, data)
    return jsonify({"response": result});
@app.route("/init")
def init_message():
    result = random.choice(data['init_messages'])
    return jsonify({"response": result});
@app.route("/train")
def train_bot(inside_context = True):
    global words
    global classes
    global data
    global lemmatizer
    global model
    data = fetch_chatbot()
    lemmatizer = WordNetLemmatizer()
    words = []
    classes = []
    # création des listes
    doc_X = []
    doc_y = []
    # parcourir avec une boucle For toutes les intentions
    # tokéniser chaque pattern et ajouter les tokens à la liste words, les patterns et
    # le tag associé à l'intention sont ajoutés aux listes correspondantes
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            tokens = nltk.word_tokenize(pattern)
            words.extend(tokens)
            doc_X.append(pattern)
            doc_y.append(intent["tag"])
        # ajouter le tag aux classes s'il n'est pas déjà là
        if intent["tag"] not in classes:
            classes.append(intent["tag"])
    # lemmatiser tous les mots du vocabulaire et les convertir en minuscule
    # si les mots n'apparaissent pas dans la ponctuation
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]
    # trier le vocabulaire et les classes par ordre alphabétique et prendre le
    # set pour s'assurer qu'il n'y a pas de doublons
    words = sorted(set(words))
    classes = sorted(set(classes))
    tmp_model = None
    training = []
    out_empty = [0] * len(classes)
    # création du modèle d'ensemble de mots
    # création du modèle d'ensemble de mots
    for idx, doc in enumerate(doc_X):
        bow = []
        text = lemmatizer.lemmatize(doc.lower())
        for word in words:
            bow.append(1) if word in text else bow.append(0)
        # marque l'index de la classe à laquelle le pattern atguel est associé à
        output_row = list(out_empty)
        output_row[classes.index(doc_y[idx])] = 1
        # ajoute le one hot encoded BoW et les classes associées à la liste training
        training.append([bow, output_row])
    # mélanger les données et les convertir en array
    random.shuffle(training)
    training = np.array(training, dtype=object)
    # séparer les features et les labels target
    train_X = np.array(list(training[:, 0]))
    train_y = np.array(list(training[:, 1]))
    # définition de quelques paramètres
    input_shape = (len(train_X[0]),)
    output_shape = len(train_y[0])
    epochs = 200
    # modèle Deep Learning
    tmp_model = Sequential()
    tmp_model.add(Dense(128, input_shape=input_shape, activation="relu"))
    tmp_model.add(Dropout(0.5))
    tmp_model.add(Dense(64, activation="relu"))
    tmp_model.add(Dropout(0.3))
    tmp_model.add(Dense(output_shape, activation = "softmax"))
    adam = tf.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)
    tmp_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])
    tmp_model.fit(x=train_X, y=train_y, epochs=200, verbose=1)
    tmp_model.save('.')
    model = tmp_model
    if inside_context:
        return jsonify({"status": "Model Trained"})
    else:
        return True

if __name__ == '__main__':
    app.run()
