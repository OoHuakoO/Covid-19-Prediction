import time
from flask import Flask, request
import pickle
import sklearn
import numpy as np

app = Flask(__name__)


@app.route('/time',  methods=['GET'])
def get_current_time():
    return {'time': time.time()}


@app.route('/Covid-19', methods=['POST'])
def predict():
    data = request.get_json()
    #print(data)
    age = data['age']
    sex = data['sex']
    # city = data['city']
    # province = data['province']
    # country = data['country']
    # history = data['history']
    hypertension = data['hypertension']
    diabetes = data['diabetes']
    Kidney = data['Kidney']
    COPD = data['COPD']
    Heart = data['Heart']
    Asthma = data['Asthma']
    Prostate = data['Prostate']
    Cancer = data['Cancer']
    Tuberculosis = data['Tuberculosis']
    Hepatitis = data['Hepatitis']
    HIV = data['HIV']
    Cereberal = data['Cereberal']
    Parkinson = data['Parkinson']
    Bronchitis = data['Bronchitis']
    Hypothyroidism = data['Hypothyroidism']
    Dyslipidemia = data['Dyslipidemia']
    anorexia = data['anorexia']
    chest = data['chest']
    chills = data['chills']
    conjunctivitis = data['conjunctivitis']
    cough = data['cough']
    diarrhea = data['diarrhea']
    dizziness = data['dizziness']
    dyspnea = data['dyspnea']
    emesis = data['emesis']
    expectoration = data['expectoration']
    eye = data['eye']
    fatigue = data['fatigue']
    fever = data['fever']
    Gasp = data['Gasp']
    Headache = data['Headache']
    Kidneyfailure = data['Kidneyfailure']
    SymptomHypertension = data['SymptomHypertension']
    Myalgia = data['Myalgia']
    Obnubilation = data['Obnubilation']
    Pneumonia = data['Pneumonia']
    Myelofibrosis = data['Myelofibrosis']
    Respiratorydistress = data['Respiratorydistress']
    Rhinorrhea = data['Rhinorrhea']
    Shortnessofbreath = data['Shortnessofbreath']
    Somnolence = data['Somnolence']
    Sorethroat = data['Sorethroat']
    Sputum = data['Sputum']
    Septicshock = data['Septicshock']
    Heartattack = data['Heartattack']
    Cold = data['Cold']
    Hypoxia = data['Hypoxia']

    predictData = [float(age), float(sex), float(hypertension), float(diabetes), float(Kidney), float(COPD), float(Heart), float(Asthma),
                    float(Prostate), float(Cancer), float(Tuberculosis),float(Hepatitis),float(HIV),float(Cereberal),float(Parkinson),float(Bronchitis),
                    float(Hypothyroidism),float(Dyslipidemia),float(anorexia),float(chest),float(chills),float(conjunctivitis),float(cough),float(diarrhea),
                    float(dizziness),float(dyspnea),float(emesis),float(expectoration),float(eye),float(fatigue),float(fever),float(Gasp),
                    float(Headache),float(Kidneyfailure),float(SymptomHypertension),float(Myalgia),float(Obnubilation),float(Pneumonia),float(Myelofibrosis),float(Respiratorydistress),
                    float(Rhinorrhea),float(Shortnessofbreath),float(Somnolence),float(Sorethroat),float(Sputum),float(Septicshock),float(Heartattack),float(Cold),float(Hypoxia)]
   # test = [float(data)]
    print(predictData)
    print('******1')
    logistic_regression = pickle.load(open('../model/NeuralNetwork.pkl', 'rb'))
    print('******2')
    #X_test=tvect.transform(predictData)
    logreg_prediction = logistic_regression.predict([predictData])[0]
    print('******3')

    # knn = pickle.load(open('../models/KNN.pkl', 'rb'))
    # knn_prediction = knn.predict([predictData])[0]

    # neural_network = pickle.load(open('../models/NeuralNetwork.pkl', 'rb'))
    # nn_prediction = neural_network.predict([predictData])[0]

    # print('logreg : '+logreg_prediction)
    # print('knn : '+knn_prediction)
    # print('nn : '+nn_prediction)
    print('***************************************************')
    print(logreg_prediction)
    return str(logreg_prediction)


# if __name__ == '__main__':
#     app.run()