import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
def train_model(data):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    df = pd.read_csv("admission_data.csv")
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    a_train, a_test, b_train, b_test = train_test_split(X, y, test_size=0.2, random_state=420)
    model = LogisticRegression()
    y_train_binary = np.where(b_train >= 0.5, 1, 0)
    y_test_binary = np.where(b_test >= 0.5, 1, 0)
    model.fit(a_train,y_train_binary)
    
    
    return model
data = pd.read_csv('admission_data.csv')


model = train_model(data)


with open('admission_model .pkl', 'wb') as f:
    pickle.dump(model, f)


model = pickle.load(open('C:\\Users\\91965\\OneDrive\\Desktop\\mini project\\project execution\\admission_model .pkl', 'rb'))

app = Flask(_name_)

@app.route('/')
def home():
    return render_template('user.html')

@app.route('/y_predict', methods=['POST'])
def y_predict():
    
    min1 = [290.0, 92.0, 1.0, 1.0, 1.0, 6.8, 0.0]
    max1 = [340.0, 120.0, 5.0, 5.0, 5.0, 9.92, 1.0]

   
    m= [float(x) for x in request.form.values()]
    p = []
    for i in range(7):
        l = (m[i] - min1[i]) / (max1[i] - min1[i])
        p.append(l)

   
    prediction = model.predict([p])
    output = prediction[0]

    
    if output == False:
        return render_template('nochance.html', prediction_text='You Dont have a chance')
    else:
        return render_template('chance.html', prediction_text='You have a chance')

if _name_ == "_main_":
    app.run(debug=True)
