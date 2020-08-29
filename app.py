from flask import Flask,render_template,url_for,request
import pandas as pd 
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import pickle
import flask


filename ='pickle.pkl'
clf = pickle.load(open(filename,'rb'))
vectorizer=pickle.load(open('tranformer.pkl','rb'))

app = Flask(__name__, template_folder='template', static_folder='/home/timon007/Desktop/Data Analysis with python UDEMY/NLP/spam classifier/static/static')
@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = vectorizer.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
	app.run()