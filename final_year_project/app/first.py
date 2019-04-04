from flask import Flask,render_template,request
from readData import preprocessing,getUnprocessedData
import requests,json
import pandas as pd
from readData import calculateSentimentScore,writeOutputToFile
import matplotlib.pyplot as plt
from flask import Flask, session, redirect, url_for, escape, request
import pickle
from plots import performanceEvaluation,getRatingCount,comparision


app = Flask(__name__)
app.secret_key = 'sentiment analyser'

@app.route('/')
def home():
    return render_template("test.html")


# function for processing of product searched by users
@app.route('/product',methods=['POST','GET'])
def getDataToClassify():
    product = ''
    if request.method =='POST':
        product = request.form['Product']
    else:
        product = request.args.get('Product')

    url="http://127.0.0.1:9000/predict"
# Get file data
    #reviews = getFileData(product)
    reviews = preprocessing(product)
    data=json.dumps({'rev':reviews})
    r = requests.post(url,data)
    classes = r.json()['results']
#writing classes to file
    writeOutputToFile(classes,product)
#adding calculated classes to session    
    session['classes']=classes
    session['product']=product
# code to show total no of comments and postive and negative comments
    """result = calculateSentimentScore(classes)
    total = result[0]
    positive = result[1]
    negative = result[2]
    return render_template('showResult.html',Total = total, Positive=positive,Negative=negative)"""
    unclassified_data = getUnprocessedData(product,classes)
    return render_template('showResult.html',unprocessed = unclassified_data)



@app.route('/summary',methods=['POST','GET'])
def getSummary():
    nb_classifier=pickle.load(open("nb_class.pkl","rb"))
    classes = session['classes']
    data = {'results':classes}
    result = calculateSentimentScore(data)
    fig1 = plt.figure()
    percentage = [result[1]/result[0], result[2]/result[0]]
    label = ['Positive', 'Negative']
    colors = ['r', 'g']
    plt.pie(percentage, labels=label, colors=colors, startangle=90, autopct='%.1f%%')
    fig1.savefig('static/summary.png')
    fig1.clear()
    """nb_classifier=pickle.load(open("nb_class.pkl","rb"))
    informative_data = nb_classifier.most_informative_features(20)
    freq_dist_data = []
    for word,label in informative_data:
        freq_dist_data.append(word)"""
    performanceEvaluation()
    product = session['product']
    getRatingCount(product)
    comparision(result[1],result[2],product)
    return render_template('summary.html')

# function for processing of single comment
@app.route('/comment',methods=['POST','GET'])
def getProduct():
    product = ''
    if request.method =='POST':
        product = request.form['Product']
    else:
        product = request.args.get('Product')

    url="http://127.0.0.1:9000/api"
    data=json.dumps({'rev':product})
    r=requests.post(url,data)
    data=r.json()
    classlbl = data['results'][0]['class']
    return render_template('result.html', result = classlbl)

if __name__=='__main__':
    app.run(debug=True)
