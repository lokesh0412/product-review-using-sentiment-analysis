from flask import Flask,render_template,request
from readData import preprocessing,getUnprocessedData
import requests,json
import pandas as pd
from readData import calculateSentimentScore
app = Flask(__name__)
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
# code to show total no of comments and postive and negative comments
    """result = calculateSentimentScore(classes)
    total = result[0]
    positive = result[1]
    negative = result[2]
    return render_template('showResult.html',Total = total, Positive=positive,Negative=negative)"""
    unclassified_data = getUnprocessedData(product,classes)
    return render_template('showResult.html',unprocessed = unclassified_data)



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
