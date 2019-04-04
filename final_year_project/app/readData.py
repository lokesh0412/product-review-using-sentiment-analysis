from nltk.tokenize import word_tokenize
import numpy as np
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pandas as pd
import matplotlib.pyplot as plt


def preprocessing(file):
    filename="files\\"+file+".csv"
    df=pd.read_csv(filename)
    data = df.to_dict()
    content = []
    porter = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    for i in range(0,len(data['content'])):
        c = word_tokenize((str(data['content'][i])))
       # c = str(data['content'][i]).split(" ")
        c = [item for item in c if len(item) >=2]
   # normalising words
        c = [ word.lower() for word in c ]
        c = [word for word in c if word.isalpha()]
        c = [w for w in c if not w in stop_words]
        c = [porter.stem(word) for word in c]
        a = {'review':c}
        content.append(a)
    return content
   # filtering stopwords

def preprocessForTraining(words):
    porter = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    words = [item for item in words if len(item) >=2]
    # normalising words
    words = [ word.lower() for word in words ]
    words = [word for word in words if word.isalpha()]
    words = [w for w in words if not w in stop_words]
    words = [porter.stem(word) for word in words]
    return words


#function for getting random unclassified comments
def getUnprocessedData(file,classes):
    filename="files\\"+file+".csv"
    df=pd.read_csv(filename)
    data = df.to_dict('records')
    result = []
    for i in range(20):
        a = {'content':data[i]['content'],'rating':data[i]['rating'],'label':classes[i]['class']}
        result.append(a)
    return result

# function for reading data from file and preparing json data
def getFileData(file):
    result = []
    i=0
    filename="files\\final_project_"+file+".txt"
    infile = open(filename,"r")
    for line in infile:
        if not line.strip():
            continue
        else:
            i=i+1
            if i%5 == 0:
                a = {'review':line}
                result.append(a)
    return result



#function for calculating sentiment score from the labelled data
def calculateSentimentScore(data):
    output = data['results']
    positiveCount = 0
    negativeCount = 0
    result = []
    l = len(output)
    for i in range(len(output)):
        if output[i]['class'] == '1':
            positiveCount = positiveCount + 1
        else:
            negativeCount = negativeCount + 1

    result.append(l)
    result.append(positiveCount)
    result.append(negativeCount)
    return result

        
def classifyUsingRating(file):
    count1=0
    filename = "files\\"+file+".csv"
    df=pd.read_csv(filename)
    data = df.to_dict('records')
    count2=0
    count3=0
    count4=0
    count5=0
    for i in range(len(data)):
        if(str(data[i]['rating']).split(" ")[0]=='1.0'):
            count1=count1+1
        elif(str(data[i]['rating']).split(" ")[0]=='2.0'):
            count2=count2+1
        elif(str(data[i]['rating']).split(" ")[0]=='3.0'):
            count3=count3+1
        elif(str(data[i]['rating']).split(" ")[0]=='4.0'):
            count4=count4+1
        else:
            count5=count5+1
    positive=count3+count4+count5
    negative=len(data)-positive
    result = [positive,negative]
    return result

def writeOutputToFile(result,product):
    file = open('output.txt','w')
    file.write('class labels for'+product)
    for i in range(len(result)):
        file.write(result[i]['class'])
        file.write('\n')
    file.close() 