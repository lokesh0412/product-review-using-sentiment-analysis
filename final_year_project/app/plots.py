import pickle
from classifier import split_label_feats
from classifier import data
from classifier import label_feats_extractor
import nltk
from nltk.metrics import ConfusionMatrix
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
import numpy as np
from nltk.probability import FreqDist
import matplotlib.pyplot as plt; plt.rcdefaults()
from readData import classifyUsingRating


nb_classifier=pickle.load(open("nb_class.pkl","rb"))
def performanceEvaluation():
    lfeats=label_feats_extractor(data)
    train_feats, test_feats = split_label_feats(lfeats, split=0.75)
    actual_output = []
    data_to_classify = []
    predicted_output = []
    for i in range(len(test_feats)):
        actual_output.append(test_feats[i][1])
        predicted_output.append(nb_classifier.classify(test_feats[i][0]))
    PPV_precision = 0
    cm = ConfusionMatrix(actual_output,predicted_output)
    labels = set([0,1])
    true_positives = 0
    false_negatives = 0
    false_positives = 0
    true_negatives = 0
    true_positives = cm[1,1]
    false_positives= cm[0,1]
    false_negatives= cm[1,0]
    true_negatives = cm[0,0]
    PPV_precision = true_positives/(true_positives+false_positives)
    PPV_precision *=100
    recall = true_positives/(true_positives+false_negatives)
    recall *=100
    accuracy = (true_positives+true_negatives)/(true_positives+true_negatives+false_positives+false_negatives)
    accuracy *=100
    f_score = (2*PPV_precision*recall)/(PPV_precision+recall)
    print(PPV_precision,recall,accuracy,f_score)
    frequencies = [PPV_precision,recall,accuracy,f_score]
# In my original code I create a series and run on that,
# so for consistency I create a series from the list.
    freq_series = pd.Series.from_array(frequencies)
    x_labels = ['Precision','Recall','Accuracy','FScore']
# Plot the figure.
    plt.figure(figsize=(10, 8))
    ax = freq_series.plot(kind='bar')
    ax.set_title('Classifier Performance')
    ax.set_xlabel('Parameters')
    ax.set_ylabel('Scores')
    ax.set_xticklabels(x_labels)
    rects = ax.patches
# Make some labels.
    labels = [PPV_precision,recall,accuracy,f_score]
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height + 1, label,
            ha='center', va='bottom')
    plt.savefig('static/performace.png')
    plt.close()
    return 1


def makeFreqDistOfGivenProduct(file):
    filename="files\\"+file+".csv" 
    df=pd.read_csv(filename)
    data = df.to_dict()
    content = []
    content = word_tokenize(data['content'][1])
    for i in range(2,len(data['content'])):
        c = word_tokenize((data['content'][i]))
        c = [item for item in c if len(item) >=2]
        content.extend(c)
   # normalising words
    content = [ word.lower() for word in content ]
    content = [word for word in content if word.isalpha()]
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    words = [w for w in content if not w in stop_words]
    from nltk.stem.porter import PorterStemmer
    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in words]
    fig = plt.figure()
    fdist = FreqDist(stemmed)
    fdist.plot(20,cumulative=False)
    fig.savefig('static/freqdistofproduct.png')
    fig.clear()
    return 1

    
#def makePieChart(data,labels):
def getRatingCount(file):
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

    fig = plt.figure()
    objects = ('One Star','Two Start','Three Star','Four Star','Five Star')
    y_pos = np.arange(len(objects))
    performance = [count1,count2,count3,count4,count5]
    plt.barh(y_pos, performance, align='center', alpha=0.5)
    plt.yticks(y_pos, objects)
    plt.xlabel('Count')
    plt.title('Rating count goiven by users for searched product')
    fig.savefig('static/rating.png')
    return 1

def comparision(positive,negative,product):
    d = classifyUsingRating(product)
    type(classifyUsingRating(product))
    fig = plt.figure()
    objects = ('PosRat','PosCls','NegRat','NegCls')
    y_pos = np.arange(len(objects))
    performance = [d[0],positive,d[1],negative]
    plt.barh(y_pos, performance, align='center', alpha=0.5)
    plt.yticks(y_pos, objects)
    plt.xlabel('Count')
    plt.title('Comparison between rating and prediction by model')
    fig.savefig('static/comparision.png')
    return 1
    