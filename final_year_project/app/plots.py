import pickle
from classifier import split_label_feats
from classifier import data
from classifier import label_feats_extractor
import nltk
from nltk.metrics import ConfusionMatrix
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
    plt.savefig('performace.png')
    return 1

#def makePieChart(data,labels):
