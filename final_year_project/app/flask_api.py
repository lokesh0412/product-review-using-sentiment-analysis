from flask import Flask,abort,jsonify,request
import pickle

nb_classifier=pickle.load(open("nb_class.pkl","rb"))
def bag_of_words(words):
    return dict([(word,True) for word in words])
app=Flask(__name__)

@app.route('/api',methods=['POST'])
def make_predict():
    data=request.get_json(force=True)
    predict_request=data['rev']
    #output = []
    '''for i in range(len(predict_request)):
        predict=bag_of_words(predict_request[i]['review'].split())
        a = {'class':str(nb_classifier.classify(predict))}
        output.append(a) '''
    predict=bag_of_words(predict_request.split())
    output=[
    {'class':str(nb_classifier.classify(predict))}
    ]
    return jsonify({'results':output})

# function for getting classes for a list of reviews
@app.route('/predict',methods=['POST'])
def predict_classes():
    data = request.get_json(force=True)
    predict_request = data['rev']
    output = []
    for i in range(len(predict_request)):
        predict=bag_of_words(predict_request[i]['review'])
        a = {'class':str(nb_classifier.classify(predict))}
        output.append(a)
    return jsonify({'results': output})



if __name__ == '__main__':
    app.run(port =9000, debug =True)
