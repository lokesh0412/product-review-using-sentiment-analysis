import pandas as pd
def getData(productName):
    url = "files/final_project_"+productName+".txt";
    data = pd.read_csv(url,delimiter=",",header=None)
    dataForClassification=[];
    #for i in range(6):
        #a = data[i][4]
    #    b = { 'review':data[i][4]}
    #    dataForClassification.append(b)
    #return dataForClassification
    a = data[2][4]
    print(a)

result = getData("iphone6")
