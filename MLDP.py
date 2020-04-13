import pandas

from MLDPTree import MLDPTree

if __name__ == '__main__':
    df = pandas.read_csv('data/iris.csv')
    d = MLDPTree.train(df)
    p = d.predict_all(df)
    data = {'y_Actual': df['Class'],
            'y_Predicted': p
            }
    cm = pandas.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
    confusion_matrix = pandas.crosstab(cm['y_Actual'], cm['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
    print(confusion_matrix)
    print()
    d.pprint()
