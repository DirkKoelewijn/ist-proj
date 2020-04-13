import math
import pandas

from MLDPTree import MLDPTree


def k_fold_test(csv: str, k: int = 10):
    df = pandas.read_csv(csv)
    df = df.sample(frac=1).reset_index(drop=True)
    b = math.ceil(len(df) / k)

    overall_leafs = []
    overall_performance = []
    overall_actual = None
    overall_predicted = None
    print('-- K-FOLD TEST --')
    print(csv, '- k:', k)
    print('block size:', b)
    for i in range(k):
        # Divide data sets
        test = df.iloc[i * b:(i + 1) * b]
        train = pandas.concat([(df.iloc[:i * b]), (df.iloc[(i + 1) * b:])])

        # Train, predict and get actual
        model = MLDPTree.train(train)
        predicted = model.predict_all(test)
        actual = test['Class']

        if overall_actual is None:
            overall_actual = actual
            overall_predicted = predicted
        else:
            overall_actual = overall_actual.append(actual)
            overall_predicted = overall_predicted.append(predicted)

        # Get leafs and performance
        leafs = model.leaf_count()
        overall_leafs.append(leafs)
        performance = predicted == actual
        performance = len(performance[performance == True]) / len(performance)
        overall_performance.append(performance)

        print('\n- K =', i, '-')
        print('Leafs:', leafs)
        print('Performance:', round(performance * 100, 2), '%')
        print_cm(actual, predicted)

    print('\n- Overall performance - ')
    print(csv, '- k:', k)
    print('block size:', b)
    leafs_ = sum(overall_leafs) / len(overall_leafs)
    print('Leafs:', leafs_)
    performance_ = sum(overall_performance) / len(overall_performance)
    print('Performance:', round(performance_ * 100, 2), '%')
    print_cm(overall_actual, overall_predicted)
    print()

    return leafs_, performance_


def print_cm(actual, predicted):
    cm = pandas.DataFrame({'a': actual, 'p': predicted}, columns=['a', 'p'])
    print(pandas.crosstab(cm['a'], cm['p'], rownames=['Actual'], colnames=['Predicted']))


if __name__ == '__main__':
    to_test = ['data/seeds.csv',
               'data/banknotes.csv',
               'data/breast-cancer.csv',
               'data/diabetes.csv',
               'data/heart_disease.csv',
               'data/iris.csv',
               'data/tic-tac-toe.csv',
               'data/glass.csv',
               'data/wine.csv',
               'data/statlog.csv']

    res = {}

    # for csv in to_test:
    #     res[csv] = k_fold_test(csv, 10)
    #     print('\n--- RESULTS SO FAR ---')
    #     print('file', 'leafs', 'performance', sep=', ')
    #     for k, (l, p) in res.items():
    #         print(k, l, p, sep=', ')

    for csv in to_test:
        df = pandas.read_csv(csv)
        model = MLDPTree.train(df)
        print(csv, model.leaf_count())
