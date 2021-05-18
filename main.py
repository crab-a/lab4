from sys import argv
import os
from cross_validation import CrossValidation
from knn import KNN
from metrics import accuracy_score
from normalization import *


def load_data():
    """
    Loads data from path in first argument
    :return: returns data as list of Point
    """
    if len(argv) < 2:
        print('Not enough arguments provided. Please provide the path to the input file')
        exit(1)
    input_path = argv[1]

    if not os.path.exists(input_path):
        print('Input file does not exist')
        exit(1)

    points = []
    with open(input_path, 'r') as f:
        for index, row in enumerate(f.readlines()):
            row = row.strip()
            values = row.split(',')
            points.append(Point(str(index), values[:-1], values[-1]))
    return points


def run_knn(points):
    m = KNN(5)
    m.train(points)
    print(f'predicted class: {m.predict(points[0])}')
    print(f'true class: {points[0].label}')
    cv = CrossValidation()
    cv.run_cv(points, 10, m, accuracy_score)


def q2(k, points):
    m = KNN(k)
    m.train(points)
    cv = CrossValidation()
    return cv.run_cv(points, 10, m, accuracy_score, False)


def q3(k, points):
    m = KNN(k)
    m.train(points)
    cv = CrossValidation()

    print("Question 3:")
    print(f'K={k}')
    print("2-fold-cross-validation:")
    cv.run_cv(points, 2, m, accuracy_score, False, True)
    print("10-fold-cross-validation:")
    cv.run_cv(points, 10, m, accuracy_score, False, True)
    print("20-fold-cross-validation:")
    cv.run_cv(points, 20, m, accuracy_score, False, True)


def run_1nn(points):
    m = KNN(1)
    m.train(points)
    predicted = m.predict(points)
    real = []
    for point in points:
        real.append(point.label)
    print(accuracy_score(real, predicted))


if __name__ == '__main__':
    loaded_points = load_data()
    # run_1nn(loaded_points)
    best = 0
    best_k = 0
    for k in range(30):
        current = q2(k + 1, loaded_points)
        best = max(current, best)
        if best == current:
            best_k = k
    q3(best_k, loaded_points)
