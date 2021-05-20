from point import Point
from numpy import mean, var


class DummyNormalizer:
    def fit(self, points):
        pass

    def transform(self, points):
        return points


class L1Norm:
    def __init__(self):
        self.sum_abs_list = []

    def fit(self, points):
        all_coordinates = [p.coordinates for p in points]
        self.sum_abs_list = []
        for i in range(len(all_coordinates[0])):
            values = [abs(x[i]) for x in all_coordinates]
            self.sum_abs_list.append(sum(values))

    def transform(self, points):
        new = []
        for p in points:
            new_coordinates = []
            n_coordinates = p.coordinates
            for i, x in enumerate(n_coordinates):
                new_coordinates.append(x / self.sum_abs_list)
            new.append(Point(p.name, new_coordinates, p.label))
        return new


class ZNormalizer:
    def __init__(self):
        self.mean_variance_list = []

    def fit(self, points):
        all_coordinates = [p.coordinates for p in points]
        self.mean_variance_list = []
        for i in range(len(all_coordinates[0])):
            values = [x[i] for x in all_coordinates]
            self.mean_variance_list.append([mean(values), var(values, ddof=1) ** 0.5])

    def transform(self, points):
        new = []
        for p in points:
            new_coordinates = p.coordinates
            new_coordinates = [(new_coordinates[i] - self.mean_variance_list[i][0]) / self.mean_variance_list[i][1]
                               for i in range(len(p.coordinates))]
            new.append(Point(p.name, new_coordinates, p.label))
        return new
