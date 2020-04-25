from pandas import read_csv
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import random
import sys


def get_random_size(_max=None):
    while 1:
        value = random.random()
        # Without this, there exists a case where a test will return an
        # empty train set as we're effectively saying we want to only have
        # 1% of the data be used for training the model, and that's not enough.
        if value < 0.99:
            yield value


class IrisDataset(object):
    def __init__(self):
        self.url = "https://raw.githubusercontent.com/jbrownlee\
/Datasets/master/iris.csv"
        self.names = ['sepal-length', 'sepal-width',
                      'petal-length', 'petal-width', 'class']
        self.dataset = self.load_data_set()
        self.values = self.dataset.values
        self.random_ratios = self.get_five_ratios()
        self.trained_model = self.train_model()

    def load_data_set(self):
        dataset = read_csv(self.url, names=self.names)
        return dataset

    def train_model(self):
        values = self.values
        # TODO: wtf is this
        x = values[:, 0:4]
        # TODO: wtf is this
        y = values[:, 4]
        model_objects = list()
        for z in self.random_ratios:
            x_train, x_valid, y_train, y_valid = train_test_split(
                                                            x,
                                                            y,
                                                            test_size=z,
                                                            random_state=0)
            model_object = {"x_train": x_train,
                            "y_train": y_train,
                            "x_valid": x_valid,
                            "y_valid": y_valid}
            model_objects.append(model_object)
        return model_objects

    def predict_outcomes(self):
        model = SVC(gamma='auto')
        trained_models = self.trained_model
        for trained_model in trained_models:
            model.fit(trained_model['x_train'], trained_model['y_train'])
            predictions = model.predict(trained_model['x_valid'])
            print(accuracy_score(trained_model['y_valid'], predictions))
            print(confusion_matrix(trained_model['y_valid'], predictions))
            print(classification_report(trained_model['y_valid'], predictions))

    def get_five_ratios(self):
        ratios = list()
        for i, x in enumerate(get_random_size()):
            if i > 5:
                break
            ratios.append(x)
        return ratios


if __name__ == '__main__':
    try:
        iris = IrisDataset()
        iris.predict_outcomes()
    except KeyboardInterrupt:
        print('Exiting on user cancel')
        sys.exit(0)
