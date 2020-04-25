from pandas import read_csv
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import random
import sys


def get_random_size(_max=None):
    """
    Generator function to create random floats

    :_max: The max float value that's allowed to be returned
    :returns: Float value between 0 and _max
    """

    if not _max:
        _max = 0.99
    while 1:
        value = random.random()
        # Without this, there exists a case where a test will return an
        # empty train set as we're effectively saying we want to only have
        # 1% of the data be used for training the model, and that's not enough.
        if value < _max:
            yield value


class IrisDataset(object):
    def __init__(self):
        """
        Init function, called on object initialization

        Grabs the CSV data from the github source and separates it out
        into values so sklearn can use it. Then forwards it to sklearn
        to train the model.
        """
        self.url = "https://raw.githubusercontent.com/jbrownlee\
/Datasets/master/iris.csv"
        self.names = ['sepal-length', 'sepal-width',
                      'petal-length', 'petal-width', 'class']
        self.dataset = self.load_data_set()
        self.values = self.dataset.values
        self.random_ratios = self.get_five_ratios()
        self.trained_model = self.train_model()

    def load_data_set(self):
        """
        Loads the dataset into CSV format using pandas and

        :returns: Pandas CSV object
        """
        dataset = read_csv(self.url, names=self.names)
        return dataset

    def train_model(self):
        """
        Trains the Iris model with the random values defined

        :returns: List of objects representing trained data results
        """
        values = self.values
        # String splitting to only get numerical data
        x = values[:, 0:4]
        # String split to only get flower names
        y = values[:, 4]
        model_objects = list()
        for z in self.random_ratios:
            x_train, x_valid, y_train, y_valid = train_test_split(
                                                            x,
                                                            y,
                                                            test_size=z,
                                                            random_state=None)
            model_object = {"x_train": x_train,
                            "y_train": y_train,
                            "x_valid": x_valid,
                            "y_valid": y_valid,
                            "z": z}
            model_objects.append(model_object)
        return model_objects

    def predict_outcomes(self):
        """
        Predict outcomes based on model returns and then grade them
        using sklearn
        """
        model = SVC(gamma='auto')
        trained_models = self.trained_model
        for trained_model in trained_models:
            print(trained_model['z'])
            model.fit(trained_model['x_train'], trained_model['y_train'])
            predictions = model.predict(trained_model['x_valid'])
            print(accuracy_score(trained_model['y_valid'], predictions))
            print(confusion_matrix(trained_model['y_valid'], predictions))
            print(classification_report(trained_model['y_valid'], predictions))

    def get_five_ratios(self):
        """
        Get five ratios from our generator function to forward into
        the  model training method

        :returns: List of Floats
        """
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
