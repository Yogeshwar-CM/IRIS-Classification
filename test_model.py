import unittest
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

class TestIrisModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        iris = datasets.load_iris()
        cls.X = iris.data
        cls.y = iris.target
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(cls.X, cls.y, test_size=0.3, random_state=42)

        # Standardize the features
        cls.scaler = StandardScaler()
        cls.X_train = cls.scaler.fit_transform(cls.X_train)
        cls.X_test = cls.scaler.transform(cls.X_test)

        # Create SVM model
        cls.model = SVC(kernel='linear')
        cls.model.fit(cls.X_train, cls.y_train)

    def test_model_accuracy(self):
        accuracy = self.model.score(self.X_test, self.y_test)
        self.assertGreater(accuracy, 0.8, "Model accuracy should be greater than 80%")

    def test_model_predict(self):
        predictions = self.model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test), "Number of predictions should match number of test samples")

if __name__ == '__main__':
    unittest.main()