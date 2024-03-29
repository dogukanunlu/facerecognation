from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Initialize the classifiers
class SVMClassifier:
    def __init__(self):
        self.svm_classifier = SVC(kernel='linear')

    def train(self, X_train, y_train):
        self.svm_classifier.fit(X_train, y_train)

    def predict(self, X_test):
        return self.svm_classifier.predict(X_test)

    def evaluate(self, X_test, y_test):
        predictions = self.svm_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print("SVM Accuracy: ", accuracy)
        return accuracy, predictions

class KNNClassifier:
    def __init__(self, n_neighbors=3):
        self.knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)

    def train(self, X_train, y_train):
        self.knn_classifier.fit(X_train, y_train)

    def predict(self, X_test):
        return self.knn_classifier.predict(X_test)

    def evaluate(self, X_test, y_test):
        predictions = self.knn_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print("kNN Accuracy: ", accuracy)
        return accuracy, predictions

class RandomForestClassifier:
    def __init__(self, n_estimators=100):
        self.rf_classifier = RandomForestClassifier(n_estimators=n_estimators)

    def train(self, X_train, y_train):
        self.rf_classifier.fit(X_train, y_train)

    def predict(self, X_test):
        return self.rf_classifier.predict(X_test)

    def evaluate(self, X_test, y_test):
        predictions = self.rf_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print("Random Forest Accuracy: ", accuracy)
        return accuracy, predictions
