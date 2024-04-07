from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier as SKRandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np

# Initialize the classifiers
class SVMClassifier:
    def __init__(self):
        self.svm_classifier = SVC(kernel='linear')
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def train(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.svm_classifier.fit(X_train, y_train)

    def predict(self, X_test):
        self.X_test = X_test
        return self.svm_classifier.predict(X_test)

    def evaluate(self, X_test, y_test):
        self.y_test = y_test
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print("SVM Accuracy: ", accuracy)
        return accuracy, predictions
    
    def cross_validation(self):
        images_combined = np.concatenate((self.X_train, self.X_test))
        labels_combined = np.concatenate((self.y_train, self.y_test))
        
        # Standardize features for the combined dataset (inside cross-validation)
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        cv_score = cross_val_score(self.svm_classifier, images_combined, labels_combined, cv=cv)
        print("Cross Validation Accuracy: ", cv_score)

class KNNClassifier:
    def __init__(self, n_neighbors=3):
        self.knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def train(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.knn_classifier.fit(X_train, y_train)

    def predict(self, X_test):
        self.X_test = X_test
        return self.knn_classifier.predict(X_test)

    def evaluate(self, X_test, y_test):
        self.y_test = y_test
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print("kNN Accuracy: ", accuracy)
        return accuracy, predictions
    
    def cross_validation(self):
        images_combined = np.concatenate((self.X_train, self.X_test))
        labels_combined = np.concatenate((self.y_train, self.y_test))
        
        # Standardize features for the combined dataset (inside cross-validation)
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        cv_score = cross_val_score(self.knn_classifier, images_combined, labels_combined, cv=cv)
        print("Cross Validation Accuracy: ", cv_score)

class RandomForestClassifier:
    def __init__(self, n_estimators=100):
        self.rf_classifier = SKRandomForestClassifier(n_estimators=n_estimators)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def train(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.rf_classifier.fit(X_train, y_train)

    def predict(self, X_test):
        self.X_test = X_test
        return self.rf_classifier.predict(X_test)

    def evaluate(self, X_test, y_test):
        self.y_test = y_test
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print("Random Forest Accuracy: ", accuracy)
        return accuracy, predictions
    
    def cross_validation(self):
        images_combined = np.concatenate((self.X_train, self.X_test))
        labels_combined = np.concatenate((self.y_train, self.y_test))
        
        # Standardize features for the combined dataset (inside cross-validation)
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        cv_score = cross_val_score(self.rf_classifier, images_combined, labels_combined, cv=cv)
        print("Cross Validation Accuracy: ", cv_score)
