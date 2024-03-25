from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def svm(X_train, y_train, X_test, y_test):
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train, y_train)
    svm_predictions = svm_model.predict(X_test)
    
    svm_accuracy = accuracy_score(y_test, svm_predictions)
    print(f'SVM Accuracy: {svm_accuracy}')

def knn(X_train, y_train, X_test, y_test):
    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(X_train, y_train)
    knn_predictions = knn_model.predict(X_test)
    
    knn_accuracy = accuracy_score(y_test, knn_predictions)
    print(f'kNN Accuracy: {knn_accuracy}')
    
def random_forest(X_train, y_train, X_test, y_test):
    rf_model = RandomForestClassifier(n_estimators=100)
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)
    
    rf_accuracy = accuracy_score(y_test, rf_predictions)
    print(f'Random Forest Accuracy: {rf_accuracy}')