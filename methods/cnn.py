from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from keras.applications import VGG19
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.regularizers import l2
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from keras.regularizers import l2
from tensorflow.keras.layers import GlobalAveragePooling2D
import matplotlib.pyplot as plt
import itertools
import numpy as np


emotion_dict = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

class cnn:
    def __init__(self, input_shape=(48, 48, 1), num_classes=7):
        self.model = self.create_model(input_shape, num_classes)
        self.label_encoder = LabelEncoder()  

    def create_model(self, input_shape, num_classes):
        model = Sequential([
            Input(shape=input_shape),
            Conv2D(32, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, train_generator, epochs=50):
        history = self.model.fit(train_generator, epochs=epochs)
        return history

    def evaluate(self, test_generator):
        test_generator.reset()
        
        num_samples = test_generator.samples
        
        predictions = self.model.predict(test_generator, steps=np.ceil(num_samples / test_generator.batch_size))
        
        predicted_class_indices = np.argmax(predictions, axis=1)
        
        true_class_indices = test_generator.classes
        
        labels = (test_generator.class_indices)
        labels = dict((v,k) for k,v in labels.items())
        predictions_labels = [labels[k] for k in predicted_class_indices]
        
        cm = confusion_matrix(true_class_indices, predicted_class_indices)
        
        print(classification_report(true_class_indices, predicted_class_indices, target_names=list(labels.values())))
        
        plt.figure(figsize=(10, 8))
        self.plot_confusion_matrix(cm, classes=list(labels.values()), title='Confusion Matrix')
        plt.show()

        def predict_single_image(self, image_array):
            image_resized = np.resize(image_array, (48, 48, 1))
            image_expanded = np.expand_dims(image_resized, axis=0)

            prediction = self.model.predict(image_expanded)
            predicted_class = np.argmax(prediction, axis=1)
            predicted_emotion = emotion_dict[predicted_class[0]]
            return predicted_emotion

    def plot_confusion_matrix(self, cm, classes,
                            normalize=False,
                            title='Confusion Matrix',
                            cmap=plt.cm.Blues):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

class cnn_vgg:
    def __init__(self, input_shape=(224, 224, 3), num_classes=7):
        self.model = self.create_model(input_shape, num_classes)

    def create_model(self, input_shape, num_classes):
        base_model = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
        base_model.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
        
        return model
    
    def train(self, train_generator, epochs=50):
        history = self.model.fit(train_generator, epochs=epochs)
        return history

    def evaluate(self, test_generator):
        test_generator.reset()
        
        num_samples = test_generator.samples
        
        predictions = self.model.predict(test_generator, steps=np.ceil(num_samples / test_generator.batch_size))
        
        predicted_class_indices = np.argmax(predictions, axis=1)
        
        true_class_indices = test_generator.classes
        
        labels = (test_generator.class_indices)
        labels = dict((v,k) for k,v in labels.items())
        predictions_labels = [labels[k] for k in predicted_class_indices]
        
        cm = confusion_matrix(true_class_indices, predicted_class_indices)
        
        print(classification_report(true_class_indices, predicted_class_indices, target_names=list(labels.values())))
        
        plt.figure(figsize=(10, 8))
        self.plot_confusion_matrix(cm, classes=list(labels.values()), title='Confusion Matrix')
        plt.show()

    def plot_confusion_matrix(self, cm, classes,
                            normalize=False,
                            title='Confusion Matrix',
                            cmap=plt.cm.Blues):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

class cnn_improved_vgg:
    def __init__(self, input_shape=(224, 224, 3), num_classes=7):
        self.model, self.callbacks = self.create_model(input_shape, num_classes)

    def create_model(self, input_shape, num_classes):
        base_model = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)

        # Unfreeze the last convolution block
        for layer in base_model.layers[:15]:
            layer.trainable = False
        for layer in base_model.layers[15:]:
            layer.trainable = True

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu', kernel_regularizer=l2(0.01))(x)  # Added L2 regularization
        x = Dropout(0.5)(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
        
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=0.00001, verbose=1)
        early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=1)
        
        return model, [reduce_lr, early_stopping]

    def train(self, train_generator, epochs=50):
        history = self.model.fit(train_generator, epochs=epochs,callbacks=self.callbacks)
        return history

    def evaluate(self, test_generator):
        test_generator.reset()
        num_samples = test_generator.samples
        predictions = self.model.predict(test_generator, steps=np.ceil(num_samples / test_generator.batch_size))
        predicted_class_indices = np.argmax(predictions, axis=1)
        true_class_indices = test_generator.classes
        labels = (test_generator.class_indices)
        labels = dict((v,k) for k,v in labels.items())
        predictions_labels = [labels[k] for k in predicted_class_indices]

        cm = confusion_matrix(true_class_indices, predicted_class_indices)
        print(classification_report(true_class_indices, predicted_class_indices, target_names=list(labels.values())))

        plt.figure(figsize=(10, 8))
        self.plot_confusion_matrix(cm, classes=list(labels.values()), title='Confusion Matrix')
        plt.show()

    def plot_confusion_matrix(self, cm, classes,
                            normalize=False,
                            title='Confusion Matrix',
                            cmap=plt.cm.Blues):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

class cnn_resnet:
    def __init__(self, input_shape=(224, 224, 3), num_classes=7):
        self.model = self.create_model(input_shape, num_classes)

    def create_model(self, input_shape, num_classes):
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        base_model.trainable = False  

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
        
        return model
    
    def train(self, train_generator, epochs=50):
        history = self.model.fit(train_generator, epochs=epochs)
        return history

    def evaluate(self, test_generator):
        test_generator.reset()
        
        num_samples = test_generator.samples
        
        predictions = self.model.predict(test_generator, steps=np.ceil(num_samples / test_generator.batch_size))
        
        predicted_class_indices = np.argmax(predictions, axis=1)
        
        true_class_indices = test_generator.classes
        
        labels = (test_generator.class_indices)
        labels = dict((v,k) for k,v in labels.items())
        predictions_labels = [labels[k] for k in predicted_class_indices]
        
        cm = confusion_matrix(true_class_indices, predicted_class_indices)
        
        print(classification_report(true_class_indices, predicted_class_indices, target_names=list(labels.values())))
        
        plt.figure(figsize=(10, 8))
        self.plot_confusion_matrix(cm, classes=list(labels.values()), title='Confusion Matrix')
        plt.show()

    def plot_confusion_matrix(self, cm, classes,
                            normalize=False,
                            title='Confusion Matrix',
                            cmap=plt.cm.Blues):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')



        