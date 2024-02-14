import matplotlib.pyplot as plt
import numpy as np
import os

import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


model_name = "model_resnet101_lr3"
BATCH_SIZE = 256
IMG_SIZE = (224, 224)
test_flag = True
train_flag = False

data_path = "/workspace/DBT_US_Soroka/semi-supervised_data/all_data_2804/labeled_data"

# ------------------------------------ prepare data --------------------------------
with tf.device('/device:GPU:0'):
    train_dir = os.path.join(data_path, 'Train')

    train_datagen = ImageDataGenerator(rotation_range=30,
                                 horizontal_flip=True,
                                 vertical_flip=True)
    train_dataset = train_datagen.flow_from_directory(
            train_dir,
            batch_size=BATCH_SIZE,
            class_mode='binary',
            shuffle=True)


    validation_dir = os.path.join(data_path, 'Val')
    val_datagen = ImageDataGenerator()
    validation_dataset = val_datagen.flow_from_directory(validation_dir,
                                                         batch_size=BATCH_SIZE,
                                                         class_mode='binary',
                                                         shuffle=True)

    AUTOTUNE = tf.data.AUTOTUNE
    #train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    #validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)


    IMG_SHAPE = IMG_SIZE + (3,)
    base_model = tf.keras.applications.ResNet101(input_shape=IMG_SHAPE,
                                                      include_top=False,
                                                      weights='imagenet')
    base_model.trainable = True
    base_model.summary()

    #for layer in base_model.layers[:-100]:
    #    layer.trainable = False

    # Add classification head
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(1)

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = preprocess_input(inputs)
    x = base_model(x)
    x = global_average_layer(x)
    #x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    # compiling
    base_learning_rate = 0.01
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(base_learning_rate, decay_steps=100000, decay_rate=0.96,
                                                                 staircase=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.summary()

    if train_flag:
        callback_es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
        callback_save = ModelCheckpoint(
            filepath=model_name+".h5",
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        )
        history = model.fit(train_dataset,
                            epochs=100,
                            validation_data=validation_dataset,
                            callbacks=[callback_es, callback_save])

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        plt.ylim([min(plt.ylim()), 1])
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel('Cross Entropy')
        plt.ylim([0, 1.0])
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        plt.savefig(model_name+'.jpg')
        plt.close()
        plt.clf()

    # Test
    if test_flag:

        test_dir = os.path.join(data_path, 'Test')
        test_dataset = tf.keras.utils.image_dataset_from_directory(test_dir,
                                                                   shuffle=False,
                                                                   batch_size=BATCH_SIZE,
                                                                   image_size=IMG_SIZE)

        model.load_weights(model_name+'.h5')
        loss, accuracy = model.evaluate(test_dataset)
        print('Accuracy - ' ,accuracy)

        # Retrieve a batch of images from the test set
        #x = np.concatenate([x for x, y in test_dataset], axis=0)
        y = np.concatenate([y for x1, y in test_dataset], axis=0)
        y = np.reshape(y, (len(y),))
        # Apply a sigmoid since our model returns logits
        predictions = model.predict(test_dataset)
        probabilites = tf.nn.sigmoid(predictions)
        probabilites = np.reshape(probabilites, (len(probabilites),))
        predictions = tf.where(probabilites < 0.5, 0, 1)

        file_names = pd.Series(os.listdir(os.path.join(test_dir,'Negative'))+os.listdir(os.path.join(test_dir,'Positive')))
        probs = pd.Series(probabilites)
        true = pd.Series(y)

        # get the subjects and view list
        uniqe_filename = []
        for name in file_names:
            if name[:12] not in uniqe_filename:
                uniqe_filename.append(name[:12])

        probabilities = []
        true_labels = []
        ten_slice_prob = []
        for name in uniqe_filename:
            res = file_names.str.contains(pat=name)
            relevant_probs = probs[res.values]
            y_true = true[res.values]
            true_labels.append(np.mean(y_true))

            maximum_prob = 0
            for idx in range(len(relevant_probs - 8)):
                current = np.mean(relevant_probs[idx:idx + 8])
                if current > maximum_prob:
                    maximum_prob = current

            ten_slice_prob.append(maximum_prob)

        ten_slice_prob = np.asarray(ten_slice_prob).reshape(len(ten_slice_prob), )
        true_labels = np.asarray(true_labels).reshape(len(true_labels), )
        ten_slice_prediction = np.where(ten_slice_prob < 0.5, 0, 1).reshape(len(ten_slice_prob), )

        results = pd.DataFrame({"Filename": uniqe_filename,
                                "true_labels": true_labels,
                                "Predictions": ten_slice_prediction,
                                "Probabilities": ten_slice_prob})
        results.to_csv("results_test_scan-based.csv", index=False)

        # Metrics
        from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score

        # Accuracy
        predictions = ten_slice_prediction
        probabilities = ten_slice_prob
        acc = accuracy_score(true_labels, predictions)
        print('Scan-based Accuracy: ', acc)

        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        print(cm)

        # sensitivity and specificity
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (fn + tp)
        spec = tn / (tn + fp)
        print('Scan-based sensitivity - ', sens)
        print('Scan-based Specificity - ', spec)

        # AUC
        auc = roc_auc_score(true_labels, probabilities, average=None)
        print('Scan-based AUC - ', auc)

        from sklearn.metrics import roc_curve
        import matplotlib.pyplot as plt

        fpr, tpr, thresholds = roc_curve(true_labels, probabilities)
        plt.plot(fpr, tpr, label="AUC = " + str(round(auc, 4)))
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.legend(loc=4)
        plt.savefig('scan_based_ROC.jpg')
        plt.close()
        plt.clf()

        # subject based
        df = pd.read_csv("results_test_scan-based.csv")

        file_names = df["Filename"]

        # get the subjects and view list
        uniqe_filename = []
        for name in file_names:
            if name[:8] not in uniqe_filename:
                uniqe_filename.append(name[:8])

        probabilities = []
        true_labels = []
        file_names = df["Filename"]
        for name in uniqe_filename:
            res = file_names.str.contains(pat=name)
            relevant_probs = df["Probabilities"][res.values]
            y_true = df["true_labels"][res.values]
            # probabilities.append(np.mean(relevant_probs))
            probabilities.append(np.mean(relevant_probs))
            true_labels.append(np.mean(y_true))
        probabilities = np.asarray(probabilities).reshape(len(probabilities), )
        true_labels = np.asarray(true_labels).reshape(len(true_labels), )
        predictions = np.where(probabilities < 0.5, 0, 1).reshape(len(probabilities), )

        acc = accuracy_score(true_labels, predictions)
        print('Case-based Accuracy: ', acc)

        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        print(cm)

        # sensitivity and specificity
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (fn + tp)
        spec = tn / (tn + fp)
        print('Case-based Sensitivity - ', sens)
        print('Case-based Specificity - ', spec)

        # AUC
        auc = roc_auc_score(true_labels, probabilities, average=None)
        print('Case-based AUC - ', auc)

        fpr, tpr, thresholds = roc_curve(true_labels, probabilities)
        plt.plot(fpr, tpr, label="AUC = " + str(round(auc, 4)))
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.legend(loc=4)
        plt.savefig('scan_based_ROC.jpg')
        plt.close()
        plt.clf()

