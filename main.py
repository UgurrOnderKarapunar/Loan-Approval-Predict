import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
from xgboost import XGBClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_seq_item', None)
pd.set_option('display.float_format', '{:.2f}'.format)

test = pd.read_csv(r"C:\Users\ugrkr\OneDrive\Masaüstü\playground-series-s4e10\test.csv")
train = pd.read_csv(r"C:\Users\ugrkr\OneDrive\Masaüstü\playground-series-s4e10\train.csv")
sub = pd.read_csv(r"C:\Users\ugrkr\OneDrive\Masaüstü\playground-series-s4e10\sample_submission.csv")


def con_cat(train, test):
    df1, df2 = train.copy(), test.copy()
    df1["group"] = "train"
    df2["group"] = "test"
    return pd.concat([df1, df2], axis=0, ignore_index=True)


df = con_cat(train, test)
df.head()





def fillna(dataframe):
    cat = dataframe.select_dtypes(include="object").columns
    num = dataframe.select_dtypes(include=["float", "int"]).columns

    for col in cat:
        dataframe[col] = dataframe[col].fillna(dataframe[col].mode()[0])

    for col in num:
        dataframe[col] = dataframe[col].fillna(dataframe[col].median())

    return dataframe

df = fillna(df)





def calculate_correlations(dataframe):
    num_columns = dataframe.select_dtypes(include=["float", "int"]).columns
    correlation_matrix = dataframe[num_columns].corr()
    return correlation_matrix

correlation_matrix = calculate_correlations(df)

print(correlation_matrix)








import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlation_heatmap(correlation_matrix):
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title("Korelasyon Matrisi Isı Haritası")
    plt.show()

plot_correlation_heatmap(correlation_matrix)







import matplotlib.pyplot as plt
import seaborn as sns


def visualize_numeric(df):
    num_columns = df.select_dtypes(include=["float", "int"]).columns

    for col in num_columns:
        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        sns.histplot(df[col], kde=True, bins=20, color='blue')
        plt.title(f'{col} Dağılımı (Histogram)')

        plt.subplot(1, 2, 2)
        sns.boxplot(x=df[col], color='green')
        plt.title(f'{col} Yayılımı (Boxplot)')

        plt.tight_layout()
        plt.show()


visualize_numeric(df)

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, roc_auc_score, roc_curve, accuracy_score, f1_score, recall_score, \
    precision_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

df.drop("cb_person_cred_hist_length", axis=1, inplace=True)


# Define the model-building function
def build_best_model(input_dim):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))

    # Layer 1
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # Layer 2
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    # Layer 3
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    # Layer 4
    model.add(Dense(96, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    # Layer 5
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())

    # Output layer
    model.add(Dense(1, activation="sigmoid"))

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.00025),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def modelling_dl(dataframe, target):
    # Drop the specified column

    train = dataframe[dataframe["group"] == "train"].drop("group", axis=1)
    test = dataframe[dataframe["group"] == "test"].drop("group", axis=1)

    X_train = train.drop(target, axis=1)
    y_train = train[target]

    y_test = test[target] if target in test.columns else None
    X_test = test.drop(target, axis=1) if y_test is not None else test

    categorical_c = X_train.select_dtypes(include="object").columns.tolist()
    numerical_c = X_train.select_dtypes(include=["float", "int"]).columns.tolist()

    print(f"Categorical Cols: {categorical_c}")
    print(f"Numerical Cols: {numerical_c}")

    doms_list = []
    for col in categorical_c:
        doms = X_train[col].value_counts(normalize=True)
        if doms.max() > 0.90:
            doms_list.append(col)

    print(f"Dominant Cols List: {doms_list}")

    categorical_c = [col for col in categorical_c if col not in doms_list]
    print(f"New Categorical Cols: {categorical_c}")

    print("******* Pipeline Process ***** ")

    categorical_pipeline = Pipeline(steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

    numerical_pipeline = Pipeline(steps=[
        ("scaler", RobustScaler()),
        ("imputer", SimpleImputer(strategy="mean"))
    ])

    preprocess = ColumnTransformer(transformers=[
        ("num", numerical_pipeline, numerical_c),
        ("cat", categorical_pipeline, categorical_c)
    ])

    X_train_processed = preprocess.fit_transform(X_train)
    X_test_processed = preprocess.transform(X_test)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_processed, y_train)).batch(64).shuffle(buffer_size=1024)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test_processed, y_test)).batch(
        64) if y_test is not None else None

    print("******* Building and Training the Model *****")

    model = build_best_model(X_train_processed.shape[1])

    early_stopping = EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)
    checkpoint = ModelCheckpoint("best_model.h5.keras", monitor='val_loss', save_best_only=True, verbose=1)

    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=1000,
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )

    print("Model training completed.")

    print("***** Evaluating the Model *****")
    y_pred_proba = model.predict(X_test_processed).flatten()

    roc_auc = None
    if y_test is not None:
        print(f"Unique values in y_test: {np.unique(y_test)}")
        if len(np.unique(y_test)) > 1:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            print(f"ROC AUC: {roc_auc:.2f}")

            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
            plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc='lower right')
            plt.show()
        else:
            print("y_test contains only one class. ROC AUC cannot be computed.")
            roc_auc = 0.0

        y_pred = (y_pred_proba > 0.5).astype("int32")

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)

        print(f"Accuracy: {accuracy:.2f}")
        print(f"F1 Score: {f1:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"ROC AUC Score: {roc_auc:.2f}" if roc_auc is not None else "ROC AUC Score: Not computable")

    mse = mean_squared_error(y_test, y_pred_proba)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test - y_pred_proba))

    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")

    def plot_history(history):
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    plot_history(history)

    joblib.dump(model, 'best_model.joblib.keras')
    print("Model saved as 'best_model.joblib'.")

    submission = pd.DataFrame({
        "id": test["id"],
        "loan_status": y_pred_proba
    })

    submission_file_path = "submissiondatas2.csv"
    submission.to_csv(submission_file_path, index=False)

    print(f"Submission file saved as {submission_file_path}.")
    return X_train, y_train, categorical_c, numerical_c, y_test, y_pred_proba, model, history


X_train, y_train, categorical_c, numerical_c, y_test, y_pred, model, history = modelling_dl(df, "loan_status")
