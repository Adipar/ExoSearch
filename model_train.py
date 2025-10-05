import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


def load_and_preprocess_data(file_path):
    """Load and preprocess the merged exoplanet dataset"""
    print("Loading dataset...")
    df = pd.read_csv(file_path)

    print(f"Initial dataset shape: {df.shape}")
    print("Target class distribution:")
    print(df['target_class'].value_counts())

    # Remove any remaining unknown classes
    df = df[df['target_class'].isin(['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE'])]

    # Separate features and target
    X = df.drop('target_class', axis=1)
    y = df['target_class']

    # Remove non-numeric columns and data_source for modeling
    X = X.select_dtypes(include=[np.number])

    print(f"Features after preprocessing: {X.shape[1]}")
    return X, y


def handle_missing_values(X):
    """Handle missing values in features"""
    print("Handling missing values...")

    # Check for missing values
    missing_percent = (X.isnull().sum() / len(X)) * 100
    print("Missing value percentage per column:")
    print(missing_percent[missing_percent > 0])

    # Remove columns with 100% missing values
    columns_to_drop = missing_percent[missing_percent == 100].index.tolist()
    if columns_to_drop:
        print(f"Dropping columns with 100% missing values: {columns_to_drop}")
        X = X.drop(columns=columns_to_drop)

    # Impute remaining missing values with median
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)

    # Save the imputer for later use
    joblib.dump(imputer, 'imputer.pkl')

    return pd.DataFrame(X_imputed, columns=X.columns), imputer


def prepare_data(X, y):
    """Prepare data for training"""
    # Encode target variable
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Save the label encoder for later use
    joblib.dump(le, 'label_encoder.pkl')

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save the scaler for later use
    joblib.dump(scaler, 'scaler.pkl')

    return X_train_scaled, X_test_scaled, y_train, y_test, le, scaler


def train_random_forest(X_train, X_test, y_train, y_test):
    """Train and evaluate Random Forest classifier"""
    print("\n" + "=" * 50)
    print("Training Random Forest...")

    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_rf = grid_search.best_estimator_
    y_pred = best_rf.predict(X_test)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    return best_rf, y_pred


def train_xgboost(X_train, X_test, y_train, y_test):
    """Train and evaluate XGBoost classifier"""
    print("\n" + "=" * 50)
    print("Training XGBoost...")

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }

    xgb_model = xgb.XGBClassifier(random_state=42)
    grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_xgb = grid_search.best_estimator_
    y_pred = best_xgb.predict(X_test)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"XGBoost Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    return best_xgb, y_pred


def train_neural_network(X_train, X_test, y_train, y_test, num_classes):
    """Train and evaluate Neural Network"""
    print("\n" + "=" * 50)
    print("Training Neural Network...")

    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Add early stopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )

    # Evaluate the model
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)

    print(f"Neural Network Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    return model, y_pred, history


def evaluate_model(model, model_name, y_test, y_pred, le, X_test=None):
    """Comprehensive model evaluation"""
    print(f"\n{model_name} Evaluation:")
    print("=" * 30)

    # Classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
    plt.show()

    # Cross-validation score (for non-NN models)
    if model_name != "Neural Network" and X_test is not None:
        cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring='accuracy')
        print(f"Cross-Validation Scores: {cv_scores}")
        print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")


def plot_training_history(history):
    """Plot training history for neural network"""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('neural_network_training_history.png')
    plt.show()


def main():
    # Load and preprocess data
    X, y = load_and_preprocess_data('unified_exoplanet_data.csv')

    if len(X) == 0:
        print("No data available for training!")
        return

    # Handle missing values
    X_imputed, imputer = handle_missing_values(X)

    print(f"Shape after handling missing values: {X_imputed.shape}")
    print(f"Columns after handling missing values: {X_imputed.columns.tolist()}")

    # Prepare data for training
    X_train, X_test, y_train, y_test, le, scaler = prepare_data(X_imputed, y)

    num_classes = len(le.classes_)
    print(f"Number of classes: {num_classes}")
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")

    # Train and evaluate models
    models = {}
    predictions = {}

    # Random Forest
    rf_model, rf_pred = train_random_forest(X_train, X_test, y_train, y_test)
    models['Random Forest'] = rf_model
    predictions['Random Forest'] = rf_pred
    evaluate_model(rf_model, "Random Forest", y_test, rf_pred, le, np.vstack([X_train, X_test]))

    # XGBoost
    xgb_model, xgb_pred = train_xgboost(X_train, X_test, y_train, y_test)
    models['XGBoost'] = xgb_model
    predictions['XGBoost'] = xgb_pred
    evaluate_model(xgb_model, "XGBoost", y_test, xgb_pred, le, np.vstack([X_train, X_test]))

    # Neural Network
    nn_model, nn_pred, history = train_neural_network(X_train, X_test, y_train, y_test, num_classes)
    models['Neural Network'] = nn_model
    predictions['Neural Network'] = nn_pred
    evaluate_model(nn_model, "Neural Network", y_test, nn_pred, le)
    plot_training_history(history)

    # Compare all models
    print("\n" + "=" * 50)
    print("MODEL COMPARISON")
    print("=" * 50)
    for model_name, y_pred in predictions.items():
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{model_name}: {accuracy:.4f}")

    # Select best model based on accuracy
    best_model_name = max(predictions.keys(),
                          key=lambda x: accuracy_score(y_test, predictions[x]))
    best_model = models[best_model_name]

    print(f"\nBest model: {best_model_name}")

    # Save the best model
    if best_model_name == "Neural Network":
        best_model.save('exoplanet_classifier.h5')
        print("Neural Network saved as 'exoplanet_classifier.h5'")
    else:
        joblib.dump(best_model, 'exoplanet_classifier.pkl')
        print(f"{best_model_name} saved as 'exoplanet_classifier.pkl'")

    # Save preprocessing objects
    joblib.dump(imputer, 'imputer.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(le, 'label_encoder.pkl')

    print("\nPreprocessing objects saved:")
    print("- imputer.pkl")
    print("- scaler.pkl")
    print("- label_encoder.pkl")

    # Feature importance for tree-based models
    if best_model_name in ['Random Forest', 'XGBoost']:
        plt.figure(figsize=(10, 8))
        feature_importance = best_model.feature_importances_
        feature_names = X_imputed.columns

        # Create DataFrame for feature importance
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)

        # Plot top features
        top_features = importance_df.head(min(15, len(importance_df)))
        plt.barh(top_features['feature'], top_features['importance'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top Feature Importance - {best_model_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.show()


if __name__ == "__main__":
    main()