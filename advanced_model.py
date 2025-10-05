import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import warnings

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Global variables to fix the X_train issue
X_train_global = None
X_test_global = None
y_test_global = None


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


def advanced_feature_engineering(X):
    """Create advanced features to boost accuracy"""
    print("Performing advanced feature engineering...")

    # Create ratio features
    if 'planetary_radius' in X.columns and 'stellar_radius' in X.columns:
        X['radius_ratio'] = X['planetary_radius'] / X['stellar_radius']

    if 'orbital_period' in X.columns and 'stellar_mass' in X.columns:
        X['period_mass_ratio'] = X['orbital_period'] / X['stellar_mass']

    # Create interaction features
    if 'planetary_radius' in X.columns and 'equilibrium_temp' in X.columns:
        X['radius_temp_interaction'] = X['planetary_radius'] * X['equilibrium_temp']

    # Create normalized features
    for col in ['orbital_period', 'planetary_radius', 'stellar_temp']:
        if col in X.columns:
            X[f'{col}_log'] = np.log1p(np.abs(X[col]))  # Use abs to handle negative values

    print(f"Features after engineering: {X.shape[1]}")
    return X


def handle_missing_values(X):
    """Handle missing values in features"""
    print("Handling missing values...")

    # Check for missing values
    missing_percent = (X.isnull().sum() / len(X)) * 100
    print("Missing value percentage per column:")
    print(missing_percent[missing_percent > 0])

    # Combine both conditions: drop columns with 100% missing OR >80% missing
    columns_to_drop = missing_percent[(missing_percent == 100) | (missing_percent > 80)].index.tolist()

    if columns_to_drop:
        print(f"Dropping columns with high missing values: {columns_to_drop}")
        X = X.drop(columns=columns_to_drop)

    # Impute remaining missing values with median
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)

    # Save the imputer for later use
    joblib.dump(imputer, 'imputer.pkl')

    return pd.DataFrame(X_imputed, columns=X.columns), imputer


def prepare_data(X, y, use_smote=True):
    """Prepare data for training with optional SMOTE"""
    # Encode target variable
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Save the label encoder for later use
    joblib.dump(le, 'label_encoder.pkl')

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded  # Reduced test size
    )

    # Apply SMOTE for class balancing
    if use_smote:
        print("Applying SMOTE for class balancing...")
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"Training data shape after SMOTE: {X_train.shape}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save the scaler for later use
    joblib.dump(scaler, 'scaler.pkl')

    # Set global variables to fix the X_train issue
    global X_train_global, X_test_global, y_test_global
    X_train_global = X_train_scaled
    X_test_global = X_test_scaled
    y_test_global = y_test

    return X_train_scaled, X_test_scaled, y_train, y_test, le, scaler


def optimize_random_forest(trial, X_train, y_train, X_test, y_test):
    """Bayesian optimization for Random Forest"""
    n_estimators = trial.suggest_int('n_estimators', 200, 1000)
    max_depth = trial.suggest_int('max_depth', 10, 50)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    bootstrap = trial.suggest_categorical('bootstrap', [True, False])

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'  # Handle class imbalance
    )

    model.fit(X_train, y_train)
    return model.score(X_test, y_test)


def train_random_forest(X_train, X_test, y_train, y_test):
    """Train and evaluate Random Forest classifier with Bayesian optimization"""
    print("\n" + "=" * 50)
    print("Training Random Forest...")

    print("Using Bayesian optimization...")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: optimize_random_forest(trial, X_train, y_train, X_test, y_test),
                   n_trials=30, n_jobs=-1)

    best_params = study.best_params
    print(f"Best Bayesian parameters: {best_params}")

    best_rf = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1, class_weight='balanced')
    best_rf.fit(X_train, y_train)

    y_pred = best_rf.predict(X_test)
    y_pred_proba = best_rf.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Random Forest Accuracy: {accuracy:.4f}")

    return best_rf, y_pred, accuracy


def train_balanced_random_forest(X_train, X_test, y_train, y_test):
    """Train Balanced Random Forest for imbalanced data"""
    print("\n" + "=" * 50)
    print("Training Balanced Random Forest...")

    brf = BalancedRandomForestClassifier(
        n_estimators=500,
        max_depth=30,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1
    )

    brf.fit(X_train, y_train)
    y_pred = brf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Balanced Random Forest Accuracy: {accuracy:.4f}")
    return brf, y_pred, accuracy


def train_xgboost(X_train, X_test, y_train, y_test):
    """Train and evaluate XGBoost with class weights"""
    print("\n" + "=" * 50)
    print("Training XGBoost...")

    # Calculate class weights
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))

    xgb_model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=12,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=class_weights[1] if len(classes) == 2 else None
    )

    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"XGBoost Accuracy: {accuracy:.4f}")
    return xgb_model, y_pred, accuracy


def create_advanced_nn(input_dim, num_classes):
    """Create advanced neural network architecture"""
    model = keras.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(input_dim,)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),

        keras.layers.Dense(256, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.4),

        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),

        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),

        keras.layers.Dense(num_classes, activation='softmax')
    ])

    return model


def train_neural_network(X_train, X_test, y_train, y_test, num_classes):
    """Train and evaluate Neural Network"""
    print("\n" + "=" * 50)
    print("Training Neural Network...")

    model = create_advanced_nn(X_train.shape[1], num_classes)

    # Calculate class weights
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Enhanced callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=20, restore_best_weights=True, min_delta=0.001
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7
    )

    history = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, reduce_lr],
        class_weight=class_weight_dict,
        verbose=1
    )

    # Evaluate the model
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Neural Network Accuracy: {accuracy:.4f}")
    return model, y_pred, accuracy, history


def create_ensemble(models, predictions, accuracies):
    """Create weighted ensemble based on model accuracies"""
    print("\n" + "=" * 50)
    print("Creating Weighted Ensemble...")

    # Calculate weights based on accuracies
    total_accuracy = sum(accuracies.values())
    weights = {name: acc / total_accuracy for name, acc in accuracies.items()}

    print("Ensemble weights:")
    for name, weight in weights.items():
        print(f"  {name}: {weight:.3f}")

    # Create weighted predictions
    ensemble_pred_proba = None
    for name, model in models.items():
        if name == "Ensemble":
            continue
        if hasattr(model, 'predict_proba'):
            pred_proba = model.predict_proba(X_test_global)
            if ensemble_pred_proba is None:
                ensemble_pred_proba = pred_proba * weights[name]
            else:
                ensemble_pred_proba += pred_proba * weights[name]
        elif hasattr(model, 'predict'):
            # For neural networks
            pred_proba = model.predict(X_test_global)
            if ensemble_pred_proba is None:
                ensemble_pred_proba = pred_proba * weights[name]
            else:
                ensemble_pred_proba += pred_proba * weights[name]

    ensemble_pred = np.argmax(ensemble_pred_proba, axis=1)
    ensemble_accuracy = accuracy_score(y_test_global, ensemble_pred)

    print(f"Weighted Ensemble Accuracy: {ensemble_accuracy:.4f}")
    return ensemble_pred, ensemble_accuracy


def evaluate_model(model, model_name, y_test, y_pred, le):
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


def save_best_model(best_model, best_model_name, le, imputer, scaler, accuracy):
    """Save the best model and all preprocessing objects"""
    print(f"\n💾 Saving best model: {best_model_name} (Accuracy: {accuracy:.4f})")

    # Save the model
    if best_model_name == "Neural Network":
        best_model.save('best_exoplanet_classifier.h5')
        print("Best model saved as 'best_exoplanet_classifier.h5'")
    else:
        joblib.dump(best_model, 'best_exoplanet_classifier.pkl')
        print(f"Best model saved as 'best_exoplanet_classifier.pkl'")

    # Save preprocessing objects
    joblib.dump(imputer, 'imputer.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(le, 'label_encoder.pkl')

    # Save model info
    model_info = {
        'model_name': best_model_name,
        'accuracy': accuracy,
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'input_features': imputer.feature_names_in_.tolist() if hasattr(imputer, 'feature_names_in_') else 'Unknown'
    }
    joblib.dump(model_info, 'model_info.pkl')

    print("\n📁 All files saved:")
    print("- best_exoplanet_classifier.h5/.pkl")
    print("- imputer.pkl")
    print("- scaler.pkl")
    print("- label_encoder.pkl")
    print("- model_info.pkl")


def main():
    # Load and preprocess data
    X, y = load_and_preprocess_data('unified_exoplanet_data.csv')

    if len(X) == 0:
        print("No data available for training!")
        return

    # Advanced feature engineering
    X = advanced_feature_engineering(X)

    # Handle missing values
    X_imputed, imputer = handle_missing_values(X)

    print(f"Final feature set: {X_imputed.columns.tolist()}")
    print(f"Shape after preprocessing: {X_imputed.shape}")

    # Prepare data with SMOTE
    X_train, X_test, y_train, y_test, le, scaler = prepare_data(X_imputed, y, use_smote=True)

    num_classes = len(le.classes_)
    print(f"Number of classes: {num_classes}")
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")

    # Train and evaluate models
    models = {}
    predictions = {}
    accuracies = {}

    # Random Forest
    rf_model, rf_pred, rf_accuracy = train_random_forest(X_train, X_test, y_train, y_test)
    models['Random Forest'] = rf_model
    predictions['Random Forest'] = rf_pred
    accuracies['Random Forest'] = rf_accuracy
    evaluate_model(rf_model, "Random Forest", y_test, rf_pred, le)

    # Balanced Random Forest
    brf_model, brf_pred, brf_accuracy = train_balanced_random_forest(X_train, X_test, y_train, y_test)
    models['Balanced RF'] = brf_model
    predictions['Balanced RF'] = brf_pred
    accuracies['Balanced RF'] = brf_accuracy
    evaluate_model(brf_model, "Balanced RF", y_test, brf_pred, le)

    # XGBoost
    xgb_model, xgb_pred, xgb_accuracy = train_xgboost(X_train, X_test, y_train, y_test)
    models['XGBoost'] = xgb_model
    predictions['XGBoost'] = xgb_pred
    accuracies['XGBoost'] = xgb_accuracy
    evaluate_model(xgb_model, "XGBoost", y_test, xgb_pred, le)

    # Neural Network
    nn_model, nn_pred, nn_accuracy, history = train_neural_network(X_train, X_test, y_train, y_test, num_classes)
    models['Neural Network'] = nn_model
    predictions['Neural Network'] = nn_pred
    accuracies['Neural Network'] = nn_accuracy
    evaluate_model(nn_model, "Neural Network", y_test, nn_pred, le)

    # Create ensemble
    ensemble_pred, ensemble_accuracy = create_ensemble(models, predictions, accuracies)
    accuracies['Ensemble'] = ensemble_accuracy

    # Create a dummy model for ensemble evaluation
    class EnsembleModel:
        def __init__(self, models, weights):
            self.models = models
            self.weights = weights

        def predict(self, X):
            ensemble_pred_proba = None
            for name, model in self.models.items():
                if name == "Ensemble":
                    continue
                if hasattr(model, 'predict_proba'):
                    pred_proba = model.predict_proba(X)
                else:
                    pred_proba = model.predict(X)

                if ensemble_pred_proba is None:
                    ensemble_pred_proba = pred_proba * self.weights[name]
                else:
                    ensemble_pred_proba += pred_proba * self.weights[name]

            return np.argmax(ensemble_pred_proba, axis=1)

    # Calculate weights for ensemble model
    total_accuracy = sum(accuracies.values()) - ensemble_accuracy  # Exclude ensemble itself
    weights = {name: acc / total_accuracy for name, acc in accuracies.items() if name != "Ensemble"}

    ensemble_model = EnsembleModel(models, weights)
    evaluate_model(ensemble_model, "Weighted Ensemble", y_test, ensemble_pred, le)

    # Compare all models
    print("\n" + "=" * 50)
    print("FINAL MODEL COMPARISON")
    print("=" * 50)
    for model_name, accuracy in accuracies.items():
        print(f"{model_name}: {accuracy:.4f}")

    # Select best model based on accuracy
    best_model_name = max(accuracies.keys(), key=lambda x: accuracies[x])
    best_accuracy = accuracies[best_model_name]

    if best_model_name == "Ensemble":
        # For ensemble, we don't have a single model object, so save the best individual model
        best_individual_name = max([k for k in accuracies.keys() if k != "Ensemble"],
                                   key=lambda x: accuracies[x])
        best_model = models[best_individual_name]
        print(f"\n🏆 BEST MODEL: {best_individual_name} (Accuracy: {accuracies[best_individual_name]:.4f})")
        save_best_model(best_model, best_individual_name, le, imputer, scaler, accuracies[best_individual_name])
    else:
        best_model = models[best_model_name]
        print(f"\n🏆 BEST MODEL: {best_model_name} (Accuracy: {best_accuracy:.4f})")
        save_best_model(best_model, best_model_name, le, imputer, scaler, best_accuracy)

    print(f"\n🎯 Target: 90%+ Accuracy")
    print(f"📈 Current Best: {best_accuracy:.2%}")
    print(f"📊 Gap to Target: {(0.90 - best_accuracy):.2%}")

    if best_accuracy < 0.85:
        print("\n💡 Suggestions to reach 90%+:")
        print("1. Collect more training data")
        print("2. Add more domain-specific features")
        print("3. Try advanced architectures (Transformers, AutoML)")
        print("4. Feature selection to remove noise")
        print("5. Hyperparameter tuning with more trials")


if __name__ == "__main__":
    main()