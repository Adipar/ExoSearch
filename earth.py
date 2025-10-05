import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
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
from scipy.spatial.distance import cosine
import warnings

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Earth reference data - the habitability benchmark
EARTH_REFERENCE = {
    'target_class': 'EARTH_REFERENCE',
    'orbital_period': 365.25,  # Earth days
    'planetary_radius': 1.0,  # Earth radii
    'insolation_flux': 1.0,  # Earth fluxes
    'equilibrium_temp': 255,  # Kelvin (Earth's equilibrium temp)
    'stellar_temp': 5778,  # Kelvin (Sun's temperature)
    'stellar_radius': 1.0,  # Solar radii
    'stellar_mass': 1.0,  # Solar masses
    'distance': 0,  # Parsecs (not used in similarity)
    'transit_duration': None,
    'transit_depth': None,
    'data_source': 'SOLAR_SYSTEM'
}


class EarthSimilarityCalculator:
    """Calculate Earth Similarity Index (ESI) for exoplanets"""

    def __init__(self, earth_reference=EARTH_REFERENCE):
        self.earth = earth_reference
        # Weights for different parameters (based on importance for habitability)
        self.weights = {
            'planetary_radius': 0.3,  # Most important - size affects gravity, atmosphere
            'insolation_flux': 0.25,  # Critical for liquid water
            'equilibrium_temp': 0.2,  # Temperature range for life
            'orbital_period': 0.1,  # Affects seasons and climate stability
            'stellar_temp': 0.1,  # Star type affects UV radiation
            'stellar_radius': 0.05  # Less critical but relevant
        }

    def calculate_similarity_score(self, planet_data):
        """Calculate Earth Similarity Index (0-100%)"""
        try:
            similarities = {}

            # Planetary radius similarity (log scale - more important for smaller differences)
            if 'planetary_radius' in planet_data and pd.notna(planet_data['planetary_radius']):
                earth_radius = self.earth['planetary_radius']
                planet_radius = planet_data['planetary_radius']
                # Avoid division by zero and handle edge cases
                if planet_radius > 0 and earth_radius > 0:
                    radius_sim = 1 - abs(np.log(planet_radius / earth_radius)) / 10
                    radius_sim = max(0, min(1, radius_sim))
                    similarities['planetary_radius'] = radius_sim

            # Insolation flux similarity (critical for habitable zone) - FIXED TYPO
            if 'insolation_flux' in planet_data and pd.notna(planet_data['insolation_flux']):
                earth_flux = self.earth['insolation_flux']
                planet_flux = planet_data['insolation_flux']
                # Avoid division by zero and handle edge cases
                if planet_flux > 0 and earth_flux > 0:
                    flux_sim = 1 - abs(np.log(planet_flux / earth_flux)) / 5
                    flux_sim = max(0, min(1, flux_sim))
                    similarities['insolation_flux'] = flux_sim

            # Temperature similarity (optimal range for liquid water)
            if 'equilibrium_temp' in planet_data and pd.notna(planet_data['equilibrium_temp']):
                earth_temp = self.earth['equilibrium_temp']
                planet_temp = planet_data['equilibrium_temp']
                temp_diff = abs(planet_temp - earth_temp)
                temp_sim = max(0, 1 - temp_diff / 100)  # 100K tolerance
                similarities['equilibrium_temp'] = temp_sim

            # Orbital period similarity (affects climate stability)
            if 'orbital_period' in planet_data and pd.notna(planet_data['orbital_period']):
                earth_period = self.earth['orbital_period']
                planet_period = planet_data['orbital_period']
                # Avoid division by zero and handle edge cases
                if planet_period > 0 and earth_period > 0:
                    period_sim = 1 - abs(np.log(planet_period / earth_period)) / 20
                    period_sim = max(0, min(1, period_sim))
                    similarities['orbital_period'] = period_sim

            # Stellar temperature similarity (affects UV and stellar lifetime)
            if 'stellar_temp' in planet_data and pd.notna(planet_data['stellar_temp']):
                earth_stellar_temp = self.earth['stellar_temp']
                planet_stellar_temp = planet_data['stellar_temp']
                stellar_temp_sim = 1 - abs(planet_stellar_temp - earth_stellar_temp) / 2000
                stellar_temp_sim = max(0, min(1, stellar_temp_sim))
                similarities['stellar_temp'] = stellar_temp_sim

            # Stellar radius similarity
            if 'stellar_radius' in planet_data and pd.notna(planet_data['stellar_radius']):
                earth_stellar_radius = self.earth['stellar_radius']
                planet_stellar_radius = planet_data['stellar_radius']
                # Avoid division by zero and handle edge cases
                if planet_stellar_radius > 0 and earth_stellar_radius > 0:
                    stellar_radius_sim = 1 - abs(np.log(planet_stellar_radius / earth_stellar_radius)) / 5
                    stellar_radius_sim = max(0, min(1, stellar_radius_sim))
                    similarities['stellar_radius'] = stellar_radius_sim

            # Calculate weighted average similarity score
            total_weight = 0
            weighted_sum = 0

            for feature, similarity in similarities.items():
                weight = self.weights.get(feature, 0.1)
                weighted_sum += similarity * weight
                total_weight += weight

            if total_weight > 0:
                earth_similarity = (weighted_sum / total_weight) * 100
            else:
                earth_similarity = 0

            return min(100, max(0, earth_similarity)), similarities

        except Exception as e:
            print(f"Error calculating similarity for planet: {e}")
            return 0, {}


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
    return X, y, df


def calculate_earth_similarity_for_dataset(df):
    """Calculate Earth Similarity Index for entire dataset"""
    print("Calculating Earth Similarity Index for all planets...")

    similarity_calculator = EarthSimilarityCalculator()
    earth_similarity_scores = []
    similarity_breakdowns = []

    successful_calculations = 0
    failed_calculations = 0

    for idx, row in df.iterrows():
        planet_data = row.to_dict()
        similarity_score, breakdown = similarity_calculator.calculate_similarity_score(planet_data)
        earth_similarity_scores.append(similarity_score)
        similarity_breakdowns.append(breakdown)

        if similarity_score > 0:
            successful_calculations += 1
        else:
            failed_calculations += 1

    df['earth_similarity_score'] = earth_similarity_scores
    df['similarity_breakdown'] = similarity_breakdowns

    # Add Earth similarity category
    conditions = [
        df['earth_similarity_score'] >= 80,
        df['earth_similarity_score'] >= 60,
        df['earth_similarity_score'] >= 40,
        df['earth_similarity_score'] >= 20
    ]
    choices = ['Very High', 'High', 'Moderate', 'Low']
    df['earth_similarity_category'] = np.select(conditions, choices, default='Very Low')

    print(f"Earth similarity scores calculated successfully!")
    print(f"Successful calculations: {successful_calculations}")
    print(f"Failed calculations: {failed_calculations}")
    print(f"Similarity score distribution:\n{df['earth_similarity_category'].value_counts()}")

    # Print some statistics about the similarity scores
    if successful_calculations > 0:
        print(f"Average Earth similarity: {df['earth_similarity_score'].mean():.2f}%")
        print(f"Maximum Earth similarity: {df['earth_similarity_score'].max():.2f}%")
        print(f"Minimum Earth similarity: {df[df['earth_similarity_score'] > 0]['earth_similarity_score'].min():.2f}%")

    return df


def enhanced_feature_engineering(X, df_with_similarity):
    """Create enhanced features including Earth similarity"""
    print("Performing enhanced feature engineering with Earth similarity...")

    # Add Earth similarity score as a feature
    X = X.copy()
    X['earth_similarity_score'] = df_with_similarity['earth_similarity_score']

    # Create ratio features
    if 'planetary_radius' in X.columns and 'stellar_radius' in X.columns:
        # Avoid division by zero
        X['radius_ratio'] = np.where(X['stellar_radius'] != 0,
                                     X['planetary_radius'] / X['stellar_radius'], 0)

    if 'orbital_period' in X.columns and 'stellar_mass' in X.columns:
        # Avoid division by zero
        X['period_mass_ratio'] = np.where(X['stellar_mass'] != 0,
                                          X['orbital_period'] / X['stellar_mass'], 0)

    # Create interaction features
    if 'planetary_radius' in X.columns and 'equilibrium_temp' in X.columns:
        X['radius_temp_interaction'] = X['planetary_radius'] * X['equilibrium_temp']

    # Create habitability-focused features
    if 'insolation_flux' in X.columns:
        X['in_habitable_zone'] = (X['insolation_flux'] >= 0.5) & (X['insolation_flux'] <= 2.0)
        X['optimal_insolation'] = np.exp(-((X['insolation_flux'] - 1.0) ** 2) / 0.5)

    if 'planetary_radius' in X.columns:
        X['earth_like_size'] = np.exp(-((X['planetary_radius'] - 1.0) ** 2) / 0.5)

    # Create normalized features (handle zeros and negative values)
    for col in ['orbital_period', 'planetary_radius', 'stellar_temp']:
        if col in X.columns:
            # Use absolute value and add small epsilon to avoid log(0)
            X[f'{col}_log'] = np.log1p(np.abs(X[col]) + 1e-10)

    print(f"Features after enhanced engineering: {X.shape[1]}")
    return X


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
        X, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save the scaler for later use
    joblib.dump(scaler, 'scaler.pkl')

    return X_train_scaled, X_test_scaled, y_train, y_test, le, scaler


def train_model_with_similarity(X_train, X_test, y_train, y_test):
    """Train model with Earth similarity features"""
    print("\n" + "=" * 50)
    print("Training Model with Earth Similarity Features...")

    # Use XGBoost for better performance
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=12,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Model Accuracy with Earth Similarity: {accuracy:.4f}")
    return model, y_pred, accuracy


def visualize_earth_similarity(df_with_similarity):
    """Create visualizations for Earth similarity analysis"""
    print("\nCreating Earth Similarity Visualizations...")

    # Filter out planets with 0 similarity score for better visualization
    df_filtered = df_with_similarity[df_with_similarity['earth_similarity_score'] > 0]

    if len(df_filtered) == 0:
        print("No planets with valid Earth similarity scores to visualize.")
        return pd.DataFrame()

    # 1. Top 10 most Earth-like planets
    top_earth_like = df_filtered.nlargest(10, 'earth_similarity_score')[['earth_similarity_score', 'target_class']]

    plt.figure(figsize=(15, 10))

    # Plot 1: Top Earth-like planets
    plt.subplot(2, 2, 1)
    plt.barh(range(len(top_earth_like)), top_earth_like['earth_similarity_score'])
    plt.yticks(range(len(top_earth_like)), [f'Planet {i + 1}' for i in range(len(top_earth_like))])
    plt.xlabel('Earth Similarity Score (%)')
    plt.title('Top 10 Most Earth-Like Planets')
    plt.gca().invert_yaxis()

    # Plot 2: Similarity distribution by class
    plt.subplot(2, 2, 2)
    similarity_by_class = df_filtered.groupby('target_class')['earth_similarity_score'].mean()
    similarity_by_class.plot(kind='bar', color=['green', 'orange', 'red'])
    plt.title('Average Earth Similarity by Planet Class')
    plt.ylabel('Average Similarity Score (%)')
    plt.xticks(rotation=45)

    # Plot 3: Similarity score distribution
    plt.subplot(2, 2, 3)
    plt.hist(df_filtered['earth_similarity_score'], bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Earth Similarity Score (%)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Earth Similarity Scores')

    # Plot 4: Habitability potential
    plt.subplot(2, 2, 4)
    habitable_potential = df_filtered[df_filtered['earth_similarity_score'] > 50]
    if len(habitable_potential) > 0:
        habitable_by_class = habitable_potential['target_class'].value_counts()
        plt.pie(habitable_by_class.values, labels=habitable_by_class.index, autopct='%1.1f%%')
        plt.title('Potentially Habitable Planets\n(Similarity > 50%)')
    else:
        plt.text(0.5, 0.5, 'No planets with\nsimilarity > 50%',
                 ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Potentially Habitable Planets')

    plt.tight_layout()
    plt.savefig('earth_similarity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Create detailed table of top candidates
    print("\n" + "=" * 50)
    print("TOP EARTH-LIKE EXOPLANET CANDIDATES")
    print("=" * 50)

    # Get top 15 most Earth-like confirmed planets
    top_candidates = df_filtered[
        df_filtered['target_class'] == 'CONFIRMED'
        ].nlargest(15, 'earth_similarity_score')[
        ['earth_similarity_score', 'earth_similarity_category',
         'planetary_radius', 'insolation_flux', 'equilibrium_temp']
    ]

    if len(top_candidates) > 0:
        print(top_candidates.round(2))
    else:
        print("No confirmed planets with valid Earth similarity scores.")

    return top_candidates


def create_habitability_report(df_with_similarity, model, le, imputer, scaler):
    """Create comprehensive habitability report"""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE HABITABILITY ANALYSIS REPORT")
    print("=" * 60)

    # Filter out planets with 0 similarity
    df_filtered = df_with_similarity[df_with_similarity['earth_similarity_score'] > 0]

    if len(df_filtered) == 0:
        print("No planets with valid Earth similarity scores to analyze.")
        return pd.DataFrame()

    # Overall statistics
    total_planets = len(df_filtered)
    high_similarity = len(df_filtered[df_filtered['earth_similarity_score'] >= 70])
    moderate_similarity = len(df_filtered[df_filtered['earth_similarity_score'] >= 50])

    print(f"Total planets analyzed: {total_planets}")
    print(
        f"Planets with high Earth similarity (≥70%): {high_similarity} ({high_similarity / total_planets * 100:.1f}%)")
    print(
        f"Planets with moderate Earth similarity (≥50%): {moderate_similarity} ({moderate_similarity / total_planets * 100:.1f}%)")

    # Best candidates
    best_candidates = df_filtered[
        (df_filtered['target_class'] == 'CONFIRMED') &
        (df_filtered['earth_similarity_score'] >= 60)
        ].nlargest(10, 'earth_similarity_score')

    if len(best_candidates) > 0:
        print(f"\n🏆 TOP HABITABILITY CANDIDATES (Confirmed Exoplanets):")
        print("-" * 50)
        for idx, (_, planet) in enumerate(best_candidates.iterrows(), 1):
            similarity = planet['earth_similarity_score']
            category = planet['earth_similarity_category']
            print(f"{idx:2d}. Similarity: {similarity:5.1f}% ({category})")
    else:
        print("\nNo confirmed planets with high Earth similarity found.")

    # Save the model and preprocessing objects
    joblib.dump(model, 'exoplanet_habitability_model.pkl')
    print(f"\n💾 Model saved as 'exoplanet_habitability_model.pkl'")

    # Save the Earth similarity calculator
    similarity_calculator = EarthSimilarityCalculator()
    joblib.dump(similarity_calculator, 'earth_similarity_calculator.pkl')
    print(f"💾 Earth similarity calculator saved as 'earth_similarity_calculator.pkl'")

    # Save the enhanced dataset
    df_with_similarity.to_csv('exoplanets_with_earth_similarity.csv', index=False)
    print(f"💾 Enhanced dataset saved as 'exoplanets_with_earth_similarity.csv'")

    return best_candidates


def prepare_single_planet_for_prediction(planet_features, imputer, scaler, similarity_calculator):
    """Prepare a single planet for prediction with all engineered features"""
    # Calculate Earth similarity
    similarity_score, breakdown = similarity_calculator.calculate_similarity_score(planet_features)

    # Create a DataFrame with the basic features
    basic_features = ['orbital_period', 'planetary_radius', 'insolation_flux',
                      'equilibrium_temp', 'stellar_temp', 'stellar_radius',
                      'stellar_mass', 'transit_duration', 'transit_depth']

    feature_dict = {}
    for feature in basic_features:
        feature_dict[feature] = planet_features.get(feature, 0)

    feature_df = pd.DataFrame([feature_dict])

    # Add Earth similarity score
    feature_df['earth_similarity_score'] = similarity_score

    # Apply the same feature engineering as during training
    feature_df = enhanced_feature_engineering(feature_df, feature_df)  # Pass the same DF for both parameters

    # Select only numerical features
    numerical_features = feature_df.select_dtypes(include=[np.number])

    # Impute and scale using the fitted imputer and scaler
    numerical_imputed = imputer.transform(numerical_features)
    numerical_scaled = scaler.transform(numerical_imputed)

    return numerical_scaled, similarity_score, breakdown


def predict_habitability_for_new_planet(model, le, imputer, scaler, planet_features):
    """Predict habitability for a new planet"""
    similarity_calculator = EarthSimilarityCalculator()

    try:
        # Prepare the planet features with all engineered features
        prepared_features, similarity_score, breakdown = prepare_single_planet_for_prediction(
            planet_features, imputer, scaler, similarity_calculator
        )

        # Predict class
        prediction = model.predict(prepared_features)[0]
        prediction_proba = model.predict_proba(prepared_features)[0]

        predicted_class = le.inverse_transform([prediction])[0]
        confidence = prediction_proba[prediction] * 100

        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'earth_similarity_score': similarity_score,
            'similarity_breakdown': breakdown,
            'habitability_potential': 'High' if similarity_score >= 70 else 'Moderate' if similarity_score >= 50 else 'Low'
        }
    except Exception as e:
        print(f"Error predicting habitability: {e}")
        return {
            'predicted_class': 'ERROR',
            'confidence': 0,
            'earth_similarity_score': 0,
            'similarity_breakdown': {},
            'habitability_potential': 'Unknown'
        }


def main():
    """Main execution function"""
    # Load and preprocess data
    X, y, original_df = load_and_preprocess_data('unified_exoplanet_data.csv')

    if len(X) == 0:
        print("No data available for training!")
        return

    # Calculate Earth similarity for all planets
    df_with_similarity = calculate_earth_similarity_for_dataset(original_df)

    # Enhanced feature engineering with Earth similarity
    X_enhanced = enhanced_feature_engineering(X, df_with_similarity)

    # Handle missing values
    X_imputed, imputer = handle_missing_values(X_enhanced)

    print(f"Final feature set: {X_imputed.columns.tolist()}")
    print(f"Shape after preprocessing: {X_imputed.shape}")

    # Prepare data for training
    X_train, X_test, y_train, y_test, le, scaler = prepare_data(X_imputed, y)

    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")

    # Train model with Earth similarity features
    model, y_pred, accuracy = train_model_with_similarity(X_train, X_test, y_train, y_test)

    # Evaluate model
    print("\n" + "=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)
    print(f"Final Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Create visualizations
    top_candidates = visualize_earth_similarity(df_with_similarity)

    # Create comprehensive report
    best_candidates = create_habitability_report(df_with_similarity, model, le, imputer, scaler)

    # Example prediction for a new planet
    print("\n" + "=" * 50)
    print("EXAMPLE HABITABILITY PREDICTION")
    print("=" * 50)

    example_planet = {
        'planetary_radius': 1.2,
        'insolation_flux': 0.9,
        'equilibrium_temp': 260,
        'orbital_period': 380,
        'stellar_temp': 5700,
        'stellar_radius': 1.1,
        'stellar_mass': 1.0,
        'transit_duration': 10.5,
        'transit_depth': 1500
    }

    prediction = predict_habitability_for_new_planet(model, le, imputer, scaler, example_planet)
    print(f"🌍 Example Planet Analysis:")
    print(f"   Predicted Class: {prediction['predicted_class']}")
    print(f"   Confidence: {prediction['confidence']:.1f}%")
    print(f"   Earth Similarity: {prediction['earth_similarity_score']:.1f}%")
    print(f"   Habitability Potential: {prediction['habitability_potential']}")

    if prediction['similarity_breakdown']:
        print(f"\n🎯 Earth Similarity Breakdown:")
        for feature, score in prediction['similarity_breakdown'].items():
            print(f"   {feature.replace('_', ' ').title()}: {score:.1%}")


if __name__ == "__main__":
    main()