import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from PIL import Image
import io
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Exoplanet Habitability Analyzer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

            # Insolation flux similarity (critical for habitable zone)
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
            return 0, {}


# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .earth-like {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .habitable {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)


class ExoplanetHabitabilityUI:
    def __init__(self):
        self.loaded_data = None
        self.model = None
        self.similarity_calculator = None
        self.imputer = None
        self.scaler = None
        self.le = None

    def load_resources(self):
        """Load the trained model and data"""
        try:
            # Load the enhanced dataset
            self.loaded_data = pd.read_csv('exoplanets_with_earth_similarity.csv')

            # Load the trained model and preprocessing objects
            self.model = joblib.load('exoplanet_habitability_model.pkl')

            # Instead of loading the similarity calculator, create a new instance
            self.similarity_calculator = EarthSimilarityCalculator()

            self.imputer = joblib.load('imputer.pkl')
            self.scaler = joblib.load('scaler.pkl')
            self.le = joblib.load('label_encoder.pkl')

            return True
        except Exception as e:
            st.error(f"Error loading resources: {e}")
            st.info(f"Detailed error: {str(e)}")
            return False

    def display_header(self):
        """Display the main header and introduction"""
        st.markdown('<h1 class="main-header">🌍 Exoplanet Habitability Analyzer</h1>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style='text-align: center; color: #666; font-size: 1.2rem;'>
            Discover and analyze potentially habitable exoplanets using advanced machine learning 
            and Earth Similarity Index (ESI) calculations
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

    def display_key_metrics(self):
        """Display key metrics about the dataset"""
        st.markdown('<h2 class="sub-header">📊 Dataset Overview</h2>', unsafe_allow_html=True)

        if self.loaded_data is None:
            st.warning("No data loaded yet. Please wait while we load the analysis results.")
            return

        # Calculate key metrics
        total_planets = len(self.loaded_data)
        confirmed_planets = len(self.loaded_data[self.loaded_data['target_class'] == 'CONFIRMED'])

        # Handle cases where earth_similarity_score might not exist
        if 'earth_similarity_score' in self.loaded_data.columns:
            # Filter out zero scores for meaningful metrics
            valid_scores = self.loaded_data[self.loaded_data['earth_similarity_score'] > 0]
            high_similarity = len(valid_scores[valid_scores['earth_similarity_score'] >= 70])
            moderate_similarity = len(valid_scores[valid_scores['earth_similarity_score'] >= 50])
            best_candidate_score = valid_scores['earth_similarity_score'].max() if len(valid_scores) > 0 else 0
        else:
            high_similarity = 0
            moderate_similarity = 0
            best_candidate_score = 0

        # Display metrics in columns
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Total Planets Analyzed", f"{total_planets:,}")

        with col2:
            st.metric("Confirmed Exoplanets", f"{confirmed_planets:,}")

        with col3:
            st.metric("High Similarity (≥70%)", f"{high_similarity}")

        with col4:
            st.metric("Moderate Similarity (≥50%)", f"{moderate_similarity}")

        with col5:
            st.metric("Best Earth Similarity", f"{best_candidate_score:.1f}%" if best_candidate_score > 0 else "N/A")

    def create_similarity_distribution_chart(self):
        """Create interactive distribution chart of Earth similarity scores"""
        st.markdown('<h2 class="sub-header">📈 Earth Similarity Distribution</h2>', unsafe_allow_html=True)

        if self.loaded_data is None or 'earth_similarity_score' not in self.loaded_data.columns:
            st.warning("Earth similarity data not available.")
            return

        # Filter out zero scores for better visualization
        filtered_data = self.loaded_data[self.loaded_data['earth_similarity_score'] > 0]

        if len(filtered_data) == 0:
            st.warning("No valid Earth similarity scores to display.")
            return

        # Create distribution plot
        fig = px.histogram(
            filtered_data,
            x='earth_similarity_score',
            nbins=30,
            title='Distribution of Earth Similarity Scores',
            labels={'earth_similarity_score': 'Earth Similarity Score (%)'},
            color_discrete_sequence=['#1f77b4']
        )

        fig.update_layout(
            xaxis_title='Earth Similarity Score (%)',
            yaxis_title='Number of Planets',
            showlegend=False
        )

        # Add vertical lines for similarity thresholds
        if len(filtered_data) > 0:
            fig.add_vline(x=50, line_dash="dash", line_color="orange", annotation_text="Moderate")
            fig.add_vline(x=70, line_dash="dash", line_color="green", annotation_text="High")

        st.plotly_chart(fig, use_container_width=True)

    def create_similarity_by_class_chart(self):
        """Create comparison chart of similarity by planet class"""
        st.markdown('<h2 class="sub-header">🔬 Similarity by Planet Classification</h2>', unsafe_allow_html=True)

        if self.loaded_data is None or 'earth_similarity_score' not in self.loaded_data.columns:
            st.warning("Earth similarity data not available.")
            return

        # Filter out zero scores
        filtered_data = self.loaded_data[self.loaded_data['earth_similarity_score'] > 0]

        if len(filtered_data) == 0:
            st.warning("No valid Earth similarity scores to display.")
            return

        # Calculate average similarity by class
        similarity_by_class = filtered_data.groupby('target_class')['earth_similarity_score'].mean().reset_index()

        fig = px.bar(
            similarity_by_class,
            x='target_class',
            y='earth_similarity_score',
            title='Average Earth Similarity by Planet Classification',
            labels={'earth_similarity_score': 'Average Similarity Score (%)', 'target_class': 'Planet Classification'},
            color='target_class',
            color_discrete_sequence=['#28a745', '#ffc107', '#dc3545']  # Green, Yellow, Red
        )

        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    def display_top_candidates(self):
        """Display the top Earth-like exoplanet candidates"""
        st.markdown('<h2 class="sub-header">🏆 Top Earth-Like Exoplanet Candidates</h2>', unsafe_allow_html=True)

        if self.loaded_data is None or 'earth_similarity_score' not in self.loaded_data.columns:
            st.warning("Earth similarity data not available.")
            return

        # Get top confirmed planets with high similarity
        filtered_data = self.loaded_data[self.loaded_data['earth_similarity_score'] > 0]
        top_candidates = filtered_data[
            (filtered_data['target_class'] == 'CONFIRMED')
        ].nlargest(10, 'earth_similarity_score')

        if len(top_candidates) == 0:
            st.info("No confirmed exoplanets with significant Earth similarity found in the dataset.")
            return

        # Create a beautiful display for top candidates
        for idx, (_, candidate) in enumerate(top_candidates.iterrows(), 1):
            similarity = candidate['earth_similarity_score']
            category = candidate.get('earth_similarity_category', 'Unknown')

            # Determine card style based on similarity
            card_class = "earth-like" if similarity >= 70 else "habitable" if similarity >= 50 else "metric-card"

            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"""
                <div class='metric-card {card_class}'>
                    <h3>#{idx} - Earth Similarity: {similarity:.1f}% ({category})</h3>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                # Create a progress bar for similarity score
                progress = similarity / 100
                st.progress(progress, text=f"{similarity:.1f}%")

        # Display detailed table
        with st.expander("View Detailed Candidate Data"):
            display_columns = [
                'earth_similarity_score', 'earth_similarity_category',
                'planetary_radius', 'insolation_flux', 'equilibrium_temp',
                'orbital_period', 'stellar_temp'
            ]
            # Select only columns that exist in the dataframe
            available_columns = [col for col in display_columns if col in top_candidates.columns]
            if available_columns:
                st.dataframe(top_candidates[available_columns].round(2))
            else:
                st.info("No detailed data available for display.")

    def create_habitability_pie_chart(self):
        """Create pie chart showing habitability potential"""
        st.markdown('<h2 class="sub-header">🥧 Habitability Potential Distribution</h2>', unsafe_allow_html=True)

        if self.loaded_data is None or 'earth_similarity_category' not in self.loaded_data.columns:
            st.warning("Earth similarity category data not available.")
            return

        # Filter out "Very Low" category for better visualization
        filtered_data = self.loaded_data[self.loaded_data['earth_similarity_category'] != 'Very Low']

        if len(filtered_data) == 0:
            st.warning("No planets with significant Earth similarity to display.")
            return

        # Count planets by similarity category
        category_counts = filtered_data['earth_similarity_category'].value_counts()

        # Create pie chart
        fig = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title='Distribution of Planets by Earth Similarity Category',
            color_discrete_sequence=px.colors.sequential.Blues_r
        )

        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    def create_feature_importance_plot(self):
        """Create visualization of feature importance in similarity calculation"""
        st.markdown('<h2 class="sub-header">⚖️ Earth Similarity Components</h2>', unsafe_allow_html=True)

        # Feature weights from the similarity calculator
        weights = {
            'Planetary Radius': 0.3,
            'Insolation Flux': 0.25,
            'Equilibrium Temperature': 0.2,
            'Orbital Period': 0.1,
            'Stellar Temperature': 0.1,
            'Stellar Radius': 0.05
        }

        fig = px.bar(
            x=list(weights.keys()),
            y=list(weights.values()),
            title='Weight Distribution in Earth Similarity Calculation',
            labels={'x': 'Planetary Features', 'y': 'Weight Importance'},
            color=list(weights.values()),
            color_continuous_scale='Blues'
        )

        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    def display_model_performance(self):
        """Display model performance metrics"""
        st.markdown('<h2 class="sub-header">🤖 Model Performance</h2>', unsafe_allow_html=True)

        # # This would typically come from your model evaluation
        # # For demonstration, we'll use placeholder values
        # col1, col2, col3, col4 = st.columns(4)
        #
        # with col1:
        #     st.metric("Model Accuracy", "73.1%")
        #
        # with col2:
        #     st.metric("Precision", "72.5%")
        #
        # with col3:
        #     st.metric("Recall", "73.8%")
        #
        # with col4:
        #     st.metric("F1-Score", "73.1%")

        st.info("""
        **Model Details:**
        - Algorithm: XGBoost Classifier
        - Features: Planetary characteristics + Earth Similarity Index
        - Training: Cross-validated with stratified sampling
        - Purpose: Classify exoplanets and assess habitability potential
        """)

    def create_comparison_radar_chart(self):
        """Create radar chart comparing Earth with top candidate"""
        st.markdown('<h2 class="sub-header">📊 Earth vs Top Candidate Comparison</h2>', unsafe_allow_html=True)

        if self.loaded_data is None:
            st.info("No data available for comparison.")
            return

        # Get top candidate
        filtered_data = self.loaded_data[self.loaded_data['earth_similarity_score'] > 0]
        top_candidate = filtered_data[
            (filtered_data['target_class'] == 'CONFIRMED')
        ].nlargest(1, 'earth_similarity_score')

        if len(top_candidate) == 0:
            st.info("No suitable candidate for comparison.")
            return

        candidate = top_candidate.iloc[0]

        # Normalized values for radar chart (0-1 scale)
        categories = ['Size', 'Temperature', 'Insolation', 'Orbit', 'Star Temp']

        earth_values = [1.0, 1.0, 1.0, 1.0, 1.0]  # Earth is reference

        # Candidate values (normalized relative to Earth)
        candidate_values = [
            min(candidate.get('planetary_radius', 0) / 1.0, 2.0),  # Cap at 2x Earth size
            min(abs(candidate.get('equilibrium_temp', 0) - 255) / 50, 2.0),  # Temp difference
            min(candidate.get('insolation_flux', 0) / 1.0, 2.0),  # Insolation relative to Earth
            min(candidate.get('orbital_period', 0) / 365.25, 2.0),  # Orbital period
            min(candidate.get('stellar_temp', 0) / 5778, 2.0)  # Stellar temperature
        ]

        # Create radar chart
        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=earth_values,
            theta=categories,
            fill='toself',
            name='Earth Reference',
            line_color='blue'
        ))

        fig.add_trace(go.Scatterpolar(
            r=candidate_values,
            theta=categories,
            fill='toself',
            name=f'Top Candidate ({candidate["earth_similarity_score"]:.1f}%)',
            line_color='red'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 2]
                )),
            showlegend=True,
            title="Feature Comparison: Earth vs Best Candidate"
        )

        st.plotly_chart(fig, use_container_width=True)

    def display_insights(self):
        """Display key insights and conclusions"""
        st.markdown('<h2 class="sub-header">💡 Key Insights</h2>', unsafe_allow_html=True)

        if self.loaded_data is None or 'earth_similarity_score' not in self.loaded_data.columns:
            st.warning("No insights available - data not loaded.")
            return

        # Filter out zero scores
        filtered_data = self.loaded_data[self.loaded_data['earth_similarity_score'] > 0]

        # Calculate insights
        total_habitable = len(filtered_data[filtered_data['earth_similarity_score'] >= 50])
        high_confidence_habitable = len(filtered_data[filtered_data['earth_similarity_score'] >= 70])

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div class='metric-card earth-like'>
                <h4>🌱 Potentially Habitable Planets</h4>
                <p>We've identified <strong>{total_habitable}</strong> planets with moderate to high 
                Earth similarity (≥50%), including <strong>{high_confidence_habitable}</strong> with high 
                similarity (≥70%).</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class='metric-card'>
                <h4>🔭 Scientific Impact</h4>
                <p>This analysis provides prioritized targets for future telescope 
                observations and helps focus the search for extraterrestrial life 
                on the most promising candidates.</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class='metric-card'>
                <h4>🎯 Observation Priorities</h4>
                <p>The top candidates identified here should be prioritized for:
                - Atmospheric spectroscopy studies
                - Follow-up radial velocity measurements
                - Future direct imaging missions
                - SETI observations</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class='metric-card'>
                <h4>📊 Methodology</h4>
                <p>Our Earth Similarity Index combines multiple planetary parameters 
                with scientifically-weighted importance to provide a comprehensive 
                habitability assessment.</p>
            </div>
            """, unsafe_allow_html=True)

    def run(self):
        """Main method to run the UI"""
        # Display header
        self.display_header()

        # Load resources
        with st.spinner("Loading exoplanet analysis data..."):
            if not self.load_resources():
                st.error("Failed to load required resources. Please ensure the model files are available.")
                st.info("""
                **To generate the required files:**
                1. Run your model training script first to generate:
                   - exoplanets_with_earth_similarity.csv
                   - exoplanet_habitability_model.pkl
                   - imputer.pkl, scaler.pkl, label_encoder.pkl
                2. Make sure all files are in the same directory as this Streamlit app
                3. Then restart this UI
                """)
                return

        # Display key metrics
        self.display_key_metrics()

        # Create two main columns for layout
        col1, col2 = st.columns(2)

        with col1:
            self.create_similarity_distribution_chart()
            self.create_habitability_pie_chart()
            self.display_top_candidates()

        with col2:
            self.create_similarity_by_class_chart()
            self.create_feature_importance_plot()
            self.create_comparison_radar_chart()

        # Full width sections
        self.display_model_performance()
        self.display_insights()

        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666;'>
        <p>Exoplanet Habitability Analyzer | Using Machine Learning and Earth Similarity Index</p>
        <p>Data Source: NASA Exoplanet Archive | Model: XGBoost Classifier</p>
        </div>
        """, unsafe_allow_html=True)


# Create and run the UI
if __name__ == "__main__":
    ui = ExoplanetHabitabilityUI()
    ui.run()