import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mysql.connector
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Set page config
st.set_page_config(page_title="Tourism Analytics System", layout="wide")

# Database connection function
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="4628@Saru",
        database="Project_4",
        auth_plugin='mysql_native_password'
    )

# Optimized data loading with correct column names
@st.cache_data(show_spinner="Loading tourism data...")
def load_data():
    conn = get_db_connection()
    query = """
    SELECT 
        UserId, VisitYear, VisitMonth, AttractionId, Rating,
        ContinentId, RegionId, CountryId, CityId, Country,
        Region, Continent, AttractionTypeId, Attraction,
        AttractionAddress, AttractionType, VisitModeId, VisitMode
    FROM TourismRatings
    """
    df = pd.read_sql(query, conn)
    conn.close()
    df = df.dropna(subset=['UserId', 'AttractionId', 'Rating'])

    np.random.seed(42)
    unique_users = df['UserId'].unique()
    user_profiles = pd.DataFrame({
        'UserId': unique_users,
        'Age': np.random.randint(18, 70, len(unique_users)),
        'Gender': np.random.choice(['Male', 'Female', 'Other'], len(unique_users)),
        'IncomeLevel': np.random.choice(['Low', 'Medium', 'High'], len(unique_users))
    })
    df = df.merge(user_profiles, on='UserId')
    return df

# Train visit mode prediction model
@st.cache_resource
def train_visit_mode_model(df):
    features = df[['Age', 'Gender', 'IncomeLevel', 'Country']]
    features = pd.get_dummies(features)
    le = LabelEncoder()
    target = le.fit_transform(df['VisitMode'])
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model, le

# Recommendation system with preference filtering
def get_recommendations(df, user_id=None, country=None, visit_mode=None, num_recommendations=10):
    df = df.dropna(subset=['UserId', 'AttractionId', 'Rating'])
    rating_matrix = df.pivot_table(index='UserId', columns='AttractionId', values='Rating').fillna(0)

    if user_id and user_id in rating_matrix.index:
        # Collaborative filtering: find similar users
        user_vec = rating_matrix.loc[user_id].values.reshape(1, -1)
        similarity = cosine_similarity(user_vec, rating_matrix.values)[0]
        similar_users = pd.Series(similarity, index=rating_matrix.index).sort_values(ascending=False)[1:11]

        # Aggregate their rated attractions
        sim_users_df = df[df['UserId'].isin(similar_users.index)]
        recommendations = sim_users_df.groupby('AttractionId').agg({
            'Rating': 'mean',
            'Attraction': 'first',
            'AttractionType': 'first',
            'AttractionAddress': 'first',
            'Country': 'first',
            'VisitMode': 'first'
        }).reset_index()
    else:
        # Fallback to content-based if user is new or not found
        recommendations = df.copy()

    # Apply content-based filters
    if country:
        recommendations = recommendations[
            recommendations['Country'].fillna('').str.lower() == country.lower()
        ]
    if visit_mode:
        recommendations = recommendations[
            recommendations['VisitMode'].fillna('').str.lower() == visit_mode.lower()
        ]

    # Rank by average rating
    recommendations = (
        recommendations.groupby('AttractionId')
        .agg({
            'Attraction': 'first',
            'AttractionType': 'first',
            'AttractionAddress': 'first',
            'Rating': 'mean'
        })
        .sort_values(by='Rating', ascending=False)
        .head(num_recommendations)
    )

    return recommendations

# Main application
def main():
    st.title("Tourism Analytics & Recommendation System")
    df = load_data()

    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose the section", ["Trend Analysis", "Visit Mode Prediction", "Personalized Recommendations"])

    if app_mode == "Trend Analysis":
        st.header("Tourism Trend Analysis")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("User Demographics")
            fig, ax = plt.subplots()
            sns.histplot(df['Age'], bins=20, kde=True, ax=ax, color='skyblue')
            ax.set_title("Age Distribution", fontsize=14)
            st.pyplot(fig)

            gender_counts = df['Gender'].value_counts()
            fig, ax = plt.subplots()
            colors = ['#66c2a5', '#fc8d62', '#8da0cb']
            gender_counts.plot.pie(autopct='%1.1f%%', ax=ax, colors=colors, startangle=90)
            ax.set_ylabel('')
            ax.set_title("Gender Distribution", fontsize=14)
            st.pyplot(fig)

        with col2:
            top_attractions = df.groupby('Attraction')['Rating'].mean().sort_values(ascending=False).head(10)

            fig, ax = plt.subplots(figsize=(10, 8))  
            sns.barplot(x=top_attractions.values, y=top_attractions.index, ax=ax, palette='viridis')

            ax.set_xlabel("Average Rating", fontsize=14)
            ax.set_ylabel("Attraction", fontsize=14)
            ax.set_title("Top Rated Attractions", fontsize=16)
            ax.tick_params(axis='both', labelsize=13)

            st.pyplot(fig)


            attraction_types = df['AttractionType'].value_counts().head(10)
            fig, ax = plt.subplots()
            attraction_types.plot.bar(ax=ax, color='coral')
            ax.set_ylabel("Count")
            ax.set_title("Popular Attraction Types", fontsize=14)
            st.pyplot(fig)

        st.subheader("Visit Modes by Region")
        visit_region = pd.crosstab(df['VisitMode'], df['Region'], normalize='index')
        fig, ax = plt.subplots(figsize=(14, 6))
        visit_region.plot.bar(stacked=True, ax=ax, colormap='tab20')
        ax.set_ylabel("Proportion")
        ax.set_title("Visit Modes by Region", fontsize=14)
        ax.legend(title="Region", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
        st.pyplot(fig)

    elif app_mode == "Visit Mode Prediction":
        st.header("Visit Mode Prediction")
        model, le = train_visit_mode_model(df)

        with st.form("user_profile"):
            st.subheader("User Profile")
            age = st.slider("Age", 18, 80, 30)
            gender = st.selectbox("Gender", ['Male', 'Female', 'Other'])
            income = st.selectbox("Income Level", ['Low', 'Medium', 'High'])
            country = st.selectbox("Country", df['Country'].unique())

            if st.form_submit_button("Predict Visit Mode"):
                input_data = pd.DataFrame([[age, gender, income, country]], columns=['Age', 'Gender', 'IncomeLevel', 'Country'])
                input_data = pd.get_dummies(input_data)
                train_cols = model.feature_names_in_
                for col in train_cols:
                    if col not in input_data.columns:
                        input_data[col] = 0
                input_data = input_data[train_cols]
                prediction = model.predict(input_data)
                predicted_mode = le.inverse_transform(prediction)[0]
                st.success(f"Predicted Visit Mode: **{predicted_mode}**")

                st.subheader("Factors Influencing Prediction")
                # Feature importance grouped by original feature
                importances = pd.DataFrame({
                    'Feature': model.feature_names_in_,
                    'Importance': model.feature_importances_
                })

            # Mapping each one-hot encoded feature to its original column
                def map_original_feature(feature_name):
                    if feature_name.startswith("Gender_"):
                        return "Gender"
                    elif feature_name.startswith("IncomeLevel_"):
                        return "IncomeLevel"
                    elif feature_name.startswith("Country_"):
                        return "Country"
                    else:
                        return feature_name  # e.g., Age

        importances['OriginalFeature'] = importances['Feature'].apply(map_original_feature)

        # Group by original features
        grouped_importances = importances.groupby('OriginalFeature').sum().reset_index()
        grouped_importances = grouped_importances.sort_values(by='Importance', ascending=False)

        # Plotting
        fig, ax = plt.subplots()
        sns.barplot(x='Importance', y='OriginalFeature', data=grouped_importances, ax=ax)
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        ax.set_title("Factors Influencing Prediction")
        st.pyplot(fig)


    elif app_mode == "Personalized Recommendations":
        st.header("🌍 Destination Recommendations")
    
    # Generate synthetic tourism data
        @st.cache_data
        def generate_synthetic_data():
            countries = ['India', 'Indonesia', 'Thailand', 'Japan', 'France', 'Italy', 'Spain', 'USA']
            attractions = {
            'India': ['Taj Mahal', 'Jaipur Palace', 'Goa Beaches', 'Kerala Backwaters', 'Varanasi Ghats'],
            'Indonesia': ['Bali Beaches', 'Borobudur Temple', 'Komodo Island', 'Raja Ampat', 'Mount Bromo'],
            'Thailand': ['Grand Palace', 'Phi Phi Islands', 'Chiang Mai Temples', 'Ayutthaya', 'Pattaya Beach'],
            'Japan': ['Mount Fuji', 'Tokyo Tower', 'Kyoto Temples', 'Osaka Castle', 'Hiroshima Peace Park'],
            'France': ['Eiffel Tower', 'Louvre Museum', 'Mont Saint-Michel', 'French Riviera', 'Versailles Palace'],
            'Italy': ['Colosseum', 'Venice Canals', 'Leaning Tower', 'Amalfi Coast', 'Pompeii Ruins'],
            'Spain': ['Sagrada Familia', 'Alhambra', 'Ibiza Beaches', 'Park Guell', 'Seville Cathedral'],
            'USA': ['Statue of Liberty', 'Grand Canyon', 'Yellowstone Park', 'Las Vegas Strip', 'Golden Gate Bridge']
        }
        
            visit_modes = ['Couples', 'Friends', 'Family', 'Solo', 'Business']
            attraction_types = ['Historical', 'Beach', 'Mountain', 'City', 'Religious', 'Natural']
        
            np.random.seed(42)
            records = []
        
            for country in countries:
                for attraction in attractions[country]:
                    for _ in range(np.random.randint(50, 200)):  # Generate 50-200 ratings per attraction
                        records.append({
                            'Country': country,
                            'Attraction': attraction,
                            'AttractionType': np.random.choice(attraction_types),
                            'AttractionAddress': f"Address of {attraction}, {country}",
                            'VisitMode': np.random.choice(visit_modes),
                            'Rating': np.random.normal(loc=4.2, scale=0.5)  # Ratings centered around 4.2
                        })
        
            df_synthetic = pd.DataFrame(records)
            df_synthetic['Rating'] = df_synthetic['Rating'].clip(1, 5)  # Ensure ratings between 1-5
            return df_synthetic
    
        df_synth = generate_synthetic_data()
    
        # User inputs
        col1, col2 = st.columns(2)
        with col1:
            country = st.selectbox("Select Country", sorted(df_synth['Country'].unique()))
        with col2:
            visit_mode = st.selectbox("Select Visit Mode", sorted(df_synth['VisitMode'].unique()))
        
        # Filter button
        if st.button("Get Recommendations", type="primary"):
            # Filter and calculate average ratings
            recommendations = (
                df_synth[(df_synth['Country'] == country) & 
                        (df_synth['VisitMode'] == visit_mode)]
                .groupby('Attraction')
                .agg({
                    'AttractionType': 'first',
                    'AttractionAddress': 'first',
                    'Rating': ['mean', 'count']
                })
                .sort_values(('Rating', 'mean'), ascending=False)
            )
            
            # Flatten multi-index columns
            recommendations.columns = ['Type', 'Address', 'AvgRating', 'RatingCount']
            
            if recommendations.empty:
                st.warning(f"No attractions found for {visit_mode} trips in {country}")
            else:
                st.success(f"Found {len(recommendations)} attractions for {visit_mode} in {country}")
                
                # Display recommendations
                st.subheader("Top Attractions")
                for i, (idx, row) in enumerate(recommendations.iterrows(), start=1):
                    st.markdown(f"**{i}. {idx}** – ⭐ {row['AvgRating']:.2f}")

                
if __name__ == "__main__":
    main()
