import pandas as pd
import numpy as np


def clean_and_merge_datasets():
    # Load datasets with proper NASA Exoplanet Archive format handling
    try:
        # For NASA Exoplanet Archive files, we need to skip the comment lines
        df1 = pd.read_csv("C:/Users/aadip/PycharmProjects/NASA/.venv/Scripts/data/koi_data.csv",
                          comment='#', skipinitialspace=True, low_memory=False)
        df2 = pd.read_csv("C:/Users/aadip/PycharmProjects/NASA/.venv/Scripts/data/tess_data.csv",
                          comment='#', skipinitialspace=True, low_memory=False)
        df3 = pd.read_csv("C:/Users/aadip/PycharmProjects/NASA/.venv/Scripts/data/k2_data.csv",
                          comment='#', skipinitialspace=True, low_memory=False)
    except Exception as e:
        print(f"Error reading files: {e}")
        return pd.DataFrame()

    # Print actual column names and first few rows for debugging
    print("Dataset 1 columns:", df1.columns.tolist())
    print("Dataset 1 shape:", df1.shape)
    if len(df1) > 0:
        print("Dataset 1 first row disposition:", df1.iloc[0].get('koi_disposition', 'Not found'))

    print("\nDataset 2 columns:", df2.columns.tolist())
    print("Dataset 2 shape:", df2.shape)
    if len(df2) > 0:
        print("Dataset 2 first row disposition:", df2.iloc[0].get('tfopwg_disp', 'Not found'))

    print("\nDataset 3 columns:", df3.columns.tolist())
    print("Dataset 3 shape:", df3.shape)
    if len(df3) > 0:
        print("Dataset 3 first row disposition:", df3.iloc[0].get('k2_disp', 'Not found'))

    # Standardize target classes across all datasets
    def standardize_disposition(disp):
        if pd.isna(disp):
            return 'UNKNOWN'
        disp = str(disp).lower()
        if 'confirm' in disp:
            return 'CONFIRMED'
        elif 'candid' in disp:
            return 'CANDIDATE'
        elif 'false' in disp or 'fp' in disp:
            return 'FALSE POSITIVE'
        else:
            return 'UNKNOWN'

    # Process each dataset
    unified_data = []

    # Dataset 1 processing (Kepler/KOI) - Common column names in KOI data
    if len(df1) > 0:
        for _, row in df1.iterrows():
            try:
                # Try different possible disposition column names for Kepler
                disp = row.get('koi_disposition') or row.get('pl_disp') or row.get('disposition')

                unified_data.append({
                    'target_class': standardize_disposition(disp),
                    'orbital_period': row.get('koi_period') or row.get('pl_orbper'),
                    'planetary_radius': row.get('koi_prad') or row.get('pl_rade'),
                    'insolation_flux': row.get('koi_insol') or row.get('pl_insol'),
                    'equilibrium_temp': row.get('koi_teq') or row.get('pl_eqt'),
                    'stellar_temp': row.get('koi_steff') or row.get('st_teff'),
                    'stellar_radius': row.get('koi_srad') or row.get('st_rad'),
                    'stellar_mass': row.get('koi_smass') or row.get('st_mass'),
                    'distance': row.get('koi_sdist') or row.get('sy_dist'),
                    'transit_duration': row.get('koi_duration') or None,
                    'transit_depth': row.get('koi_depth') or None,
                    'data_source': 'kepler'
                })
            except Exception as e:
                print(f"Error processing Kepler data row: {e}")
                continue

    # Dataset 2 processing (TESS) - Common column names in TESS data
    if len(df2) > 0:
        for _, row in df2.iterrows():
            try:
                # Try different possible disposition column names for TESS
                disp = row.get('tfopwg_disp') or row.get('disposition')

                unified_data.append({
                    'target_class': standardize_disposition(disp),
                    'orbital_period': row.get('pl_orbper') or row.get('period'),
                    'planetary_radius': row.get('pl_rade') or row.get('pl_rad'),
                    'insolation_flux': row.get('pl_insol') or None,
                    'equilibrium_temp': row.get('pl_eqt') or row.get('eqt'),
                    'stellar_temp': row.get('st_teff') or row.get('teff'),
                    'stellar_radius': row.get('st_rad') or row.get('srad'),
                    'stellar_mass': row.get('st_mass') or row.get('mass'),
                    'distance': row.get('sy_dist') or row.get('dist'),
                    'transit_duration': row.get('pl_trandur') or row.get('duration'),
                    'transit_depth': row.get('pl_trandep') or row.get('depth'),
                    'data_source': 'tess'
                })
            except Exception as e:
                print(f"Error processing TESS data row: {e}")
                continue

    # Dataset 3 processing (K2)
    if len(df3) > 0:
        for _, row in df3.iterrows():
            try:
                unified_data.append({
                    'target_class': standardize_disposition(row.get('k2_disp')),
                    'orbital_period': row.get('pl_orbper'),
                    'planetary_radius': row.get('pl_rade'),
                    'insolation_flux': None,
                    'equilibrium_temp': None,
                    'stellar_temp': row.get('st_teff'),
                    'stellar_radius': row.get('st_rad'),
                    'stellar_mass': row.get('st_mass'),
                    'distance': None,
                    'transit_duration': None,
                    'transit_depth': None,
                    'data_source': 'k2'
                })
            except Exception as e:
                print(f"Error processing K2 data row: {e}")
                continue

    result_df = pd.DataFrame(unified_data)
    print(f"\nUnified data shape before cleaning: {result_df.shape}")
    return result_df


def handle_missing_values(df):
    if len(df) == 0:
        return df

    print(f"\nBefore missing value handling: {df.shape}")
    print("Target class distribution:")
    print(df['target_class'].value_counts())

    # Remove rows where target_class is UNKNOWN or missing
    df = df[df['target_class'] != 'UNKNOWN']
    print(f"After removing UNKNOWN targets: {df.shape}")

    # Remove rows with >50% missing data
    df = df.dropna(thresh=len(df.columns) // 2)
    print(f"After removing rows with >50% missing data: {df.shape}")

    # Impute numerical columns with median
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
            print(f"Imputed missing values in {col}")

    return df


def remove_outliers(df):
    if len(df) == 0:
        return df

    initial_shape = df.shape
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].nunique() > 1:  # Only remove outliers if we have variation
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            if IQR > 0:  # Only filter if IQR is meaningful
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    print(f"Outlier removal: {initial_shape[0]} -> {df.shape[0]} rows")
    return df


# Execute pipeline
print("Starting data processing...")
final_df = clean_and_merge_datasets()

if len(final_df) > 0:
    print("Handling missing values...")
    final_df = handle_missing_values(final_df)

    print("Removing outliers...")
    final_df = remove_outliers(final_df)

    print("Saving unified dataset...")
    final_df.to_csv('unified_exoplanet_data.csv', index=False)
    print("Done! Unified dataset saved as 'unified_exoplanet_data.csv'")

    # Print final statistics
    print("\nFinal dataset info:")
    print(f"Total records: {len(final_df)}")
    print("Records by data source:")
    print(final_df['data_source'].value_counts())
    print("Records by target class:")
    print(final_df['target_class'].value_counts())
else:
    print("No data was processed. Check the file paths and formats.")