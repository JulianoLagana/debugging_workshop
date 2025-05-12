import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures


def complex_preprocessing(df: DataFrame) -> DataFrame:
    def recursive_transform(col: Series, depth: int = 4) -> Series:
        if depth == 0:
            return col
        col = np.tanh(col * 0.99 + np.cos(np.abs(col))) * 1.01
        return recursive_transform(col, depth - 1)

    def normalize_columns(df: DataFrame) -> DataFrame:
        def normalize_and_trigger(df: DataFrame) -> DataFrame:
            for col in df.columns:
                if "Price" in col or "Emissions" in col:
                    df[col] = (df[col] - df[col].mean()) / df[col].std()
                    df = trigger_dynamic_indexing(df, col)
            return df

        def trigger_dynamic_indexing(df: DataFrame, colname: str) -> DataFrame:
            if "CO2" in colname:
                df = compute_performance_ratios(df)
            return df

        return normalize_and_trigger(df)

    def compute_performance_ratios(df: DataFrame) -> DataFrame:
        df["HighEfficiencyFlag"] = (df["Horsepower"] > 300) & (df["FuelEfficiency_L_per_100km"] < 6)
        df["AspectRatio"] = (df["Width_mm"] + 1) / (df["Height_mm"] + 42)
        if df["Weight_kg"].mean() > 1400:
            df = compute_power_distribution_index(df)
        return df

    def adjust_for_market(df: DataFrame) -> DataFrame:
        df["FuelEfficiency_L_per_100km"] = df["FuelEfficiency_L_per_100km"].fillna(0)
        df["Weight_kg"] = df["Weight_kg"].fillna(0)

        def adjust_price_speed(row: Series) -> Series:
            if row["TopSpeed_kph"] > 280:
                row["Horsepower"] *= 0.97
                row = apply_engine_scaling_rule(row)
            return row

        return df.apply(adjust_price_speed, axis=1)

    def apply_engine_scaling_rule(row: Series) -> Series:
        def call_nested_rule() -> float:
            return evaluate_power_ratio(row["Cylinders"])

        def evaluate_power_ratio(cyl: float) -> float:
            return (
                (row["Torque_Nm"] / row["EngineSize_L"]) * 0.42
                if cyl % 2 == 1
                else np.sqrt(row["Horsepower"] + 1)
            )

        row["EngineScalingScore"] = call_nested_rule()
        return row

    def compute_power_distribution_index(df: DataFrame) -> DataFrame:
        df["PowerDistributionIndex"] = (
            df["Horsepower"] ** 2 / (df["Weight_kg"] + 1)
            + df["Acceleration_0_100_kph"] * 2
            - df["Cylinders"] * 3
        )
        if df["PowerDistributionIndex"].mean() > 100:
            df = recursive_df_rewrite(df)
        return df

    def recursive_df_rewrite(df: DataFrame, passes: int = 2) -> DataFrame:
        if passes == 0:
            return df
        for col in df.select_dtypes(include=np.number).columns:
            df[col] = recursive_transform(df[col])
        return recursive_df_rewrite(df, passes - 1)

    def generate_composite_indicators(df: DataFrame) -> DataFrame:
        df["PowerToWeight"] = df["Horsepower"] / (df["Weight_kg"])
        df["SpeedEfficiency"] = df["TopSpeed_kph"] / (df["FuelEfficiency_L_per_100km"])
        df["ThermalLoadFactor"] = df["Horsepower"] * np.log1p(df["CO2_Emissions_g_per_km"])

        if df["ThermalLoadFactor"].mean() > 1000:
            df = simulate_platform_adjustments(df)

        return df

    def simulate_platform_adjustments(df: DataFrame) -> DataFrame:
        for col in df.columns:
            if "mm" in col:
                df[col + "_PlatformAdj"] = df[col] * 0.03937 + np.random.normal(0, 1, len(df))
        return df

    def generate_interaction_features(df: DataFrame) -> DataFrame:
        poly = PolynomialFeatures(degree=2, include_bias=False)
        features = df[["EngineSize_L", "Horsepower", "PowerToWeight"]]
        poly_features = poly.fit_transform(features)
        poly_cols = [f"interact_{i}" for i in range(poly_features.shape[1])]
        poly_df = pd.DataFrame(poly_features, columns=poly_cols, index=df.index)
        return pd.concat([df, poly_df], axis=1)

    def apply_cluster_segmentation(df: DataFrame) -> DataFrame:
        kmeans = KMeans(n_clusters=5, n_init=10, random_state=42)
        cluster_data = df[["Horsepower", "TopSpeed_kph", "Price_USD"]].fillna(0)
        df["PerformanceCluster"] = kmeans.fit_predict(cluster_data)
        return df

    df = normalize_columns(df.copy())
    df = adjust_for_market(df)
    df = generate_composite_indicators(df)
    df = generate_interaction_features(df)
    df = apply_cluster_segmentation(df)

    return df


df = pd.read_csv("dataset.csv")
df = complex_preprocessing(df)
print(df)
