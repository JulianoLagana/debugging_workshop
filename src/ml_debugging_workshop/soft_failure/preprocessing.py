import os

import pandas as pd


def map_top_k_values(
    df: pd.DataFrame, column: str, min_ratio: float = 0.8, other_label: str = "other"
) -> pd.Series:
    value_counts = df[column].value_counts(normalize=True)
    cumulative_ratio = value_counts.cumsum()

    top_values = cumulative_ratio[cumulative_ratio <= min_ratio].index.tolist()
    if len(top_values) < len(cumulative_ratio):
        top_values.append(cumulative_ratio.index[len(top_values)])

    return df[column].apply(lambda x: x if x in top_values else other_label)


def load_data() -> pd.DataFrame:
    df = pd.read_csv("mushrooms.csv", na_values=" ?", skipinitialspace=True, delimiter=";")
    df.fillna("missing", inplace=True)
    df = df.dropna()

    categorical_cols = df.select_dtypes(include="object").columns
    for c in categorical_cols:
        df[c] = map_top_k_values(df, c)

    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns
    df[numerical_cols] = (df[numerical_cols] - df[numerical_cols].mean()) / df[numerical_cols].std()

    return df


def one_hot_encode(values: pd.Series) -> list[list[int]]:
    mapping: dict[str, int] = {}
    for v in values:
        if v not in mapping:
            mapping[v] = len(mapping)

    result = []
    for v in values:
        vec = [0] * len(mapping)
        vec[mapping[v]] = 1
        result.append(vec)

    return result


def one_hot_encode_df(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include="object").columns:
        one_hot = one_hot_encode(df[col])
        for i in range(len(one_hot[0])):
            df[f"{col}_{i}"] = [vec[i] for vec in one_hot]
        df.drop(columns=[col], inplace=True)

        if col == "class":
            df.drop(columns=["class_0"], inplace=True)

    return df


def split_dataframe(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.20,
    test_frac: float = 0.10,
    seed: float = 1337,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Fractions must sum to 1"

    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    n_total = len(df_shuffled)
    n_train = int(n_total * train_frac)
    n_val = int(n_total * val_frac)

    df_train = df_shuffled.iloc[:n_train].copy()
    df_val = df_shuffled.iloc[n_train : n_train + n_val].copy()
    df_test = df_shuffled.iloc[n_train + n_val :].copy()

    df_train = one_hot_encode_df(df_train)
    df_val = one_hot_encode_df(df_val)
    df_test = one_hot_encode_df(df_test)

    return df_train, df_val, df_test


def main() -> None:
    df = load_data()
    df_train, df_val, df_test = split_dataframe(df)

    os.makedirs("splits", exist_ok=True)
    df_train.to_csv("splits/train.csv", index=False)
    df_val.to_csv("splits/val.csv", index=False)
    df_test.to_csv("splits/test.csv", index=False)


if __name__ == "__main__":
    main()
