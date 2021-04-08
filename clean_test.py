import pandas as pd


def impute_impossible_values(df):
    # Replace impossible values in test with training average
    # Negative full_live diff
    df.loc[df.full_live_diff < 0, "full_live_diff"] = (
        train_average["full_sq"] - train_average["life_sq"]
    )

    # max_floor = 0 results in infinity
    df.loc[df.max_floor <= 0, "ratio_floor_max_floor"] = (
        train_average["floor"] / train_average["max_floor"]
    )
    df.loc[df.max_floor <= 0, "max_floor"] = train_average["max_floor"]

    # Negative floor from top
    df.loc[df.floor_from_top < 0, "floor_from_top"] = (
        train_average["max_floor"] - train_average["floor"]
    )
    # Negative age of building
    df.loc[df.age_of_building < 0, "age_of_building"] = (
        df.loc[df.age_of_building < 0]["year"] - train_average["build_year"]
    )
    df.loc[df.build_year < 0, "build_year"] = train_average["build_year"]

    return df


# Environment Variables
TRAIN_PATH = "input/train.csv"
TEST_PATH = "feature_eng/test_X_1.csv"
TEST_OUT_PATH = "feature_eng/test_X_1_clean.csv"

df_train = pd.read_csv(TRAIN_PATH, index_col="id")
df_test = pd.read_csv(TEST_PATH)
train_average = df_train.mean(axis=0)
df_test_clean = impute_impossible_values(df_test)
df_test_clean.to_csv(TEST_OUT_PATH, index=False)
