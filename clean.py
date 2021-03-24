import numpy as np
import pandas as pd
import logging
import os
from sklearn import model_selection, preprocessing
from scipy import stats

pd.options.mode.chained_assignment = None  # default='warn'

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger("clean")


def correct_lon_lat(df, ref_df):
    log.info("Correcting longitude latitude info")
    df["lon"] = ref_df["lon"]
    df["lat"] = ref_df["lat"]
    return df


def truncate_extreme_values(df):
    # Truncate extreme values above [1] percentile and below [2] percentile in [0] col
    cols = [("price_doc", 99.5, 0.5), ("full_sq", 99.5, 0.5), ("life_sq", 95, 5)]
    for col in cols:
        col_values = df[col[0]].to_numpy()
        upper_limit = np.percentile(
            np.delete(col_values, np.where(col_values == -99)), col[1]
        )
        lower_limit = np.percentile(
            np.delete(col_values, np.where(col_values == -99)), col[2]
        )
        df[col[0]].loc[df[col[0]] > upper_limit] = upper_limit
        df[col[0]].loc[df[col[0]] < lower_limit] = lower_limit
        log.info(
            "Truncating extreme values for col {}, {} percentile = {}, {} percentile = {}".format(
                col[0], col[1], upper_limit, col[2], lower_limit
            )
        )
    return df


def convert_categorical_to_numerical(df):
    log.info("Converting categorical data to numerical")
    lb_encoder = preprocessing.LabelEncoder()
    for f in df.columns:
        if df[f].dtype == "object":
            lb_encoder.fit(
                list(df[f].values.astype("str")) + list(df[f].values.astype("str"))
            )
            df[f] = lb_encoder.transform(list(df[f].values.astype("str")))
    return df


def impute_missing_values(df):
    log.info("Filling missing values with -99")
    # Fill NAN with the unlikely -99. Remove this value from arrays when calculating
    df.fillna(-99, inplace=True)
    return df


def handle_bad_data(df):
    log.info("Handling bad data")
    # Remove records where kitchen squares and life squres are larger than full squares
    df = df.drop(df[df["life_sq"] > df["full_sq"]].index)
    df = df.drop(df[df["kitch_sq"] > df["full_sq"]].index)

    # Remove an outlier record with full_sq == 5326
    df = df.drop(df_train[df_train["full_sq"] == 5326].index)

    # Remove records where build year is less than 1691 and greater than 2018. Some entries include 0, 1, 3, 20, 71
    df = df.drop(df[df["build_year"] <= 1691].index)
    df = df.drop(df[df["build_year"] > 2018].index)

    # Remove records with max floor > 57 (99, 117)
    df = df.drop(df[df["max_floor"] > 57].index)

    # Remove records where actual floor > max floor
    df = df.drop(df[df["floor"] > df["max_floor"]].index)

    # Remove records where full_sq are 0 and 1 which are obvious errors
    df = df.drop(df_train[df_train["full_sq"] == 0].index)
    df = df.drop(df_train[df_train["full_sq"] == 1].index)

    # State should be discrete valued between 1 and 4. There is a 33 in it that is clearly a data entry error.
    # Replace it with mode of state
    df["state"].loc[df["state"] == 33] = stats.mode(df["state"].values)[0][0]

    # build_year has an erroneous value 20052009. Since its unclear which it should be.
    # Replace with 2007
    df["build_year"].loc[df["build_year"] == 20052009] = 2007

    return df


def clean(df, ref_df):
    log.info("Cleaning pipeline started")
    df = correct_lon_lat(df, ref_df)
    df = impute_missing_values(df)
    df = truncate_extreme_values(df)
    df = convert_categorical_to_numerical(df)
    df = handle_bad_data(df)
    return df


# Environment Variables
TRAIN_PATH = "input/train.csv"
TEST_PATH = "input/test.csv"
MACRO_PATH = "input/macro.csv"
TRAIN_LAT_LON_PATH = "input/train_lat_lon.csv"
TEST_LAT_LON_PATH = "input/test_lat_lon.csv"
TRAIN_OUT_PATH = "output/train.csv"
TEST_OUT_PATH = "output/test.csv"

df_train = pd.read_csv(TRAIN_PATH)
df_test = pd.read_csv(TEST_PATH)
df_train_lat_lon = pd.read_csv(
    TRAIN_LAT_LON_PATH, usecols=["id", "lat", "lon"], index_col="id"
).sort_index()
df_test_lat_lon = pd.read_csv(
    TRAIN_LAT_LON_PATH, usecols=["id", "lat", "lon"], index_col="id"
).sort_index()

# Clean
df_train_clean = clean(df_train, df_train_lat_lon)
df_test_clean = clean(df_train, df_test_lat_lon)
df_train_clean.to_csv(TRAIN_OUT_PATH)
df_test_clean.to_csv(TEST_OUT_PATH)
