# Databricks notebook source
pip install pytest chispa

# COMMAND ----------

import pytest
from datetime import date, datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, LongType,
    DateType, BooleanType, TimestampType
)
from chispa.dataframe_comparer import assert_df_equality

# COMMAND ----------

def sub_transform(sub_df):
    sub_df = sub_df.filter(col("adsh").isNotNull() & col("cik").isNotNull() & col("name").isNotNull() & col("sic").isNotNull())

    sub_df = sub_df.drop("bas1", "bas2", "countryma", "stprma", "cityma", "zipma", "mas1", "mas2", "countryinc", "stprinc", "ein", "former", "changed", "afs", "wksi", "prevrpt", "detail", "nciks", "aciks")

    sub_df = sub_df.withColumn("period", to_date("period", "yyyyMMdd"))
    sub_df = sub_df.withColumn("filed", to_date("filed", "yyyyMMdd"))

    sub_df = sub_df.withColumn("baph", regexp_replace("baph", "[^0-9]", ""))

    columns_to_trim = ["name", "countryba", "stprba", "cityba", "zipba", "form", "instance"]
    for col_name in columns_to_trim:
        sub_df = sub_df.withColumn(col_name, trim(col_name))
    
    sub_df=sub_df.withColumn("delay_days", datediff(col("filed"), col("period")))

    sub_df = sub_df.select(
        "adsh", "cik", "name", "sic", "countryba", "stprba", "cityba", "zipba", "baph",
        "fye", "form", "period", "fy", "fp", "filed", "delay_days", "accepted", "instance"
    )
    
    return sub_df

# COMMAND ----------

schema = StructType([
    StructField("adsh", StringType(), True),
    StructField("cik", LongType(), True),
    StructField("name", StringType(), True),
    StructField("sic", IntegerType(), True),
    StructField("countryba", StringType(), True),
    StructField("stprba", StringType(), True),
    StructField("cityba", StringType(), True),
    StructField("zipba", StringType(), True),
    StructField("bas1", StringType(), True),
    StructField("bas2", StringType(), True),
    StructField("baph", StringType(), True),
    StructField("countryma", StringType(), True),
    StructField("stprma", StringType(), True),
    StructField("cityma", StringType(), True),
    StructField("zipma", StringType(), True),
    StructField("mas1", StringType(), True),
    StructField("mas2", StringType(), True),
    StructField("countryinc", StringType(), True),
    StructField("stprinc", StringType(), True),
    StructField("ein", StringType(), True),
    StructField("former", StringType(), True),
    StructField("changed", DateType(), True),
    StructField("afs", StringType(), True),
    StructField("wksi", BooleanType(), True),
    StructField("fye", StringType(), True),
    StructField("form", StringType(), True),
    StructField("period", StringType(), True),
    StructField("fy", IntegerType(), True),
    StructField("fp", StringType(), True),
    StructField("filed", StringType(), True),
    StructField("accepted", TimestampType(), True),
    StructField("prevrpt", BooleanType(), True),
    StructField("detail", BooleanType(), True),
    StructField("instance", StringType(), True),
    StructField("nciks", IntegerType(), True),
    StructField("aciks", StringType(), True),
    StructField("year", IntegerType(), True),
    StructField("quarter", IntegerType(), True)
])

expected_schema = StructType([
    StructField("adsh", StringType(), True),
    StructField("cik", LongType(), True),
    StructField("name", StringType(), True),
    StructField("sic", IntegerType(), True),
    StructField("countryba", StringType(), True),
    StructField("stprba", StringType(), True),
    StructField("cityba", StringType(), True),
    StructField("zipba", StringType(), True),
    StructField("baph", StringType(), True),
    StructField("fye", StringType(), True),
    StructField("form", StringType(), True),
    StructField("period", DateType(), True),
    StructField("fy", IntegerType(), True),
    StructField("fp", StringType(), True),
    StructField("filed", DateType(), True),
    StructField("delay_days", IntegerType(), True),
    StructField("accepted", TimestampType(), True),
    StructField("instance", StringType(), True)
])

sample_data = [
    # Valid input
    ("0001", 1000001, " Test Company ", 1234, " USA ", " CA ", " San Francisco ", "94105",
     "addr1", "addr2", "(123) 456-7890",
     "USA", "CA", "NYC", "10001", "HQ", "HQ2", "USA", "CA", "12-3456789", "Former Co",
     date(2020, 1, 1), "AFS", True, "1231", " 10-K ", "20240101", 2024, "FY", "20240131",
     datetime(2024, 2, 1, 10, 0), False, True, " Instance_1 ", 5, "ACIKS", 2024, 1),

    # Invalid input (null adsh) -> should be filtered
    (None, 1000002, "Another Co", 1235, "USA", "NY", "New York", "10002", "111-222", "", "", "", "", "", "", "", "", "", "", "", "", None,
     "AFS", True, "1231", "10-K", "20240101", 2024, "FY", "20240131", datetime(2024, 2, 1, 10, 0), False, True, "Instance_2", 5, "ACIKS", 2024, 1),

    # Valid input with already clean data (no trimming needed)
    ("0003", 1000003, "CleanCo", 5678, "USA", "TX", "Austin", "73301","","", "9999999999", "", "", "", "", "", "", "", "","", "", date(2021, 5, 10),
 "AFS", False, "0630", "10-Q", "20230630", 2023, "Q2", "20230715", datetime(2023, 7, 20, 9, 0), False, False, "Instance_3", 3, "", 2023, 2),
]

expected_data = [
    ("0001", 1000001, "Test Company", 1234, "USA", "CA", "San Francisco", "94105", "1234567890",
     "1231", "10-K", date(2024, 1, 1), 2024, "FY", date(2024, 1, 31), 30, datetime(2024, 2, 1, 10, 0), "Instance_1"),

    ("0003", 1000003, "CleanCo", 5678, "USA", "TX", "Austin", "73301", "9999999999",
     "0630", "10-Q", date(2023, 6, 30), 2023, "Q2", date(2023, 7, 15), 15, datetime(2023, 7, 20, 9, 0), "Instance_3")
    
    
]

# COMMAND ----------


def spark():
    return SparkSession.builder.appName("SubTransformTests").getOrCreate()

def test_sub_transformation_basic(spark):
    print("\nRunning basic sub transformation test...")
    input_df = spark.createDataFrame(sample_data, schema=schema)
    expected_df = spark.createDataFrame(expected_data, schema=expected_schema)

    result_df = sub_transform(input_df)

    try:
        assert_df_equality(result_df, expected_df, ignore_nullable=True)
        print("✅ Basic transformation test passed")
    except AssertionError as e:
        print("❌ Basic transformation test failed")
        raise e


def test_required_columns_filtering(spark):
    print("\nRunning required columns filtering test...")
    test_data = [  # missing adsh and name
        (None, 1000003, None, None, "USA", "TX", "Austin", "73301", None, "", "", "", "", "", "", "", "", "", "", "", "", None,
         "AFS", True, "1231", "10-K", "20240101", 2024, "FY", "20240131", datetime(2024, 2, 1), False, True, "Instance", 1, "ACIKS", 2024, 1)
    ]
    df = spark.createDataFrame(test_data, schema=schema)
    result_df = sub_transform(df)

    try:
        assert result_df.count() == 0
        print("✅ Required columns filtering test passed")
    except AssertionError:
        print("❌ Required columns filtering test failed")
        raise


def test_delay_days_calculation(spark):
    print("\nRunning delay days calculation test...")
    test_data = [
        ("0002", 1000002, "Company B", 9999, "USA", "IL", "Chicago", "60601", "123-999-4567", "", "", "", "", "", "", "", "", "", "", "", "", None,
         "AFS", False, "0630", "10-Q", "20240101", 2024, "Q1", "20240120", datetime(2024, 1, 21), False, False, "Instance_2", 5, "ACIKS", 2024, 1)
    ]
    df = spark.createDataFrame(test_data, schema=schema)
    result_df = sub_transform(df)

    try:
        assert result_df.first()["delay_days"] == 19
        print("✅ Delay days calculation test passed")
    except AssertionError:
        print("❌ Delay days calculation test failed")
        raise



# COMMAND ----------

spark_session = spark()
test_sub_transformation_basic(spark_session)
test_required_columns_filtering(spark_session)
test_delay_days_calculation(spark_session)
print("\nAll tests completed ✅")
