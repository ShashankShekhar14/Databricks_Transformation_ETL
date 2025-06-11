# Databricks notebook source
pip install pytest chispa

# COMMAND ----------

import pytest
from decimal import Decimal
from datetime import date
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import (
    StructType, StructField, StringType, DateType,
    IntegerType, DecimalType
)
from chispa.dataframe_comparer import assert_df_equality

# COMMAND ----------

def num_transform(num_df):
    num_df = num_df.drop("segments", "coreg", "footnote")

    num_df = num_df.filter(col("value").isNotNull())
    
    month = substring(col("ddate"), 5, 2)
    day = substring(col("ddate"), 7, 2)
    original_year = substring(col("ddate"), 1, 4)
    original_year_int = original_year.cast("int")
    corrected_year = when(
    (original_year_int >= 2009) & (original_year_int <= year(current_date())),original_year).otherwise(col("year").cast("string"))

    num_df = num_df.withColumn("ddate",to_date(concat_ws("", corrected_year, month, day), "yyyyMMdd"))

    num_df = num_df.filter(col("ddate").isNotNull())

    num_df = num_df.filter(col("adsh").isNotNull() & col("tag").isNotNull() & col("version").isNotNull())

    num_df = num_df.withColumn("version", upper("version"))
    num_df = num_df.drop("year", "quarter")
    return num_df  

# COMMAND ----------

sample_data = [
    ("0001", "Tag1", "us-gaap", "20320101", 1, "USD", "seg1", "coreg1", Decimal("100.0000"), "note1", 2022, 1),
    ("0002", "Tag2", "ifrs", "19920102", 2, "EUR", None, None, Decimal("200.0000"), None, 2022, 1),
    (None, "Tag3", "us-gaap", "20220103", 3, "USD", "seg2", "coreg2", Decimal("300.0000"), "note2", 2022, 1),
    ("0004", None, "ifrs", "20220104", 4, "EUR", "seg3", "coreg3", None, "note3", 2022, 1),
    ("0005", "Tag5", None, "19970105", 5, "USD", "seg4", "coreg4", Decimal("500.0000"), "note4", 2022, 1)
]

expected_data = [
    ("0001", "Tag1", "US-GAAP", date(2022, 1, 1), 1, "USD", Decimal("100.0000")),
    ("0002", "Tag2", "IFRS", date(2022, 1, 2), 2, "EUR", Decimal("200.0000"))
]

schema = StructType([
    StructField("adsh", StringType(), True),
    StructField("tag", StringType(), True),
    StructField("version", StringType(), True),
    StructField("ddate", StringType(), True),
    StructField("qtrs", IntegerType(), True),
    StructField("uom", StringType(), True),
    StructField("segments", StringType(), True),
    StructField("coreg", StringType(), True),
    StructField("value", DecimalType(28, 4), True),
    StructField("footnote", StringType(), True),
    StructField("year", IntegerType(), True),
    StructField("quarter", IntegerType(), True)
])

expected_schema = StructType([
    StructField("adsh", StringType(), True),
    StructField("tag", StringType(), True),
    StructField("version", StringType(), True),
    StructField("ddate", DateType(), True),
    StructField("qtrs", IntegerType(), True),
    StructField("uom", StringType(), True),
    StructField("value", DecimalType(28, 4), True)
])

# COMMAND ----------

def spark():
    return SparkSession.builder.appName("UnitTesting").getOrCreate()

# Basic transformation test
def test_basic_transformation(spark):
    print("\nRunning basic transformation test...")

    input_df = spark.createDataFrame(sample_data, schema)
    expected_df = spark.createDataFrame(expected_data, expected_schema)
    transformed_df = num_transform(input_df)

    try:
        assert_df_equality(transformed_df, expected_df, ignore_nullable=True)
        print("✅ Basic transformation test passed")
    except AssertionError as e:
        print("❌ Basic transformation test failed")
        raise e

# Null value filtering
def test_null_value_filtering(spark):
    print("\nRunning null value filtering test...")

    test_data = [("0001", "Tag1", "us-gaap", "20220101", 1, "USD", None, None, None, None, 2022, 1)]
    input_df = spark.createDataFrame(test_data, schema)
    transformed_df = num_transform(input_df)

    try:
        assert transformed_df.count() == 0
        print("✅ Null value filtering test passed")
    except AssertionError:
        print("❌ Null value filtering test failed")
        raise

# Required fields filtering
def test_required_fields_filtering(spark):
    print("\nRunning required fields filtering test...")

    test_data = [
        (None, "Tag1", "us-gaap", "20220101", 1, "USD", None, None, Decimal("100.0"), None, 2022, 1),
        ("0002", None, "us-gaap", "20220101", 1, "USD", None, None, Decimal("100.0"), None, 2022, 1),
        ("0003", "Tag3", None, "20220101", 1, "USD", None, None, Decimal("100.0"), None, 2022, 1)
    ]
    input_df = spark.createDataFrame(test_data, schema)
    transformed_df = num_transform(input_df)

    try:
        assert transformed_df.count() == 0
        print("✅ Required fields filtering test passed")
    except AssertionError:
        print("❌ Required fields filtering test failed")
        raise

# Version to uppercase test
def test_version_uppercase(spark):
    print("\nRunning version uppercase test...")

    test_data = [("0001", "Tag1", "us-gaap", "20220101", 1, "USD", None, None, Decimal("100.0"), None, 2022, 1)]
    input_df = spark.createDataFrame(test_data, schema)
    transformed_df = num_transform(input_df)

    try:
        assert transformed_df.first()["version"] == "US-GAAP"
        print("✅ Version uppercase test passed")
    except AssertionError:
        print("❌ Version uppercase test failed")
        raise

# Date conversion test
def test_date_conversion(spark):
    print("\nRunning date conversion test...")

    test_data = [("0001", "Tag1", "us-gaap", "20220101", 1, "USD", None, None, Decimal("100.0"), None, 2022, 1)]
    input_df = spark.createDataFrame(test_data, schema)
    transformed_df = num_transform(input_df)

    try:
        assert str(transformed_df.first()["ddate"]) == "2022-01-01"
        print("✅ Date conversion test passed")
    except AssertionError:
        print("❌ Date conversion test failed")
        raise



# COMMAND ----------

spark_session = spark()
test_basic_transformation(spark_session)
test_null_value_filtering(spark_session)
test_required_fields_filtering(spark_session)
test_version_uppercase(spark_session)
test_date_conversion(spark_session)
print("\nAll tests completed!")
