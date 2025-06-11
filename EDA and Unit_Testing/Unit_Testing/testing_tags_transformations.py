# Databricks notebook source
pip install pytest chispa

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.window import Window
import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from chispa.dataframe_comparer import assert_df_equality

# COMMAND ----------

def tag_transform(tag_df):
    tag_df = tag_df.filter(tag_df["version"].isNotNull())

    tag_df = tag_df.withColumn("version", upper("version"))

    # Define a dummy window over the entire dataset (no partitioning)
    windowSpec = Window.orderBy("tag", "version")

    # Add surrogate key column
    tag_df = tag_df.withColumn("tag_id", row_number().over(windowSpec))

    tag_df = tag_df.where("abstract != 1")

    tag_df = tag_df.drop("custom", "abstract", "crdr")

    # Get all columns except tag_id
    cols = [c for c in tag_df.columns if c != "tag_id"]

    # Reorder columns with tag_id first
    tag_df = tag_df.select(["tag_id"] + cols)

    return tag_df

# COMMAND ----------

schema = StructType([
    StructField("tag", StringType(), True),
    StructField("version", StringType(), True),
    StructField("custom", IntegerType(), True),
    StructField("abstract", BooleanType(), True),
    StructField("datatype", StringType(), True),
    StructField("iord", StringType(), True),
    StructField("crdr", StringType(), True),
    StructField("tlabel", StringType(), True),
    StructField("doc", StringType(), True),
    StructField("year", IntegerType(), True),
    StructField("quarter", IntegerType(), True),
])

expected_schema = StructType([
    StructField("tag_id", IntegerType(), True),
    StructField("tag", StringType(), True),
    StructField("version", StringType(), True),
    StructField("datatype", StringType(), True),
    StructField("iord", StringType(), True),
    StructField("tlabel", StringType(), True),
    StructField("doc", StringType(), True),
    StructField("year", IntegerType(), True),
    StructField("quarter", IntegerType(), True),
])
sample_data = [
    # Valid row – should be included
    ("Rev", "us-gaap", 0, False, "monetary", "I", "CR", "Revenue", "Revenue line", 2024, 1),

    # Null version – should be excluded
    ("Loss", None, 1, False, "monetary", "I", "DR", "Net Loss", "Loss line", 2024, 1),

    # Abstract is True – should be excluded
    ("Asset", "ifrs", 0, True, "monetary", "I", "CR", "Asset Value", "Assets line", 2024, 1),

    # Extra valid row with lowercase version
    ("Equity", "ifrs", 0, False, "monetary", "C", "DR", "Equity Val", "Equity line", 2024, 1),

    # Duplicate version-tag with different iord/tlabel
    ("Rev", "us-gaap", 0, False, "monetary", "A", "CR", "Revenue Alt", "Revenue Alt line", 2024, 1),
]

expected_data = [
    (1, "Equity", "IFRS", "monetary", "C", "Equity Val", "Equity line", 2024, 1),
    (2, "Rev", "US-GAAP", "monetary", "A", "Revenue Alt", "Revenue Alt line", 2024, 1),
    (3, "Rev", "US-GAAP", "monetary", "I", "Revenue", "Revenue line", 2024, 1),
]

# COMMAND ----------

def spark():
    return SparkSession.builder.appName("TagTransformTests").getOrCreate()

def test_tag_transformation_basic(spark):
    print("\nRunning basic tag transformation test...")
    input_df = spark.createDataFrame(sample_data, schema=schema)
    expected_df = spark.createDataFrame(expected_data, schema=expected_schema)
    result_df = tag_transform(input_df)

    try:
        # Ignore surrogate key value, just check the rest match
        assert_df_equality(result_df.drop("tag_id"), expected_df.drop("tag_id"), ignore_row_order=True)
        print("✅ Basic transformation test passed")
    except AssertionError as e:
        print("❌ Basic transformation test failed")
        raise e

def test_null_version_filtered(spark):
    print("\nRunning version null filtering test...")
    df = spark.createDataFrame(sample_data, schema=schema)
    result_df = tag_transform(df)

    try:
        assert result_df.filter(col("version").isNull()).count() == 0
        print("✅ Null version filtering test passed")
    except AssertionError:
        print("❌ Null version filtering test failed")
        raise

def test_version_uppercase(spark):
    print("\nRunning version uppercasing test...")
    df = spark.createDataFrame(sample_data, schema=schema)
    result_df = tag_transform(df)

    try:
        versions = [row["version"] for row in result_df.select("version").collect()]
        assert all(v == v.upper() for v in versions)
        print("✅ Version uppercasing test passed")
    except AssertionError:
        print("❌ Version uppercasing test failed")
        raise


# COMMAND ----------

spark_session = spark()
test_tag_transformation_basic(spark_session)
test_null_version_filtered(spark_session)
test_version_uppercase(spark_session)
print("\n✅ All `tag_transform` tests completed")
