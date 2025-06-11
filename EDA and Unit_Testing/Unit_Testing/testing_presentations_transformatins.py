# Databricks notebook source
pip install pytest chispa

# COMMAND ----------

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
from chispa.dataframe_comparer import assert_df_equality

# COMMAND ----------

def pre_transform(pre_df):
    pre_df=pre_df.drop("inpth", "rfile", "negating")
    pre_df=pre_df.filter(col("stmt").isNotNull())
    
    columns_to_trim = ["adsh", "stmt", "tag", "version", "plabel"]
    for col_name in columns_to_trim:
        pre_df = pre_df.withColumn(col_name, trim(col_name))
    return pre_df

# COMMAND ----------

schema = StructType([
    StructField("adsh", StringType(), True),
    StructField("report", IntegerType(), True),
    StructField("line", IntegerType(), True),
    StructField("stmt", StringType(), True),
    StructField("inpth", BooleanType(), True),
    StructField("rfile", StringType(), True),
    StructField("tag", StringType(), True),
    StructField("version", StringType(), True),
    StructField("plabel", StringType(), True),
    StructField("negating", StringType(), True),
    StructField("year", IntegerType(), True),
    StructField("quarter", IntegerType(), True),
])

expected_schema = StructType([
    StructField("adsh", StringType(), True),
    StructField("report", IntegerType(), True),
    StructField("line", IntegerType(), True),
    StructField("stmt", StringType(), True),
    StructField("tag", StringType(), True),
    StructField("version", StringType(), True),
    StructField("plabel", StringType(), True),
    StructField("year", IntegerType(), True),
    StructField("quarter", IntegerType(), True),
])

sample_data = [
    (" 0001 ", 1, 10, " BS ", True, "R1", " Tag1 ", " Ver1 ", " Label1 ", "N", 2024, 1),   # Valid, needs trim
    ("0002", 2, 20, None, False, "R2", "Tag2", "Ver2", "Label2", "Y", 2024, 1),            # Invalid: stmt is null
    ("0003", 3, 30, "IS", True, "R3", " Tag3 ", " ver3 ", " Label3 ", "N", 2023, 4),       # Valid, needs trim and case check
    (" 0004", 4, 40, " CF ", False, "R4", "Tag4", "Ver4", "Label4", "N", 2022, 3),         # Valid, needs trim
    ("0005", 5, 50, "EQ", True, "R5", " Tag5 ", None, " Label5 ", "Y", 2022, 2),           # Valid, version is null (keep None)
    ("0006", 6, 60, "BS", False, "R6", "Tag6", "Ver6", "Label6", None, 2022, 1),           # Valid, no trimming needed
    ("0007", 7, 70, "RE", True, "R7", "Tag7", "Ver7", "Label7", "Y", None, None),          # Valid, null year/quarter
]

expected_data = [
    ("0001", 1, 10, "BS", "Tag1", "Ver1", "Label1", 2024, 1),
    ("0003", 3, 30, "IS", "Tag3", "ver3", "Label3", 2023, 4),
    ("0004", 4, 40, "CF", "Tag4", "Ver4", "Label4", 2022, 3),
    ("0005", 5, 50, "EQ", "Tag5", None, "Label5", 2022, 2),
    ("0006", 6, 60, "BS", "Tag6", "Ver6", "Label6", 2022, 1),
    ("0007", 7, 70, "RE", "Tag7", "Ver7", "Label7", None, None),
]


# COMMAND ----------

def spark():
    return SparkSession.builder.appName("PreTransformTests").getOrCreate()

def test_pre_transformation_basic(spark):
    print("\nRunning basic pre transformation test...")
    input_df = spark.createDataFrame(sample_data, schema=schema)
    expected_df = spark.createDataFrame(expected_data, schema=expected_schema)
    result_df = pre_transform(input_df)

    try:
        assert_df_equality(result_df, expected_df, ignore_nullable=True)
        print("✅ Basic transformation test passed")
    except AssertionError as e:
        print("❌ Basic transformation test failed")
        raise e

def test_null_stmt_filtered(spark):
    print("\nRunning stmt null filtering test...")
    df = spark.createDataFrame(sample_data, schema=schema)
    result_df = pre_transform(df)

    try:
        assert result_df.filter(col("stmt").isNull()).count() == 0
        print("✅ Null stmt filtering test passed")
    except AssertionError:
        print("❌ Null stmt filtering test failed")
        raise

def test_trim_columns(spark):
    print("\nRunning whitespace trimming test...")
    df = spark.createDataFrame(sample_data, schema=schema)
    result_df = pre_transform(df)

    try:
        row = result_df.first()
        assert row["adsh"] == "0001"
        assert row["stmt"] == "BS"
        assert row["tag"] == "Tag1"
        assert row["version"] == "Ver1"
        assert row["plabel"] == "Label1"
        print("✅ Whitespace trimming test passed")
    except AssertionError:
        print("❌ Whitespace trimming test failed")
        raise


# COMMAND ----------

spark_session = spark()
test_pre_transformation_basic(spark_session)
test_null_stmt_filtered(spark_session)
test_trim_columns(spark_session)
print("\n✅ All `pre_transform` tests completed")
