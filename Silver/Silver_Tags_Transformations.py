# Databricks notebook source
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number


# COMMAND ----------

def tag_transform(tag_df):
    tag_df = tag_df.filter(tag_df["version"].isNotNull())
    tag_df = tag_df.drop("year","quarter")
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

tag_df = (
    spark.read
    .option("header", "true")         
    .option("inferSchema", "true")    
    .load("dbfs:/user/hive/warehouse/bronzes.db/tags")
)

tag_df=tag_transform(tag_df)
display(tag_df.limit(20))

# COMMAND ----------

# MAGIC %sql
# MAGIC drop table if exists silver.tags

# COMMAND ----------

tag_df.write.format("delta").mode("overwrite").saveAsTable("silver.tags")
tag_df_loaded = spark.read.format("delta").table("silver.tags")
display(tag_df_loaded)
