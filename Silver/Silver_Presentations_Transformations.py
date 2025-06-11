# Databricks notebook source
from pyspark.sql.functions import *

# COMMAND ----------

pre_df = spark.read.option("header", True) \
                   .option("delimiter", "\t") \
                   .option("inferSchema", True) \
                   .option('format','delta')\
                   .load("dbfs:/user/hive/warehouse/bronzes.db/presentations")

# COMMAND ----------

from pyspark.sql import DataFrame, Window
from pyspark.sql.functions import col, when, row_number

def impute_pre_stmt(df_pre: DataFrame) -> DataFrame:
    """
    Impute nulls in the PRE DataFrame’s stmt column by:
      1. Computing each tag’s most frequent statement type (mode) across filings.
      2. Filling null stmt values with that per-tag mode.
      3. As a final fallback, assigning 'UN' (Unclassifiable) to any remaining nulls.

    Parameters:
        df_pre (DataFrame): Cleaned PRE DataFrame containing at least 'tag' and 'stmt'.

    Returns:
        DataFrame: A new DataFrame with no nulls in stmt.
    """
    # 1) Build per-tag mode lookup for stmt
    tag_mode_stmt = (
        df_pre
          .filter(col("stmt").isNotNull())
          .groupBy("tag", "stmt")
          .count()
          .withColumn(
              "rn",
              row_number().over(
                  Window.partitionBy("tag")
                        .orderBy(col("count").desc())
              )
          )
          .filter(col("rn") == 1)
          .select("tag", col("stmt").alias("stmt_mode"))
    )

    # 2) Left-join and fill with per-tag mode
    df_filled = (
        df_pre
          .join(tag_mode_stmt, on="tag", how="left")
          .withColumn(
              "stmt",
              when(col("stmt").isNull(), col("stmt_mode"))
              .otherwise(col("stmt"))
          )
          .drop("stmt_mode")
    )

    # 3) Final fallback: assign 'UN' to any still-null stmt
    df_result = df_filled.withColumn(
        "stmt",
        when(col("stmt").isNull(), "UN")
        .otherwise(col("stmt"))
    )

    return df_result

# COMMAND ----------

pre_df=impute_pre_stmt(pre_df)
display(pre_df)

# COMMAND ----------

# pre_df.filter(col("stmt").isNull()).count()

# COMMAND ----------

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when, regexp_replace, initcap

def impute_plabel_from_tag(df_pre: DataFrame) -> DataFrame:
    """
    Fill null 'plabel' values in a PRE DataFrame by deriving a label from the 'tag':
      - Inserts spaces before capital letters that follow lowercase letters
      - Converts the result to title case

    Parameters:
        df_pre (DataFrame): Cleaned PRE DataFrame containing 'tag' and 'plabel'.

    Returns:
        DataFrame: A new DataFrame with no nulls in 'plabel'.
    """
    return (
        df_pre
          .withColumn(
              "plabel",
              when(
                  col("plabel").isNull(),
                  initcap(
                      regexp_replace(col("tag"), "([a-z])([A-Z])", "$1 $2")
                  )
              )
              .otherwise(col("plabel"))
          )
    )

# COMMAND ----------

pre_df=impute_plabel_from_tag(pre_df)
display(pre_df)

# COMMAND ----------

pre_df.filter(col("plabel").isNull()).count()

# COMMAND ----------

def pre_transform(pre_df):
    pre_df=pre_df.drop("inpth", "rfile", "negating")
    pre_df=pre_df.filter(col("stmt").isNotNull())
    pre_df = pre_df.drop("year","quarter")
    columns_to_trim = ["adsh", "stmt", "tag", "version", "plabel"]
    for col_name in columns_to_trim:
        pre_df = pre_df.withColumn(col_name, trim(col_name))
    return pre_df

# COMMAND ----------

pre_df=pre_transform(pre_df)
# Display the first 20 rows
display(pre_df.limit(20))

# COMMAND ----------

pre_df = pre_df.select("adsh", "report", "line", "stmt", "tag", "version", "plabel")
display(pre_df.limit(20))

# COMMAND ----------

# MAGIC %sql
# MAGIC drop table if exists silver.presentations;

# COMMAND ----------

pre_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("silver.presentations")
pre_df_loaded = spark.read.format("delta").table("silver.presentations")
display(pre_df_loaded)
