# Databricks notebook source
num_df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .option('format','delta')
    .load("dbfs:/user/hive/warehouse/bronzes.db/numbers")
)

sub_df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .option('format','delta')
    .load("dbfs:/user/hive/warehouse/bronzes.db/submissions")
)

# COMMAND ----------

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when, count, percentile_approx

def impute_num_value_medians(df_num: DataFrame, df_sub: DataFrame) -> DataFrame:
    """
    Impute nulls in the NUM DataFrame’s value column by:
      1. Joining to SUB on adsh to bring in cik
      2. Grouping by (cik, tag, uom) to compute:
         - grp_count  : number of non-null values per group
         - grp_median : median of value per group
      3. Flagging groups with no non-null values as is_incomplete
      4. Filling null value entries with their group’s median where available

    Parameters:
        df_num (DataFrame): Cleaned NUM DataFrame containing adsh, tag, uom, and value.
        df_sub (DataFrame): Cleaned SUB DataFrame containing adsh and cik.

    Returns:
        DataFrame: A new DataFrame with:
          - value nulls imputed by group median
          - is_incomplete flag set True for rows in groups with zero non-null values
    """
    # 1) Join NUM → SUB to get CIK
    num_joined = df_num.join(
        df_sub.select("adsh", "cik"),
        on="adsh",
        how="left"
    )

    # 2) Compute per-group non-null count & median
    group_cols = ["cik", "tag", "uom"]
    group_stats = (
        num_joined
          .filter(col("value").isNotNull())
          .groupBy(*group_cols)
          .agg(
              count(col("value")).alias("grp_count"),
              percentile_approx(col("value"), 0.5).alias("grp_median")
          )
    )

    # 3) Join stats back
    df_with_stats = num_joined.join(group_stats, on=group_cols, how="left")

    # 4) Flag empty groups & impute medians
    df_result = (
        df_with_stats
          .withColumn(
              "is_value_incomplete",
              when(col("grp_count").isNull(), True).otherwise(False)
          )
          .withColumn(
              "value",
              when(
                  col("value").isNull() & col("grp_count").isNotNull(),
                  col("grp_median")
              )
              .otherwise(col("value"))
          )
          .drop("grp_count", "grp_median")
    )

    df_result = df_result.drop("cik")

    return df_result

# COMMAND ----------

num_df = impute_num_value_medians(num_df, sub_df)
display(num_df)

# COMMAND ----------

from pyspark.sql.functions import *

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

    num_df = num_df.filter(col("adsh").isNotNull() & col("tag").isNotNull() & col("version").isNotNull())

    num_df = num_df.withColumn("version", upper("version"))

    num_df=num_df.where(col("qtrs") < 5)
    num_df=num_df.select("adsh", "tag", "version", "ddate", "qtrs", "uom", "value", "is_value_incomplete")
    return num_df

# COMMAND ----------

display(num_df)


# COMMAND ----------

num_df=num_transform(num_df)
display(num_df.limit(20))

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS silver.numbers;

# COMMAND ----------

num_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("silver.numbers")
num_df_loaded = spark.read.format("delta").table("silver.numbers")
display(num_df_loaded)
