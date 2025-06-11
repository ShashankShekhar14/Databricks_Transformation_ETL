# Databricks notebook source
sub_df = spark.read.option("header", True) \
                   .option("delimiter", "\t") \
                   .option('format','delta')\
                   .load("dbfs:/user/hive/warehouse/bronzes.db/submissions")

# sub_df=sub_transform(sub_df)
# Display the first 20 rows
display(sub_df.limit(20))

# COMMAND ----------

from pyspark.sql import DataFrame, Window
from pyspark.sql.functions import col, when, row_number

def impute_sub_categorical_modes(df_sub: DataFrame) -> DataFrame:
    """
    Imputes nulls in the 'sic' and 'countryba' columns of a SUB DataFrame by:
      1. Per-CIK mode (most frequent) fill
      2. Global mode fallback for any remaining nulls

    Parameters:
        df_sub (DataFrame): Cleaned SUB DataFrame containing at least 'cik', 'sic', and 'countryba'.

    Returns:
        DataFrame: A new DataFrame with no nulls in 'sic' or 'countryba'.
    """
    df = df_sub
    to_impute = ["sic", "countryba"]

    # 1) Per-CIK mode fill
    for c in to_impute:
        mode_per_cik = (
            df
              .filter(col(c).isNotNull())
              .groupBy("cik", c)
              .count()
              .withColumn(
                  "rn",
                  row_number().over(
                      Window.partitionBy("cik")
                            .orderBy(col("count").desc())
                  )
              )
              .filter(col("rn") == 1)
              .select("cik", col(c).alias(f"{c}_cik_mode"))
        )
        df = (
            df
              .join(mode_per_cik, on="cik", how="left")
              .withColumn(
                  c,
                  when(col(c).isNull(), col(f"{c}_cik_mode"))
                  .otherwise(col(c))
              )
              .drop(f"{c}_cik_mode")
        )

    # 2) Global mode fallback for any remaining nulls
    global_modes = {}
    for c in to_impute:
        mode_val = (
            df
              .filter(col(c).isNotNull())
              .groupBy(c)
              .count()
              .orderBy(col("count").desc())
              .limit(1)
              .collect()[0][c]
        )
        global_modes[c] = mode_val

    # 3) Apply global fallback
    for c, mode_val in global_modes.items():
        df = df.withColumn(
            c,
            when(col(c).isNull(), mode_val)
            .otherwise(col(c))
        )

    return df

# COMMAND ----------

sub_df = impute_sub_categorical_modes(sub_df)
display(sub_df.limit(20))

# COMMAND ----------

from pyspark.sql import DataFrame, Window
from pyspark.sql.functions import (
    col, when, row_number,
    percentile_approx, to_date,
    year, lit
)

def impute_sub_date_fields(df_sub: DataFrame) -> DataFrame:
    """
    Impute nulls in the SUB DataFrame for the following columns:
      - fy, fp       : categorical (mode per CIK, then fallback)
      - period, fye  : numeric dates (median per CIK, then fallback)

    Parameters:
        df_sub (DataFrame): Cleaned SUB DataFrame containing at least
                            'cik', 'filed', 'fy', 'fp', 'period', and 'fye'.

    Returns:
        DataFrame: New DataFrame with no nulls in 'fy', 'fp', 'period', or 'fye'.
    """
    df = df_sub

    # Convert 'filed' (int YYYYMMDD) to date for extracting year
    df = df.withColumn(
        "filed_dt",
        to_date(col("filed").cast("string"), "yyyyMMdd")
    )

    # 1) Impute categorical 'fy' and 'fp' by mode per CIK
    for c in ["fy", "fp"]:
        mode_df = (
            df.filter(col(c).isNotNull())
              .groupBy("cik", c)
              .count()
              .withColumn(
                  "rn",
                  row_number().over(
                      Window.partitionBy("cik")
                            .orderBy(col("count").desc())
                  )
              )
              .filter(col("rn") == 1)
              .select("cik", col(c).alias(f"{c}_mode"))
        )
        df = (
            df.join(mode_df, on="cik", how="left")
              .withColumn(
                  c,
                  when(col(c).isNull(), col(f"{c}_mode"))
                  .otherwise(col(c))
              )
              .drop(f"{c}_mode")
        )

    # 2) Impute numeric 'period' and 'fye' by median per CIK
    for c in ["period", "fye"]:
        med_df = (
            df.filter(col(c).isNotNull())
              .groupBy("cik")
              .agg(
                  percentile_approx(col(c), 0.5).alias(f"{c}_med")
              )
        )
        df = (
            df.join(med_df, on="cik", how="left")
              .withColumn(
                  c,
                  when(col(c).isNull(), col(f"{c}_med"))
                  .otherwise(col(c))
              )
              .drop(f"{c}_med")
        )

    # 3) Final fallback defaults
    df = (
        df
          .withColumn("fye",
              when(col("fye").isNull(), lit(1231))
              .otherwise(col("fye"))
          )
          .withColumn("fy",
              when(col("fy").isNull(), year(col("filed_dt")))
              .otherwise(col("fy"))
          )
          .withColumn("period",
              when(col("period").isNull(), col("fy") * 10000 + col("fye"))
              .otherwise(col("period"))
          )
          .withColumn("fp",
              when(col("fp").isNull(), lit("FY"))
              .otherwise(col("fp"))
          )
          .drop("filed_dt")
    )

    return df

# COMMAND ----------

sub_df = impute_sub_date_fields(sub_df)
display(sub_df)

# COMMAND ----------

from pyspark.sql.functions import col, to_date, regexp_replace, trim, datediff

def sub_transform(sub_df):
    sub_df = sub_df.filter(col("adsh").isNotNull() & col("cik").isNotNull() & col("name").isNotNull() & col("sic").isNotNull())

    sub_df = sub_df.drop("bas1", "bas2", "countryma", "stprma", "cityma", "zipma", "mas1", "mas2", "countryinc", "stprinc", "ein", "former", "changed", "afs", "wksi", "prevrpt", "detail", "nciks", "aciks")

    sub_df = sub_df.withColumn("period", to_date("period", "yyyyMMdd"))
    sub_df = sub_df.withColumn("filed", to_date("filed", "yyyyMMdd"))
    sub_df = sub_df.drop("year","quarter")
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

sub_df=sub_transform(sub_df)
# Display the first 20 rows
display(sub_df.limit(20))

# COMMAND ----------

# from pyspark.sql.functions import col

# # Check if nulls are present in either 'sic' or 'countryba'
# sub_df.filter(col("sic").isNull() | col("countryba").isNull()).show()

# COMMAND ----------

# from pyspark.sql.functions import col

# # Check if nulls are present in either 'sic' or 'countryba'
# sub_df.filter(col("fye").isNull() | col("fp").isNull() | col("fy").isNull() | col("period").isNull()).display()

# COMMAND ----------

# MAGIC %sql
# MAGIC drop table if exists silver.submissions;

# COMMAND ----------

sub_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("silver.submissions")
sub_df_loaded = spark.read.format("delta").table("silver.submissions")
display(sub_df_loaded)
