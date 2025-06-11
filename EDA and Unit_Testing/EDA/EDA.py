# Databricks notebook source
# MAGIC %md
# MAGIC # **Libraries and Data**

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import round
from pyspark.sql.functions import col, count, isnan, isnull, countDistinct, sum, when

spark = SparkSession.builder.getOrCreate()

num = spark.read.option("header", True).option("delimiter", "\t").option("inferSchema", True).csv("dbfs:/FileStore/Data/2024q1/num.txt")

tag = spark.read.option("header", True).option("delimiter", "\t").option("inferSchema", True).csv("dbfs:/FileStore/Data/2024q1/tag.txt")

pre = spark.read.option("header", True).option("delimiter", "\t").option("inferSchema", True).csv("dbfs:/FileStore/Data/2024q1/pre.txt")

sub = spark.read.option("header", True).option("delimiter", "\t").option("inferSchema", True).csv("dbfs:/FileStore/Data/2024q1/sub.txt")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Defining the generate_df_info function for df summary generation

# COMMAND ----------

# MAGIC %md
# MAGIC ### Resulting DF contains
# MAGIC * **`cols`**
# MAGIC
# MAGIC   * **Description**: The name of each column in the original DataFrame.
# MAGIC
# MAGIC * **`data_types`**
# MAGIC
# MAGIC   * **Description**: The Spark‐inferred data type for that column (e.g., “string,” “int,” “timestamp”).
# MAGIC
# MAGIC * **`null_percentage`**
# MAGIC
# MAGIC   * **Description**: The percentage of rows in which that column is null (empty).
# MAGIC   * **Importance**: Shows us how incomplete a column is; if it’s mostly null, we might drop it or treat it differently in your pipeline.
# MAGIC
# MAGIC * **`distinct_count`**
# MAGIC
# MAGIC   * **Description**: The number of unique, non‐null values that column contains.
# MAGIC   * **Importance**: Helps us understand cardinality. A low distinct count often means a small set of categories (good for dimension tables).
# MAGIC
# MAGIC * **`top1_value`**
# MAGIC
# MAGIC   * **Description**: The single most frequent (non‐null) value that appears in that column.
# MAGIC   * **Importance**: Quickly tells you if one category dominates. For example, if “USP” appears 95% of the time, that column may not be very informative for analysis.
# MAGIC
# MAGIC * **`top1_value_count`**
# MAGIC
# MAGIC   * **Description**: How many times the `top1_value` appears.
# MAGIC   * **Importance**: Paired with `distinct_count`, it shows us how skewed the distribution is. A high count relative to total rows indicates low variability.
# MAGIC
# MAGIC * **`top1_value_percentage`**
# MAGIC
# MAGIC   * **Description**: The `top1_value_count` expressed as a percentage of total rows.
# MAGIC   * **Importance**: Gives immediate context for how dominant the top value is (e.g., 80% means four out of five rows share that same value, suggesting low diversity in that column).
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.types import *


def generate_df_info(df):
    """
    Given any Spark DataFrame `df`, returns a summary DataFrame with:
      - cols
      - data_types
      - null_percentage
      - distinct_count
      - top1_value
      - top1_value_count
      - top1_value_percentage
    """
    # 1. Total row count
    total_rows = df.count()

    # 2. Computing null counts per column (only isNull)
    null_exprs = [
        sum(when(col(c).isNull(), 1).otherwise(0)).alias(f"{c}_nullCount")
        for c in df.columns
    ]
    null_counts_dict = df.select(*null_exprs).collect()[0].asDict()

    # 3. Build initial rows: (col_name, dtype_str, null_pct)
    log_rows = []
    for col_name, dtype_str in df.dtypes:
        null_count = null_counts_dict[f"{col_name}_nullCount"]
        null_pct = float(null_count) * 100.0 / total_rows
        log_rows.append((col_name, dtype_str, null_pct))

    # 4. Defining schema for core df_info
    core_schema = StructType([
        StructField("cols", StringType(), nullable=False),
        StructField("data_types", StringType(), nullable=False),
        StructField("null_percentage", DoubleType(), nullable=False)
    ])
    df_info = spark.createDataFrame(log_rows, schema=core_schema)

    # 5. Computing distinct (non-null) count per column
    distinct_exprs = [
        countDistinct(col(c)).alias(f"{c}_distinctCount")
        for c in df.columns
    ]
    distinct_counts_dict = df.select(*distinct_exprs).collect()[0].asDict()

    distinct_rows = [
        (column.replace("_distinctCount", ""), distinct_counts_dict[column])
        for column in distinct_counts_dict
    ]
    distinct_schema = StructType([
        StructField("cols", StringType(), nullable=False),
        StructField("distinct_count", IntegerType(), nullable=False)
    ])
    distinct_df = spark.createDataFrame(distinct_rows, schema=distinct_schema)

    # 6. Computing top-1 value, count, and percentage for string columns
    string_cols = [name for name, dtype in df.dtypes if dtype == "string"]
    top1_rows = []
    for c in string_cols:
        top_row = (
            df.filter(col(c).isNotNull())
              .groupBy(c).count()
              .orderBy(desc("count"))
              .limit(1)
              .collect()
        )
        if top_row:
            val = top_row[0][c]
            cnt = top_row[0]["count"]
            pct = float(cnt) * 100.0 / total_rows
        else:
            val, cnt, pct = None, 0, 0.0
        top1_rows.append((c, val, cnt, pct))

    top1_schema = StructType([
        StructField("cols", StringType(), nullable=False),
        StructField("top1_value", StringType(), nullable=True),
        StructField("top1_value_count", IntegerType(), nullable=False),
        StructField("top1_value_percentage", DoubleType(), nullable=False)
    ])
    top1_df = spark.createDataFrame(top1_rows, schema=top1_schema)

    # 7. Joining everything together on "cols"
    df_info = df_info \
        .join(distinct_df, on="cols", how="left") \
        .join(top1_df, on="cols", how="left")

    return df_info

# COMMAND ----------

# MAGIC %md
# MAGIC # Understanding the Data

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Submissions DF

# COMMAND ----------

print("sub DF row count = ", sub.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ##### INSPECT SCHEMA & DATA TYPES

# COMMAND ----------

print("---- Schema of sub ----")
sub.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Sub_Info Summary table

# COMMAND ----------

sub_info = generate_df_info(sub)
sub_info.display(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC sub_file:
# MAGIC "adsh", "cik", "name", "sic", "countryba", "stprba", "cityba", "zipba", "baph",
# MAGIC "fye", "form", "period", "fy", "fp", "filed", "delay_days", "accepted", "instance"

# COMMAND ----------

# MAGIC %md
# MAGIC 1. **Column Selection for Gold**
# MAGIC
# MAGIC    * If a column’s `distinct_count` is extremely high and we only need a few metrics, so we might drop that column from the Gold table (too many distinct values can bloat your fact tables).
# MAGIC    * If a column’s `null_percentage` > 50%, might be that it’s too sparse to include as a Gold attribute and either fill it or drop it.
# MAGIC
# MAGIC 2. **Data‐Type Corrections**
# MAGIC
# MAGIC    * The `note` column already flags “Convert to datetime” or “Convert to bool.” Once converted, we can set `is_good = True` for those rows in the next run.
# MAGIC
# MAGIC 3. **Dimension Strategy**
# MAGIC
# MAGIC    * Low‐cardinality string columns (distinct\_count < 50) are good candidates for dimension tables.
# MAGIC
# MAGIC 4. **Automated Alerts**
# MAGIC
# MAGIC    * If `top1_count / total_rows` is > 90% for a column (i.e., one value dominates 90% of records), we might want to examine whether the column is actually useful or should be dropped.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Numbers DF

# COMMAND ----------

print("sub DF row count = ", num.count())

# COMMAND ----------

print("---- Schema of sub ----")
num.printSchema()

# COMMAND ----------

num_info = generate_df_info(num)
num_info.display(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC num_file:
# MAGIC adsh
# MAGIC tag
# MAGIC version
# MAGIC ddate
# MAGIC qtrs
# MAGIC uom
# MAGIC value
# MAGIC

# COMMAND ----------

num.where(col("adsh") == "0001161697-24-000084").display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### One to many connection of SUB and NUM
# MAGIC Each SUB row is like a “folder” for one filing, and that folder can contain **hundreds or even thousands** of individual numbers. For example, a single 10-K report might tag and report:
# MAGIC
# MAGIC * **Balance sheet line items**: cash, receivables, inventory, property, debt…
# MAGIC * **Income statement items**: revenue, cost of goods sold, operating expenses, net income…
# MAGIC * **Cash flow items**: operating cash flow, investing cash flow, financing cash flow…
# MAGIC * **Footnotes and segments**: interest expense by segment, tax footnotes, etc.
# MAGIC
# MAGIC Each of those tagged numbers becomes **one row** in NUM.

# COMMAND ----------

# MAGIC %md
# MAGIC #### **Quarter**

# COMMAND ----------

num.groupBy("qtrs").count().orderBy(col("count").desc()).show()

# COMMAND ----------

# MAGIC %md
# MAGIC - Values are mostly recorded as "Point-in-time"(0) or "for a year"(4)
# MAGIC - in vary rare cases we see the use of values other than 0,1,2,3 and 4 which might be an anomaly.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Units of measurement

# COMMAND ----------

num.groupBy("uom").agg(count("*").alias("count"), (count("*")*100/num.count()).alias("percentage rows covered")).orderBy(col("count").desc()).show()

# COMMAND ----------

# MAGIC %md
# MAGIC Description for each of the units 
# MAGIC
# MAGIC * **USD**: United States dollars (the standard currency unit for most financial facts).
# MAGIC * **shares**: Number of equity shares outstanding or issued.
# MAGIC * **pure**: A unitless, “pure” numeric value (no currency or other unit).
# MAGIC * **CAD**: Canadian dollars.
# MAGIC * **EUR**: Euros (the common currency of the Eurozone).
# MAGIC * **GBP**: British pounds sterling.
# MAGIC * **CNY**: Chinese yuan renminbi.
# MAGIC * **BRL**: Brazilian reais.
# MAGIC * **CHF**: Swiss francs.
# MAGIC * **CLP**: Chilean pesos.
# MAGIC * **COP**: Colombian pesos.
# MAGIC * **Rate**: A proportion or percentage rate (e.g., interest or growth rate).
# MAGIC * **AUD**: Australian dollars.
# MAGIC * **JPY**: Japanese yen.
# MAGIC * **DKK**: Danish kroner.
# MAGIC * **SEK**: Swedish kronor.
# MAGIC * **PHP**: Philippine pesos.
# MAGIC * **HKD**: Hong Kong dollars.
# MAGIC * **ILS**: Israeli new shekels.
# MAGIC * **SGD**: Singapore dollars.
# MAGIC

# COMMAND ----------

summary_table = num.summary()
summary_table.select("summary", "value").display()

# COMMAND ----------

# MAGIC %md
# MAGIC We see values in both positives and negatives, depicting that '-' negative sign is being used to measure cretid from the repective company

# COMMAND ----------

# MAGIC %md
# MAGIC ### Inferences on NUM
# MAGIC
# MAGIC 1. **Tag cardinality vs. “hot” tags**
# MAGIC
# MAGIC    * **`tag`** has **65 705** distinct values—an extremely long tail of seldom-used tags. The single most common tag, **`StockholdersEquity`**, appears \~102 770 times (3 %).
# MAGIC    * **Inference**: You’ll want to **filter** in Silver to a curated list of “Gold” tags (e.g., top 20 by frequency) rather than ingest every one of the \~65 000 tags into your main fact table.
# MAGIC
# MAGIC 4. **Quarter vs. date granularity**
# MAGIC
# MAGIC    * **`qtrs`** shows 67 distinct values—more than the expected {0,1,2,3,4}. Any other numbers are not valid under the XBRL definition
# MAGIC    * **Inference**: We have to implement a cleaning transformation step in Silver to enforce that `qtrs ∈ {0,1,2,3,4}` and that `ddate` falls on end-of-quarter.
# MAGIC
# MAGIC 5. **Unit of measure concentration**
# MAGIC
# MAGIC    * **`uom`** has 89 distinct units, but **84 %** of rows are **`USD`**; the rest include “shares,” “pure,” and dozens of other currencies.
# MAGIC    * **Inference**:
# MAGIC
# MAGIC      * **Filter** out non-USD if our client only care about dollar amounts.
# MAGIC
# MAGIC 6. **Sparse dimension columns**
# MAGIC
# MAGIC    * **`segments`** is \~45 % null.
# MAGIC    * **`coreg`** is \~99 % null (only a handful of co-registrant cases).
# MAGIC    * **`footnote`** is \~99.8 % null.
# MAGIC    * **Inference**:
# MAGIC
# MAGIC      * Treat `coreg` and `footnote` as **“detail”** fields that belong in a secondary table (e.g., only join when the user asks for footnotes).
# MAGIC
# MAGIC 7. **Value coverage and cleaning**
# MAGIC
# MAGIC    * **`value`** (cast to `double`) has a small null rate (3.6 % of rows failed to convert).
# MAGIC    * **Inference**:
# MAGIC
# MAGIC      * Add a Silver step to **clean** the `value` field (strip non-numeric characters, catch “NM” or “—” cases) before final cast.
# MAGIC      * We may Consoder dropping rows where `value` remains null after cleaning.

# COMMAND ----------

# MAGIC %md
# MAGIC ## A Look on NULLS in num

# COMMAND ----------

# MAGIC %md
# MAGIC ### Nulls vs Tags

# COMMAND ----------

# Filtering the null value rows from the df
null_vals = num.filter(col("value").isNull())

# Total number of null-value rows
total_nulls = null_vals.count()
print(f"Total null-value rows: {total_nulls}")

# TAGs have with most nulls
null_by_tag = (
    null_vals.groupBy("tag")
             .agg(count("*").alias("null_count"))
)

total_by_tag = num.groupBy("tag").agg(count("*").alias("total_count"))

tag_null_summary = (
    null_by_tag
      .join(total_by_tag, on="tag", how="inner")
      .withColumn("null_pct_of_tag", round(col("null_count") / col("total_count") * 100, 2))
      .orderBy(col("null_count").desc())
)

print("=================== Top 10 tags by # of null values ==================")
tag_null_summary.select("tag", "null_count", "total_count", "null_pct_of_tag") \
                .display(10, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Taking tags which have more than 50 percent nulls and ordering them by null count (desc)

# COMMAND ----------

tag_null_summary.select("tag", "null_count", "total_count", "null_pct_of_tag").where(col("null_pct_of_tag") >= 50).orderBy(col("null_count").desc()).display()

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### Nulls vs Units of Measure

# COMMAND ----------

null_by_uom = (
    null_vals.groupBy("uom")
             .agg(count("*").alias("null_count"))
)
total_by_uom = num.groupBy("uom").agg(count("*").alias("total_count"))

uom_null_summary = (
    null_by_uom
      .join(total_by_uom, on="uom", how="inner")
      .withColumn("null_pct_of_uom",
                  round(col("null_count") / col("total_count") * 100, 2))
      .orderBy(col("null_count").desc())
)

print("================ Top 10 UOMs by # of null values =============")
uom_null_summary.select("uom", "null_count", "total_count", "null_pct_of_uom").display(10, truncate=False)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Nulls vs Quarters

# COMMAND ----------


null_by_qtrs = (
    null_vals.groupBy("qtrs")
             .agg(count("*").alias("null_count"))
)
total_by_qtrs = num.groupBy("qtrs").agg(count("*").alias("total_count"))

qtrs_null_summary = (
    null_by_qtrs
      .join(total_by_qtrs, on="qtrs", how="inner")
      .withColumn("null_pct_of_qtrs",
                  round(col("null_count") / col("total_count") * 100, 2))
      .orderBy(col("null_count").desc())
)

print("================ Null-value breakdown by qtrs =================")
qtrs_null_summary.select("qtrs", "null_count", "total_count", "null_pct_of_qtrs").display(truncate=False)


# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Tag DF

# COMMAND ----------

print("sub DF row count = ", tag.count())

# COMMAND ----------

print("---- Schema of sub ----")
tag.printSchema()

# COMMAND ----------

tag_info = generate_df_info(tag)
tag_info.display(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC tag_file:
# MAGIC tag_id
# MAGIC tag
# MAGIC version
# MAGIC datatype
# MAGIC iord
# MAGIC tlabel
# MAGIC doc

# COMMAND ----------

# MAGIC %md
# MAGIC ### Custom tag proportion in TAG table VS. Custom tags usage in NUM table

# COMMAND ----------

tag.groupby("custom").agg(count("*").alias("count"), (count("*")*100/tag.count()).alias("percentage")).display()

# COMMAND ----------

# MAGIC %md
# MAGIC - This 79993 non-custom tags were filed

# COMMAND ----------

num_tag_joined = num.join(
    tag.select("tag", "version", "custom", "abstract"),
    on=["tag", "version"],
    how="left"
)
num_tag_joined.groupby("custom").agg(count("*").alias("count"), (count("*")*100/num_tag_joined.count()).alias("percentage")).display()

# COMMAND ----------

# MAGIC %md
# MAGIC Although Custom tags are 92.4 percent of the definitions but only use 8.8 percent of the numbers table

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Presentation DF

# COMMAND ----------

print("sub DF row count = ", pre.count())

# COMMAND ----------

print("---- Schema of sub ----")
pre.printSchema()

# COMMAND ----------

pre_info = generate_df_info(pre)
pre_info.display(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC pre_file:
# MAGIC adsh
# MAGIC report
# MAGIC line
# MAGIC stmt
# MAGIC tag
# MAGIC version
# MAGIC plabel
# MAGIC

# COMMAND ----------

sub.select("form").distinct().show()

# COMMAND ----------

tag.groupBy("datatype").count().display()

# COMMAND ----------

tag.where(col("datatype") != "monetary").display()

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Sub nulls

# COMMAND ----------

# MAGIC %md
# MAGIC #### sic and countryba
# MAGIC - Per-CIK mode: filling nulls with each company’s most common value for that field.
# MAGIC
# MAGIC - Global fallback mode: for any CIK that has no non-null values, we fill with the overall most frequent value in the entire table.

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



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC #### fye, fy, period, fp
# MAGIC
# MAGIC ##### following are default
# MAGIC
# MAGIC - fye → 1231 (Dec 31)
# MAGIC
# MAGIC - fy → the filing’s calendar year (extracted from the filed date)
# MAGIC
# MAGIC - period → fiscal-year end date in YYYYMMDD form, i.e. fy * 10000 + fye
# MAGIC
# MAGIC - fp → "FY"

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

# MAGIC %md
# MAGIC We pick **mode** for the categorical fields (`fy` and `fp`) and **median** for the numeric‐date fields (`period` and `fye`) because of the nature of those columns:
# MAGIC
# MAGIC 1. **fy (Fiscal Year) and fp (Fiscal Period)**
# MAGIC
# MAGIC    * Those are **labels**, not numbers you’d average.
# MAGIC    * You want the single year or period code a company most often uses, so filling with the **most frequent** value (the mode) makes sense.
# MAGIC    * Example: if Acme Corp. has five filings and four of them say `fp = "Q2"`, then any missing `fp` should almost certainly be `"Q2"`, not `"Q1"` or `"FY"`.
# MAGIC
# MAGIC 2. **period (Reporting Date) and fye (Fiscal Year-End Month/Day)**
# MAGIC
# MAGIC    * These are **numeric dates** stored as integers. Averaging them can produce nonsensical halfway dates (e.g. a “.5” day), and means are easily skewed by one outlier.
# MAGIC    * The **median** finds the exact middle of a company’s historical dates, so if one quarter was filed very late or early, it won’t drag your fill date to that extreme.
# MAGIC    * Example: if your company usually files around June 30 but had one odd March 31 filing, the median will stay at June 30 rather than shift toward March.
# MAGIC
# MAGIC In short:
# MAGIC
# MAGIC * **Mode** → best for categorical or discrete codes you want “most common.”
# MAGIC * **Median** → best for continuous or ordered data (like dates) where you need a robust center point.
# MAGIC

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Num file 

# COMMAND ----------

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when, count, percentile_approx

def impute_num_value_medians(df_num: DataFrame, df_sub: DataFrame) -> DataFrame:
    """
    Impute nulls in the NUM DataFrame’s `value` column by:
      1. Joining to SUB on `adsh` to bring in `cik`
      2. Grouping by (cik, tag, uom) to compute:
         - grp_count  : number of non-null values per group
         - grp_median : median of value per group
      3. Flagging groups with no non-null values as `is_incomplete`
      4. Filling null `value` entries with their group’s median where available

    Parameters:
        df_num (DataFrame): Cleaned NUM DataFrame containing `adsh`, `tag`, `uom`, and `value`.
        df_sub (DataFrame): Cleaned SUB DataFrame containing `adsh` and `cik`.

    Returns:
        DataFrame: A new DataFrame with:
          - `value` nulls imputed by group median
          - `is_incomplete` flag set True for rows in groups with zero non-null values
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
              "is_incomplete",
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

    return df_result


# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # PRE

# COMMAND ----------

# MAGIC %md
# MAGIC #### stmt 
# MAGIC 1. Compute each tag’s most common stmt (its mode) across all rows.
# MAGIC
# MAGIC 2. Join that back onto PRE and fill any null stmt with the tag’s mode.
# MAGIC
# MAGIC 3. Fallback to "UN" (Unclassifiable) for any tag that never appeared with a non-null stmt.

# COMMAND ----------

from pyspark.sql import DataFrame, Window
from pyspark.sql.functions import col, when, row_number

def impute_pre_stmt(df_pre: DataFrame) -> DataFrame:
    """
    Impute nulls in the PRE DataFrame’s `stmt` column by:
      1. Computing each tag’s most frequent statement type (mode) across filings.
      2. Filling null `stmt` values with that per-tag mode.
      3. As a final fallback, assigning 'UN' (Unclassifiable) to any remaining nulls.

    Parameters:
        df_pre (DataFrame): Cleaned PRE DataFrame containing at least 'tag' and 'stmt'.

    Returns:
        DataFrame: A new DataFrame with no nulls in `stmt`.
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



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


