# Databricks notebook source
from pyspark.sql.functions import *

# COMMAND ----------

def fact_table_creation(num_df, sub_df, tag_df, date_df, pre_df, dim_pre, dim_company):
    num_tag_df = num_df.join(
        tag_df,
        on=["tag", "version"],
        how="inner"
    )

    num_tag_df = num_tag_df.drop("datatype", "iord", "tlabel", "doc")

    num_tag_df = num_tag_df.select(
        "adsh",
        "tag_id",   # move tag_id to second
        "ddate",
        "value",
        "tag",
        "version"
    )

    fact_table = num_tag_df.join(sub_df, on="adsh", how="inner")

    fact_table = fact_table.drop(
        "name",
        "sic",
        "countryba",
        "stprba",
        "cityba",
        "zipba",
        "baph",
        "fye",
        "fy",
        "fp"
    )

    fact_table = fact_table.drop(
        "form",
        "period",
        "filed",
        "delay_days",
        "accepted",
        "instance"
    )

    fact_table = fact_table.join(
        date_df,
        on=["ddate"],
        how="inner"
    )

    fact_table=fact_table.drop("quarter", "year", "month", "day")
    fact_table=fact_table.withColumnRenamed("id", "dim_date_id")

    fact_table = fact_table.join(
        pre_df,
        on=["adsh", "tag", "version"],
        how="inner"
    )

    fact_table = fact_table.join(
        dim_pre,
        on=["stmt", "report", "line", "plabel"],
        how="inner"
    )

    fact_table=fact_table.withColumnRenamed("id", "dim_pre_id").withColumnRenamed("tag_id", "dim_tag_id")
    fact_table=fact_table.drop("stmt", "report", "line", "plabel")

    fact_table = fact_table.join(
        dim_company,
        on=["cik"],
        how="inner"
    )
    fact_table=fact_table.withColumnRenamed("id", "dim_company_id")
    fact_table=fact_table.drop("name", "sic", "countryba", "stprba", "cityba", "zipba", "form", "fy", "fp", "fye", "baph")
    fact_table=fact_table.select("adsh", "tag", "version", "ddate", "value", "cik", "dim_company_id", "dim_date_id", "dim_tag_id", "dim_pre_id")
    return fact_table

# COMMAND ----------

num_df = spark.read.format("delta").table("silver.numbers")
num_df = num_df.drop("is_value_incomplete")
sub_df = spark.read.format("delta").table("silver.submissions")
tag_df = spark.read.format("delta").table("silver.tags")
date_df = spark.read.format("delta").table("gold.dim_date")
pre_df = spark.read.format("delta").table("silver.presentations")
dim_pre= spark.read.format("delta").table("gold.dim_pre")
dim_company= spark.read.format("delta").table("gold.dim_company")
fact_table=fact_table_creation(num_df, sub_df, tag_df, date_df, pre_df, dim_pre, dim_company)
display(fact_table)

# COMMAND ----------

# MAGIC %sql
# MAGIC drop table if exists gold.fact_financial;

# COMMAND ----------

fact_table.write.format("delta").mode("overwrite").saveAsTable("gold.fact_financial")
financial_df_loaded = spark.read.format("delta").table("gold.fact_financial")
display(financial_df_loaded)

# COMMAND ----------


