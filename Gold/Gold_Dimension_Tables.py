# Databricks notebook source
spark.sql("DROP TABLE IF EXISTS gold.dim_tag") 

# COMMAND ----------

tag_df = spark.read.format("delta").load("dbfs:/user/hive/warehouse/silver.db/tags")
tag_df.write.format("delta").mode("overwrite").saveAsTable("gold.dim_tag")

# COMMAND ----------

sub_df=spark.read.format("delta").load("dbfs:/user/hive/warehouse/silver.db/submissions")

# COMMAND ----------

spark.sql("DROP TABLE IF EXISTS gold.dim_submission") 

# COMMAND ----------

dim_sub_df=sub_df.select("adsh", "form", "period", "filed", "delay_days", "accepted", "instance")
dim_sub_df.write.format("delta").mode("overwrite").saveAsTable("gold.dim_submission")
sub_df=sub_df.drop("period", "filed", "delay_days", "accepted", "instance")

# COMMAND ----------

# display(sub_df.limit(20))

# COMMAND ----------

from pyspark.sql.window import Window
from pyspark.sql.functions import row_number

dim_company_df=sub_df.select("cik", "name", "sic", "countryba", "stprba", "cityba", "zipba", "baph", "fye","form", "fy", "fp")
dim_company_df.dropDuplicates()

# Define a window over the entire dataframe
window_spec = Window.orderBy("cik")  # or any other stable column

# Add surrogate key starting from 1
dim_company_df = dim_company_df.withColumn("id", row_number().over(window_spec))
dim_company_df = dim_company_df.select("id", "cik", "name", "sic", "countryba", "stprba", "cityba", "zipba", "baph", "fye", "form", "fy", "fp")
display(dim_company_df)

# COMMAND ----------

spark.sql("DROP TABLE IF EXISTS gold.dim_company") 

# COMMAND ----------

dim_company_df.write.format("delta").mode("overwrite").option("overwriteSchema","true").saveAsTable("gold.dim_company")
sub_df=sub_df.drop("name", "sic", "countryba", "stprba", "cityba", "zipba", "baph", "fye", "form", "fy", "fp")

# COMMAND ----------

# display(sub_df.limit(20))

# COMMAND ----------

tag_df_loaded = spark.read.format("delta").load("/FileStore/Mock_Project/Star_Schema_Delta/dim_tag")
display(tag_df_loaded)

# COMMAND ----------

submission_df_loaded = spark.read.format("delta").load("/FileStore/Mock_Project/Star_Schema_Delta/dim_submission")
display(submission_df_loaded)

# COMMAND ----------

company_df_loaded = spark.read.format("delta").load("/FileStore/Mock_Project/Star_Schema_Delta/dim_company")
display(company_df_loaded)

# COMMAND ----------

num_df = spark.read.format("delta").load("dbfs:/user/hive/warehouse/silver.db/numbers")
display(num_df.limit(20))

# COMMAND ----------

dim_date=num_df.select("ddate")
display(dim_date.limit(20))

# COMMAND ----------

from pyspark.sql.functions import year, month, dayofmonth

dim_date = dim_date.withColumn("year", year("ddate")) \
                   .withColumn("month", month("ddate")) \
                   .withColumn("day", dayofmonth("ddate"))
dim_date = dim_date.dropDuplicates()
display(dim_date.limit(20))

# COMMAND ----------

from pyspark.sql.functions import when

dim_date = dim_date.withColumn(
    "quarter",
    when(dim_date.month.isin(1, 2, 3), 1)
    .when(dim_date.month.isin(4, 5, 6), 2)
    .when(dim_date.month.isin(7, 8, 9), 3)
    .when(dim_date.month.isin(10, 11, 12), 4)
)
display(dim_date.limit(20))

# COMMAND ----------

from pyspark.sql.window import Window
from pyspark.sql.functions import row_number,col

dim_date = dim_date.filter(col("ddate").isNotNull())
# Define a window over the entire dataframe
window_spec = Window.orderBy("ddate")  # or any other stable column

# Add surrogate key starting from 1
dim_date = dim_date.withColumn("id", row_number().over(window_spec))
display(dim_date)

# COMMAND ----------

dim_date=dim_date.select("id", "ddate", "year", "quarter", "month", "day")
display(dim_date.limit(20))

# COMMAND ----------

spark.sql("DROP TABLE IF EXISTS gold.dim_date") 

# COMMAND ----------

dim_date.write.format("delta").mode("overwrite").saveAsTable("gold.dim_date")

# COMMAND ----------

# date_df_loaded = spark.read.format("delta").load("/FileStore/Mock_Project/Star_Schema_Delta/dim_date")
# display(date_df_loaded)

# COMMAND ----------

pre_df = spark.read.format("delta").table("silver.presentations")
display(pre_df.limit(20))

# COMMAND ----------

pre_df=pre_df.select("stmt", "report", "line", "plabel")
pre_df=pre_df.drop_duplicates()
# display(pre_df)

# COMMAND ----------

from pyspark.sql.window import Window
from pyspark.sql.functions import row_number

# Define a window over the entire dataframe
window_spec = Window.orderBy("stmt")  # or any other stable column

# Add surrogate key starting from 1
pre_df = pre_df.withColumn("id", row_number().over(window_spec))
pre_df=pre_df.select("id", "stmt", "report", "line", "plabel")
# display(pre_df)

# COMMAND ----------

spark.sql("DROP TABLE IF EXISTS gold.dim_pre") 

# COMMAND ----------

pre_df.write.format("delta").mode("overwrite").saveAsTable("gold.dim_pre")


# COMMAND ----------

# pre_df_loaded = spark.read.format("delta").load("/FileStore/Mock_Project/Star_Schema_Delta/dim_pre")
# display(pre_df_loaded)
