# Databricks notebook source
# MAGIC %md
# MAGIC **Credentials**

# COMMAND ----------

secret_value = dbutils.secrets.get('connection-credentials', 'secret-value')
application_id = dbutils.secrets.get('connection-credentials', 'application-id')
directory_id = dbutils.secrets.get('connection-credentials', 'app-registration-directory-id')
storage_account = dbutils.secrets.get('connection-credentials', 'storage-account-name')
container_name = dbutils.secrets.get('connection-credentials', 'container-name')
security_key = dbutils.secrets.get('connection-credentials', 'storage-account-security-key')

# COMMAND ----------

# MAGIC %md
# MAGIC **Connection**

# COMMAND ----------

spark.conf.set(f"fs.azure.account.auth.type.{storage_account}.dfs.core.windows.net", "OAuth")
spark.conf.set(f"fs.azure.account.oauth.provider.type.{storage_account}.dfs.core.windows.net", "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider")
spark.conf.set(f"fs.azure.account.oauth2.client.id.{storage_account}.dfs.core.windows.net", application_id)
spark.conf.set(f"fs.azure.account.oauth2.client.secret.{storage_account}.dfs.core.windows.net", secret_value)
spark.conf.set(f"fs.azure.account.oauth2.client.endpoint.{storage_account}.dfs.core.windows.net", f"https://login.microsoftonline.com/{directory_id}/oauth2/token")

# COMMAND ----------

# MAGIC %md
# MAGIC **Mount the container**

# COMMAND ----------

dbutils.fs.mount(
  source = f"wasbs://{container_name}@{storage_account}.blob.core.windows.net",
  mount_point = f"/mnt/{container_name}",
  extra_configs = {
    f"fs.azure.account.key.storageaccb1.blob.core.windows.net":security_key
  }
)

# COMMAND ----------

# MAGIC %md
# MAGIC **Load the data in Bronze layer**

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from delta.tables import *
import re

spark = SparkSession.builder.appName("ParquetToDelta").enableHiveSupport().getOrCreate()

root_directory = "/mnt/unzip-financial-data"

# Ensure the bronze_delta schema exists
spark.sql("CREATE SCHEMA IF NOT EXISTS bronzes")

# Schemas
submissions_schema = StructType([
    StructField("adsh", StringType(), False),
    StructField("cik", LongType(), True),
    StructField("name", StringType(), True),
    StructField("sic", IntegerType(), True),
    StructField("countryba", StringType(), False),
    StructField("stprba", StringType(), True),
    StructField("cityba", StringType(), False),
    StructField("zipba", StringType(), True),
    StructField("bas1", StringType(), True),
    StructField("bas2", StringType(), True),
    StructField("baph", StringType(), True),
    StructField("countryma", StringType(), True),
    StructField("stprma", StringType(), True),
    StructField("cityma", StringType(), True),
    StructField("zipma", StringType(), True),
    StructField("mas1", StringType(), True),
    StructField("mas2", StringType(), True),
    StructField("countryinc", StringType(), False),
    StructField("stprinc", StringType(), True),
    StructField("ein", StringType(), True),
    StructField("former", StringType(), True),
    StructField("changed", DateType(), True),
    StructField("afs", StringType(), True),
    StructField("wksi", BooleanType(), False),
    StructField("fye", StringType(), False),
    StructField("form", StringType(), False),
    StructField("period", StringType(), False),
    StructField("fy", IntegerType(), False),
    StructField("fp", StringType(), False),
    StructField("filed", StringType(), False),
    StructField("accepted", TimestampType(), False),
    StructField("prevrpt", BooleanType(), False),
    StructField("detail", BooleanType(), False),
    StructField("instance", StringType(), False),
    StructField("nciks", IntegerType(), False),
    StructField("aciks", StringType(), True)
])

tags_schema = StructType([
    StructField("tag", StringType(), False),
    StructField("version", StringType(), False),
    StructField("custom", IntegerType(), False),
    StructField("abstract", BooleanType(), False),
    StructField("datatype", StringType(), True),
    StructField("iord", StringType(), True),
    StructField("crdr", StringType(), True),
    StructField("tlabel", StringType(), True),
    StructField("doc", StringType(), True)
])

numbers_schema = StructType([
    StructField("adsh", StringType(), True),
    StructField("tag", StringType(), True),
    StructField("version", StringType(), True),
    StructField("ddate", StringType(), True),
    StructField("qtrs", IntegerType(), True),
    StructField("uom", StringType(), True),
    StructField("coreg", StringType(), True),
    StructField("value", DecimalType(28,4), True),
    StructField("footnote", StringType(), True)
])

presentations_schema = StructType([
    StructField("adsh", StringType(), False),
    StructField("report", IntegerType(), True),
    StructField("line", IntegerType(), False),
    StructField("stmt", StringType(), False),
    StructField("inpth", BooleanType(), False),
    StructField("rfile", StringType(), False),
    StructField("tag", StringType(), False),
    StructField("version", StringType(), False),
    StructField("plabel", StringType(), False)
])

def extract_year_quarter(folder_path):
    match = re.search(r'(\d{4})[^\d]*q([1-4])', folder_path, re.IGNORECASE)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None
def process_quarter_folder(folder_path):
    year, quarter = extract_year_quarter(folder_path)
    if not year or not quarter:
        print(f"Skipping folder {folder_path} due to invalid naming convention.")
        return

    files = dbutils.fs.ls(folder_path)
    for file in files:
        file_path = file.path
        if file.name == 'pre.parquet':
            df = spark.read.parquet(file_path)
            for field in presentations_schema.fields:
                df = df.withColumn(field.name, col(field.name).cast(field.dataType))
            df = df.withColumn("year", lit(year)).withColumn("quarter", lit(quarter))
            if not spark._jsparkSession.catalog().tableExists("bronzes", "Presentations"):
                df.write.format("delta").mode("append").partitionBy("year", "quarter").saveAsTable("bronzes.Presentations")
            else:
                df.write.format("delta").mode("append").saveAsTable("bronzes.Presentations")

        elif file.name == 'num.parquet':
            df = spark.read.parquet(file_path)
            for field in numbers_schema.fields:
                df = df.withColumn(field.name, col(field.name).cast(field.dataType))
            df = df.withColumn("year", lit(year)).withColumn("quarter", lit(quarter))
            if not spark._jsparkSession.catalog().tableExists("bronzes", "Numbers"):
                df.write.format("delta").mode("append").partitionBy("year", "quarter").saveAsTable("bronzes.Numbers")
            else:
                df.write.format("delta").mode("append").saveAsTable("bronzes.Numbers")

        elif file.name == 'sub.parquet':
            df = spark.read.parquet(file_path)
            for field in submissions_schema.fields:
                df = df.withColumn(field.name, col(field.name).cast(field.dataType))
            df = df.withColumn("year", lit(year)).withColumn("quarter", lit(quarter))
            if not spark._jsparkSession.catalog().tableExists("bronzes", "Submissions"):
                df.write.format("delta").mode("append").partitionBy("year", "quarter").saveAsTable("bronzes.Submissions")
            else:
                df.write.format("delta").mode("append").saveAsTable("bronzes.Submissions")

        elif file.name == 'tag.parquet':
            df = spark.read.parquet(file_path)
            for field in tags_schema.fields:
                df = df.withColumn(field.name, col(field.name).cast(field.dataType))
            df = df.withColumn("year", lit(year)).withColumn("quarter", lit(quarter))
            if not spark._jsparkSession.catalog().tableExists("bronzes", "Tags"):
                df.write.format("delta").mode("append").partitionBy("year", "quarter").saveAsTable("bronzes.Tags")
            else:
                df.write.format("delta").mode("append").saveAsTable("bronzes.Tags")

        else:
            print(f"Skipping file {file.name} as it doesn't match any known pattern")

def process_all_quarters():
    quarters = dbutils.fs.ls(root_directory)
    for quarter in quarters:
        print(f"Processing folder: {quarter.name}")
        process_quarter_folder(quarter.path)

spark.conf.set("spark.databricks.delta.schema.autoMerge.enabled", "true")

print(f"Starting to process all quarter folders from {root_directory}")
process_all_quarters()
print("Data upload completed for all quarters.")
