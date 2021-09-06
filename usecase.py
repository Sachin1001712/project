from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql.functions import *
import pandas

def creat_data_frame(spark, file_path):
    """
    function read csv file to data frame
    :param file_path:
    :return: Data Frame
    """
    df = spark.read \
        .format("csv") \
        .option("header", "true") \
        .option("inferSchema", "true")\
        .option("samplingRatio", 0.01) \
        .option("delimiter", ",") \
        .load(file_path)
    return df

def write_df_to_csv(output_df,file_path):
    """
    function to write data frame to output path in csv format
    :param output_df:
    :param filePath:
    :return:
    """
    output_df\
        .coalesce(1)\
        .write\
        .format("csv")\
        .option("header","true")\
        .mode("overwrite")\
        .save(file_path)

def analysis_1_result(primary_person_df,output_folder_path):
    """
    function to - find the number of crashes (accidents) in which number of persons killed are male?
    :param deathDF: inputDF of DataFrame type
    :return: None
    """
    male_death_count_df = primary_person_df\
        .filter(col("PRSN_GNDR_ID") == "MALE").agg(count("PRSN_GNDR_ID").alias("MALE_DEATH_CNT"))
    print("Analysis 1: \nTotal number of crashes (accidents) in which number of persons killed are male is :")
    male_death_count_df.show() #Dispalying result
    write_df_to_csv(male_death_count_df,output_folder_path+"analysis_1_result") #Writing to csv file

def analysis_2_result(units_df,output_folder_path):
    """
    function to calculate - How many two wheelers are booked for crashes?
    :param units_df:
    :return:
    """
    two_wheeler_df = units_df\
        .filter(col("VEH_BODY_STYL_ID").isin(["POLICE MOTORCYCLE", "MOTORCYCLE"]))\
        .distinct()\
        .agg(count("VEH_BODY_STYL_ID").alias("TWO_WHEELER_COUNT"))
    # distinct is calculated as there are entries with duplicate details
    # : with_duplicate_count = 784 2 wheelers
    # : without_duplicates_count = 773 2 wheelers
    print("Analysis 2: \nTotal number of two wheelers are booked for crashes is :")
    two_wheeler_df.show() #Displaying result DF
    write_df_to_csv(two_wheeler_df,output_folder_path+"analysis_2_result") #Writing to csv file

def analysis_3_result(primary_person_df,output_folder_path):
    """
    function to find out which state has highest number of accidents in which females are involved
    :param primary_person_df:
    :return:
    """
    rankWindowSpec = Window.orderBy(col("MAX_FEMALE_COUNT").desc())
    state_with_female_involved_df = primary_person_df \
        .filter(col("PRSN_GNDR_ID") == "FEMALE") \
        .groupBy("DRVR_LIC_STATE_ID", "PRSN_GNDR_ID") \
        .agg(count("*").alias("MAX_FEMALE_COUNT"))\
        .withColumn("RANK", row_number().over(rankWindowSpec))\
        .filter("RANK=1") \
        .drop("RANK","PRSN_GNDR_ID")\
        .select(col("DRVR_LIC_STATE_ID").alias("STATE"),"MAX_FEMALE_COUNT")
    print("Analysis 3:\nState with highest number of accidents in which females are involved is: ")
    state_with_female_involved_df.show()
    write_df_to_csv(state_with_female_involved_df,output_folder_path+"analysis_3_result")

def analysis_4_result(units_df,output_folder_path):
    """
            function to find - Which are the Top 5th to 15th VEH_MAKE_IDs that contribute to a largest number of injuries including death
            :param units_df:
            :return:
            """
    rank_window_spec = Window.orderBy(col("TOT_INJRY_PLUS_DEATH").desc())
    veh_make_id_df = units_df \
        .drop_duplicates() \
        .filter((units_df.TOT_INJRY_CNT != 0) | (col("DEATH_CNT") != 0)) \
        .withColumn("INJRY_PLUS_DEATH", col("TOT_INJRY_CNT") + col("DEATH_CNT")) \
        .groupBy("VEH_MAKE_ID") \
        .agg(sum("INJRY_PLUS_DEATH").alias("TOT_INJRY_PLUS_DEATH")) \
        .withColumn("RANK", row_number().over(rank_window_spec)) \
        .filter((col("RANK") >= 5) & (col("RANK") <= 15))
    print("Analysis 4:\n5th to 15th VEH_MAKE_IDs that contribute to a largest number of injuries including death : ")
    veh_make_id_df.show()
    write_df_to_csv(veh_make_id_df, output_folder_path+"analysis_4_result")

def analysis_5_result(primary_person_df,units_df,output_folder_path):
    """
    function to find - top ethnic user group of each unique body style
    :param primary_person_df:
    :param units_df:
    :return:
    """
    df1 = primary_person_df.distinct().select("CRASH_ID","UNIT_NBR","PRSN_ETHNICITY_ID")
    df2 = units_df.distinct().select("CRASH_ID","UNIT_NBR","VEH_BODY_STYL_ID")
    join_res_df = df1.join(broadcast(df2), ["CRASH_ID","UNIT_NBR"])
    rank_window_spec = Window.partitionBy("VEH_BODY_STYL_ID").orderBy(col("COUNT").desc())
    top_ethnic_unique_body_df = join_res_df \
        .groupBy("VEH_BODY_STYL_ID","PRSN_ETHNICITY_ID") \
        .agg(count("*").alias("COUNT")) \
        .withColumn("RANK", dense_rank().over(rank_window_spec)) \
        .filter("RANK == 1")\
        .drop("RANK")

    print("Analysis 5: \nMention the top ethnic user group of each unique body style")
    top_ethnic_unique_body_df.show(truncate=False)
    write_df_to_csv(top_ethnic_unique_body_df, output_folder_path+"analysis_5_result")

def analysis_6_result(primary_person_df,output_folder_path):
    rank_window_spec = Window.orderBy(col("COUNT").desc())
    top_zip_codes_df = primary_person_df \
        .filter(col("PRSN_ALC_RSLT_ID") == "Positive") \
        .groupBy("DRVR_ZIP") \
        .agg(count("*").alias("COUNT")) \
        .withColumn("RANK", row_number().over(rank_window_spec)) \
        .filter((col("RANK")) <= 5).select("DRVR_ZIP", "COUNT", "RANK")

    print("Analysis 6: \nTop 5 Zip Codes with highest number crashes with alcohols as the contributing factor to a crash (Use Driver Zip Code):")
    top_zip_codes_df.show()
    write_df_to_csv(top_zip_codes_df, output_folder_path+"analysis_6_result")

def analysis_7_result(units_df,damages_df,output_folder_path):
    """

    :param units_df:
    :param damages_df:
    :param charges_df:
    :return:
    """
    res1_df = damages_df \
        .filter(col("DAMAGED_PROPERTY").isin(["NONE", "NONE1"])) \
        .select("CRASH_ID")\
        .distinct()

    res2_df = units_df\
        .withColumn("DAMAGE_SCL_1",regexp_extract("VEH_DMAG_SCL_1_ID",'(\d)', 1).cast("Int"))\
        .withColumn("DAMAGE_SCL_2",regexp_extract("VEH_DMAG_SCL_2_ID",'(\d)', 1).cast("Int"))\
        .select("*",(when(col("DAMAGE_SCL_1").isNull(), 0).otherwise(col("DAMAGE_SCL_1")) +
                  when(col("DAMAGE_SCL_2").isNull(), 0).otherwise(col("DAMAGE_SCL_2"))).alias("TOTAL_DAMAGE_SCALE"))\
        .filter("TOTAL_DAMAGE_SCALE > 4")\
        .filter(col("VEH_BODY_STYL_ID").contains("CAR"))\
        .filter(col("FIN_RESP_TYPE_ID").rlike("INSURANCE"))\
        .select("CRASH_ID")\
        .distinct()


    final_result_df = res1_df\
        .join(broadcast(res2_df),["CRASH_ID"],how="inner")\
        .distinct()\
        .agg(count("*").alias("DISTINCT_CRASH_ID_COUNT"))

    print("Analysis 7:\nCount of Distinct Crash IDs where No Damaged Property was observed and "+
          "\nDamage Level (VEH_DMAG_SCL~) is above 4 and car avails Insurance")
    final_result_df.show(truncate=False)
    write_df_to_csv(final_result_df, output_folder_path+"analysis_7_result")

def analysis_8_result(units_df, spark,output_folder_path):
    """
    function to find - Top 5 Vehicle Makes where drivers are charged with speeding related offences,
    has licensed Drivers, uses top 10 used vehicle colours and has car licensed
    with the Top 25 states with highest number of offences
    :param units_df:
    :param spark:
    :return:
    """

    rank_window_spec = Window.orderBy(col("COUNT").desc())
    top10_veh_color_list = list(units_df\
        .distinct()\
        .select("VEH_COLOR_ID")\
        .groupBy("VEH_COLOR_ID")\
        .agg(count("*").alias("COUNT"))\
        .withColumn("RANK", dense_rank().over(rank_window_spec))\
        .filter("RANK <= 10")\
        .drop("RANK","COUNT")\
        .toPandas()["VEH_COLOR_ID"])


    top25_states_list = list(units_df\
        .distinct()\
        .select("VEH_LIC_STATE_ID","VEH_BODY_STYL_ID")\
        .filter(col("VEH_BODY_STYL_ID").contains("CAR"))\
        .groupBy("VEH_LIC_STATE_ID")\
        .agg(count("*").alias("COUNT"))\
        .withColumn("RANK", row_number().over(rank_window_spec))\
        .filter("RANK <= 25")\
        .drop("COUNT","RANK")\
        .toPandas()["VEH_LIC_STATE_ID"])

    top10_veh_col_bv = spark.sparkContext.broadcast(top10_veh_color_list)
    top25_states_bv = spark.sparkContext.broadcast(top25_states_list)


    result_df = units_df\
        .distinct()\
        .select("VEH_MAKE_ID","VEH_COLOR_ID","VEH_LIC_STATE_ID","CONTRIB_FACTR_1_ID","CONTRIB_FACTR_2_ID","CONTRIB_FACTR_P1_ID")\
        .filter((col("CONTRIB_FACTR_1_ID").contains("SPEED")) | (col("CONTRIB_FACTR_2_ID").contains("SPEED")) | (col("CONTRIB_FACTR_P1_ID").contains("SPEED")))\
        .filter((col("VEH_COLOR_ID").isin(top10_veh_col_bv.value)) & (col("VEH_LIC_STATE_ID").isin(top25_states_bv.value)))\
        .groupBy("VEH_MAKE_ID")\
        .agg(count("*").alias("COUNT"))\
        .withColumn("RANK", row_number().over(rank_window_spec))\
        .filter((col("RANK")) <= 5)\
        .drop("COUNT")

    print("Analysis 7:\nTop 5 Vehicle Makes where drivers are charged with speeding related offences, has licensed Drivers, "+
          "uses top 10 used vehicle colours and has car licensed with the Top 25 states with highest number of offences : ")
    result_df.show()
    write_df_to_csv(result_df,output_folder_path+"analysis_8_result")


if __name__ == "__main__":

    import sys
    input_folder_path = ''
    output_folder_path = ''
    if (len(sys.argv)-1) == 2:
        input_folder_path = sys.argv[1].rstrip("/")+"/"
        output_folder_path = sys.argv[2].rstrip("/")+"/"
    else:
        msg = "Usage-using spark submit: spark-submit --master local/yarn [--deploy_mode client/cluster] <python_file_path> <input_folder_path> <output_folder_path>" \
              "\nor\n" \
              "Usage - python <python_file_path> <input_folder_path> <output_folder_path>" \
              "Ex: python /Users/sachin/PycharmProjects/Codes/usecase.py /Users/sachin/PycharmProjects/Codes/input /Users/sachin/PycharmProjects/Codes/output"
        print(msg)
        raise Exception("Missing input arguments : "+ msg)

    spark = SparkSession.builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    units_df = creat_data_frame(spark, input_folder_path+"Units_use.csv")
    primary_person_df = creat_data_frame(spark, input_folder_path+"Primary_Person_use.csv")
    charges_df = creat_data_frame(spark, input_folder_path+"Charges_use.csv")
    damages_df = creat_data_frame(spark, input_folder_path+"Damages_use.csv")
    endorse_df = creat_data_frame(spark, input_folder_path+"Endorse_use.csv")
    restrict_df = creat_data_frame(spark, input_folder_path+"Restrict_use.csv")

    #Adding to cache as these data frames are used multiple times
    units_df.cache()
    primary_person_df.cache()

    analysis_1_result(primary_person_df,output_folder_path)
    analysis_2_result(units_df,output_folder_path)
    analysis_3_result(primary_person_df,output_folder_path)
    analysis_4_result(units_df,output_folder_path)
    analysis_5_result(primary_person_df,units_df,output_folder_path)
    analysis_6_result(primary_person_df,output_folder_path)
    analysis_7_result(units_df, damages_df,output_folder_path)
    analysis_8_result(units_df, spark,output_folder_path)

