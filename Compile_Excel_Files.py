import pandas as pd
import openpyxl
import numpy as np
import os
import string
import glob

''' This program compiles all (individual) saved excel files to compare different models in one environment
'''


__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
path_core = __location__+ "/Results/Train/"

print("OK")
# SELECT THE ENVIRONMENTS
# env_path_list = ["Env_1",
#                  "Env_2",
#                  "Env_3",
#                  "Env_8",
#                  "Env_9",
#                  "Env_10",
#                  "Env_11"]
env_path_list = ["Env_1",
                 "Env_2",
                 "Env_3",
                 "Env_4"]

env_path_list = ["Env_1"]
alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'AA', 'AB', 'AC', 'AD', 'AE', 'AF', 'AG', 'AH', 'AI', 'AJ', 'AK', 'AL', 'AM', 'AN', 'AO', 'AP', 'AQ', 'AR', 'AS', 'AT', 'AU', 'AV', 'AW', 'AX', 'AY', 'AZ']
list_sheets = ["Run_Conf", "Score", "Percent", "Loss", "Time"]


for env_path in env_path_list:
    file_path_list = []
    path = path_core + env_path + "/Train_Env_1_DQN*.xlsx"
    for fname in sorted(glob.glob(path)):
        file_path_list.append(fname)
    print("LEN(FILE_PATH_LIST):", len(file_path_list))

    load_path = __location__+ "/Results/Train/Compare_Models.xlsx"
    excel_data_base = pd.ExcelFile(load_path)
    load_path_new = __location__+ "/Results/Train/" + env_path + "/Compare_Models_new_" + env_path + ".xlsx"
    excel_writer_to_append = pd.ExcelWriter(load_path_new)

    workbook = excel_writer_to_append.book

    excel_data_base_col = pd.read_excel(excel_data_base, sheetname="Run_Conf")

    df_Run_Conf_list = pd.DataFrame()
    df_Score_list = pd.DataFrame()
    df_Percent_list = pd.DataFrame()
    df_Loss_list = pd.DataFrame()
    df_Time_list = pd.DataFrame()

    for i in range(len(file_path_list)):
        print("File:", i)
        excel_file = pd.ExcelFile(file_path_list[i])
        # print("excel_file ", excel_file )
        df_Run_Conf = pd.read_excel(excel_file, sheetname=list_sheets[0], converters={'A': str})
        df_Run_Conf = df_Run_Conf.set_index(list_sheets[0])
        df_Score = pd.read_excel(excel_file, sheetname=list_sheets[1], parse_cols="A:B")
        df_Score = df_Score.set_index(list_sheets[1])
        df_Percent = pd.read_excel(excel_file, sheetname=list_sheets[2], parse_cols="A:B")
        df_Percent = df_Percent.set_index(list_sheets[2])
        df_Loss = pd.read_excel(excel_file, sheetname=list_sheets[3], parse_cols="A:B")
        df_Loss = df_Loss.set_index(list_sheets[3])
        df_Time = pd.read_excel(excel_file, sheetname=list_sheets[4], parse_cols="A:B")
        df_Time = df_Time.set_index(list_sheets[4])

        df_Run_Conf_list = pd.concat([df_Run_Conf_list, df_Run_Conf], axis=1, join="outer")
        df_Score_list = pd.concat([df_Score_list, df_Score], axis=1, join="outer")
        df_Percent_list = pd.concat([df_Percent_list, df_Percent], axis=1, join="outer")
        df_Loss_list = pd.concat([df_Loss_list, df_Loss], axis=1, join="outer")
        df_Time_list = pd.concat([df_Time_list, df_Time], axis=1, join="outer")

        list_of_df = [df_Run_Conf_list,df_Score_list,df_Percent_list,df_Loss_list,df_Time_list]

    # print("df_Run_Conf_list\n\n", df_Run_Conf_list)

    i = 0
    df = pd.DataFrame()
    for sheet in list_sheets:
        print("Sheet:", sheet)
        # if sheet == "Run_Conf":
        #     dict = {} # In order to parse the correct data the headers should be strings
        #     for n in range(len(excel_data_base_col.columns)):
        #         dict[n] = str
        #     df_data_base = pd.read_excel(excel_data_base, sheetname=sheet, converters=dict)
        # else:
        df_data_base = pd.read_excel(excel_data_base, sheetname=sheet)
        new_df_data_base = df_data_base.set_index(sheet)
        # print("df_Run_Conf_list\n\n", new_df_data_base)

        df = list_of_df[i]
        new_df_data_base = pd.concat([new_df_data_base, df], axis=1, join_axes=[df.index], join="outer")
        # print("new_df_data_base \n\n", new_df_data_base )

        new_df_data_base.to_excel(excel_writer_to_append, sheet_name=sheet, index=True)

        if sheet == "Run_Conf":
            worksheet = excel_writer_to_append.sheets[sheet]
            format1 = workbook.add_format()
            format1.set_center_across()
            l = len(new_df_data_base.columns)
            worksheet.set_column('A:A', 18)
            worksheet.set_column(1,l, 12, format1)
            worksheet.write(0, l, str(range(1, l, 1)))

        worksheet = excel_writer_to_append.sheets[sheet]
        for j in range(len(new_df_data_base.columns)):
            worksheet.write(0, j+1, alphabet[j])

        i += 1

    excel_writer_to_append.save()