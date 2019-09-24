import pandas as pd
import os

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

s_env = 10
path_core = "/Results/Train/Env_" + str(s_env) + "/"

Save_Path_Comp = path_core + "Env_" + str(s_env) + "_COMPILED"
writer = pd.ExcelWriter(__location__ + Save_Path_Comp + ".xlsx", engine='xlsxwriter')

ps_list = ["Score","Percent"]
for ps in ps_list:
    QL = pd.read_excel(__location__ + path_core + "QL/QL_averaged" + ".xlsx", sheetname=ps, skiprows=10, parse_cols="B:B", index=True)
    DSRL = pd.read_excel(__location__ + path_core + "DSRL/DSRL_averaged" + ".xlsx", sheetname=ps, skiprows=10, parse_cols="B:B", index=True)
    DSRL_trick = pd.read_excel(__location__ + path_core + "DSRL/DSRL_tricked_averaged" + ".xlsx",  sheetname=ps, skiprows=10,parse_cols="B:B")
    DQN = pd.read_excel(__location__ + path_core + "DQN/DQN_averaged" + ".xlsx", sheetname=ps, skiprows=10, parse_cols="B:B")

    QL.to_excel(startrow=1, startcol=0, excel_writer=writer,sheet_name=ps,index=False, header=False)
    DSRL.to_excel(startrow=1, startcol=1, excel_writer=writer,sheet_name=ps,index=False, header=False)
    DSRL_trick.to_excel(startrow=1, startcol=2, excel_writer=writer, sheet_name=ps, index=False, header=False)
    DQN.to_excel(startrow=1, startcol=3, excel_writer=writer, sheet_name=ps, index=False, header=False)

    worksheet = writer.sheets[ps]
    worksheet.write(0, 0, "QL")
    worksheet.write(0, 1, "DSRL")
    worksheet.write(0, 2, "DSRL_trick")
    worksheet.write(0, 3, "DQN")
writer.save()


