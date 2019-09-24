import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import matplotlib.patches as mpatches
import os

# plt.style.use('ggplot')
plt.interactive(False)

# FOR SERVER:
# __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
# path = __location__+ "/Results/Train/"

# FOR MY MACHINE:

path = r"E:\ENGLAND\City University of London\Events\NIPS\NIPS_RESULTS\Comparison between DSRL, QL, DSRL_trick in ALL ENVs/env"
file_path_list = []
#######################################################
''' CHOOSE WHAT TO PLOT '''
plot_Score = True
plot_Percent_2 = False
plot_Percent = False

''' CHOOSE ENVIRONMENTS '''
# env_list = [1,2,3,4,8,9,10,11,12,13,14,15,16,17,18,19]
env_list = [2,3]
# path_core = "/Train_Env_"
path_core = "/Test_Env_"

percent_env_list = [8, 9, 10, 11]
''' CHOOSE SAVE and SHOW '''
save_Plot = True
save_path = r"E:\ENGLAND\City University of London\Events\NIPS\NIPS_RESULTS\Comparison between DSRL, QL, DSRL_trick in ALL ENVs/"
show_Plot = True


''' CHOOSE MODELS '''
# choose_models_list = [0]
choose_models_list = [0,1,2,3]
color_dict = {"DSRL_object_near ": "purple", "QL ": "green", "SRL ": "red", "DSRL-": "black", "DQN_": "blue", "DQN-":"blue"}
# Becareful with the spaces after the names:
names = ["DSRL_object_near ",
         "QL ",
         "SRL ",
         "DQN_"
         ]

''' ACTIVATE THIS FOR ENV 11 - THE SIMULATED PLOTS'''
# names = ["DSRL_object_near ",
#          "QL ",
#          "SRL ",
#          "DQN-",
#          "DSRL-",
#          ]



sub_name = {"QL ": "Q-Learning", "DQN_":"DQN", "DQN-":"DQN", "DSRL-":"DSRL ", "SRL ":"SRL", "DSRL_object_near ":"SRL+CS"}
#######################################################
alpha = 0.45 # 0.55
linewidth = 2.4 # 1.4
if plot_Score == True:
    print("\nPLOTTING SCORE")
    for i in env_list:
        print("\nEnv", i)
        Score_list = []

        title_name = "Env " + str(i)
        colors = []
        n=0
        for model in choose_models_list:

            name = names[model]
            file_name = path_core + str(i) + "_" + name + "*.xlsx"
            path_new = path + str(i) + file_name
            print("name:", name)
            for fname in glob.glob(path_new):
                print("fname:", fname)
                '''SCORE'''
            Score = pd.read_excel(fname, parse_cols="C:L", sheetname="Score")
            if n == 0:
                plot1 = Score.plot(grid=True,
                                   color=color_dict[name],
                                   alpha=alpha,
                                   linewidth=linewidth,
                                   fontsize=14)
            else:
                Score.plot(ax=plot1,
                           grid=True,
                           color=color_dict[name],
                           alpha=alpha,
                           linewidth=linewidth,
                           fontsize=14)

            color = mpatches.Patch(alpha=0.8, color=color_dict[name], label=sub_name[name])
            print("color_dict[name]:", color_dict[name])
            colors.append(color)
            n += 1


        # plt.title(title_name, fontsize=16)
        print(colors)
        plot1.legend(handles=colors,
                   loc=0,
                   borderaxespad=1,
                   fontsize=14)

        plt.xlabel("Steps (during 1000 episodes)", fontsize=20)
        plt.ylabel("Accumulated Score", fontsize=20)
        plt.tight_layout()
        # plt.ylim(ymax=-1000)
        # plt.ylim(ymin=-1000)
        if save_Plot:
            plt.savefig(save_path + 'SCORE_'+str(i)+'_NEW.png')
        if show_Plot:
            plt.show()


alpha = 0.4
linewidth = 1
if plot_Percent_2 == True:
    print("\nPLOTTING PERCENT 2")
    for i in env_list:
        if i in percent_env_list:
            print("\nEnv", i)
            Score_list = []

            title_name = "Env " + str(i)
            colors = []
            n=0
            for model in choose_models_list:

                name = names[model]
                if name == "DSRL_object_near ":
                    file_name = "/Test_Env_" + str(i) + "_" + name + "*.xlsx"
                else:
                    file_name = "/Train_Env_" + str(i) + "_" + name + "*.xlsx"

                path_new = path + str(i) + file_name
                print("name:", name)
                for fname in glob.glob(path_new):
                    print("fname:", fname)
                    '''SCORE'''
                Percent = pd.read_excel(fname, parse_cols="B:B", sheetname="Percent")

                rolling = Percent.rolling(window=10)
                rolling_mean = 100 * rolling.mean()

                if n == 0:
                    plot3 = rolling_mean.plot(ls="-",
                                              grid=True,
                                              color=color_dict[name],
                                              linewidth=2,
                                              fontsize=14)
                    plt.tick_params(axis='y', labelleft='on', labelright='on')

                else:
                    rolling_mean.plot(ax=plot3,
                                      ls="-",
                                      grid=True,
                                      color=color_dict[name],
                                      linewidth=2,
                                      fontsize=14)

                color = mpatches.Patch(alpha=0.8, color=color_dict[name], label=sub_name[name])
                print(color_dict[name])
                colors.append(color)
                n += 1

            # plt.title(title_name, fontsize=16)
            axes = plt.gca()
            axes.set_xlim([0, 1000])
            print(colors)
            plt.legend(handles=colors,
                       loc=4,
                       borderaxespad=1,
                       fontsize=14)

            # plt.tick_params(axis='y', labelleft='on', labelright='on')
            plot3.set_xlabel("Episodes", fontsize=20)
            plot3.set_ylabel("% of collected positive objects", fontsize=18)
            plt.yticks(np.arange(0, 105, 10))
            plt.ylim((0, 100))

            # plt.yaxis.tick_right()
            plt.tight_layout()
            if save_Plot:
                plt.savefig(save_path + 'PERCENT_'+str(i)+'_NEW.png')
            if show_Plot:
                plt.show()

#
# if plot_Percent == True:
#     print("\nPLOTTING PERCENT")
#     for i in env_list:
#         if i in percent_env_list:
#             print("Env", i)
#             file_name_list = []
#             for name in names:
#                 file_name = "/Train_Env_" + str(i) + "_" + name + "*.xlsx"
#                 path_new = path + str(i) + file_name
#                 print("name:", name)
#                 for fname in glob.glob(path_new):
#                     print("fname:", fname)
#                     if fname == None:
#                         print("FAILED to Load:", name)
#                     '''PERCENT'''
#                 Percent = pd.read_excel(fname, parse_cols="B:B", sheetname="Percent")
#
#                 title_name = "Env " + str(i)
#                 plot2 = Percent.plot(title=title_name,
#                                      grid=True,
#                                      alpha = 0.6)
#
#                 rolling = Percent.rolling(window=100)
#                 rolling_mean = rolling.mean()
#                 rolling_mean.plot(ax=plot2,
#                                   ls="-",
#                                   grid=True,
#                                   color='red',
#                                   linewidth=2)
#
#                 blue_percent = mpatches.Patch(color='b', label=name)
#                 red_mov_avg = mpatches.Patch(color='red', label=name + '_Mov_Avg')
#                 plt.legend(handles=[blue_percent, red_mov_avg],
#                            loc=4,
#                            borderaxespad=1)
#
#                 plot2.set_xlabel("Episodes")
#                 plot2.set_ylabel("% of collected objects with positive reward")
#                 plt.yticks(np.arange(0, 1.05, 0.1))
#                 plt.ylim((0, 1))
#                 plt.tick_params(axis='y',labelleft='on', labelright='on')
#
#                 if save_Plot:
#                     plt.savefig(save_path + 'PERCENT_Env_' + str(i) + "_" + name + '.png')
#                 if show_Plot:
#                     plt.show()
