# importing libraries
import pandas as pd
import glob
import os
import seaborn as poseidon
import matplotlib.pyplot as plt
  




def plotLine(label,fileName,filePath): #alpha performance
    # merging the files
    joined_files = os.path.join(filePath, fileName)
    
    # A list of all joined files is returned
    joined_list = glob.glob(joined_files)
    
    # Finally, the files are joined
    df = pd.concat(map(pd.read_csv, joined_list), ignore_index=True)

    # #drops the unnamed trash column
    df.drop('Unnamed: 0', inplace=True, axis=1)
    df["method"] = label

    
    return df



def main():
    factors = [0.001]
    connectomes = ['human_func_scale250']#macaque_modha', 'celegans','mouse']
    normalizations = ['False']
    methods = ['empirical', 'rewire']#,'reverse']
    folder = ('/home/mingzeli/neuro/neuromorphicnetworks/'+ 'human_results/tsk_results/reliability/')

    for connectome in connectomes:
        for normalization in normalizations:
                #for method in methods:
                small_dfs = []
                large_df = []               
                for method in methods:
                    small_dfs.append(plotLine(method,(method+'_'+ str(0.001)+'_'+ connectome+'_'+ normalization +'*.csv'), folder))

                large_df = pd.concat(small_dfs, ignore_index=True)
                #poseidon.color_palette('gist_ncar')
                print(large_df)
                large_df.to_csv('/home/mingzeli/neuro/neuromorphicnetworks/analysis/2022/large_df.csv')

                custom_palette = poseidon.color_palette("Set1", 2)
                poseidon.lineplot(x="alpha", y="performance", data= large_df, hue = 'method',palette = custom_palette, err_style = 'band')

                picName = (connectome +'_'+ normalization + ".png")
                plt.title(picName)

                plt.grid(True, which="both", linestyle='--')
                plt.legend(labels=methods)
                plt.savefig("/home/mingzeli/neuro/neuromorphicnetworks/analysis/2022/"+picName, dpi = 300)
                plt.clf()

if __name__ == '__main__':
    main()