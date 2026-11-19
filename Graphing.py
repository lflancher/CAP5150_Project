import pandas as pd
import matplotlib.pyplot as plt

traditional_df = pd.read_csv('Results/Traditional_Implementation.csv')
llm_df = pd.read_csv('Results/LLM_Implementation.csv')

full_df = pd.merge(traditional_df, llm_df, on='Statistic', how='outer', suffixes=('_Traditional', '_LLM'))

full_df.to_csv('Results/Full_Implementation.csv', index=False)

rawmean = full_df.loc[full_df['Statistic'] == 'Mean', 'Non-Private'].values[0]
rawmedian = full_df.loc[full_df['Statistic'] == 'Median', 'Non-Private'].values[0]
rawstd = full_df.loc[full_df['Statistic'] == 'STD', 'Non-Private'].values[0]
rawtwentyfifth = full_df.loc[full_df['Statistic'] == '25th Percentile', 'Non-Private'].values[0]
rawseventyfifth = full_df.loc[full_df['Statistic'] == '75th Percentile', 'Non-Private'].values[0]
rawmin = full_df.loc[full_df['Statistic'] == 'Minimum', 'Non-Private'].values[0]
rawmax = full_df.loc[full_df['Statistic'] == 'Maximum', 'Non-Private'].values[0]

tradmean = full_df.loc[full_df['Statistic'] == 'Mean', 'Private_Traditional'].values[0]
tradmedian = full_df.loc[full_df['Statistic'] == 'Median', 'Private_Traditional'].values[0]
tradstd = full_df.loc[full_df['Statistic'] == 'STD', 'Private_Traditional'].values[0]
tradtwentyfifth = full_df.loc[full_df['Statistic'] == '25th Percentile', 'Private_Traditional'].values[0]
tradseventyfifth = full_df.loc[full_df['Statistic'] == '75th Percentile', 'Private_Traditional'].values[0]
tradmin = full_df.loc[full_df['Statistic'] == 'Minimum', 'Private_Traditional'].values[0]
tradmax = full_df.loc[full_df['Statistic'] == 'Maximum', 'Private_Traditional'].values[0]

tradavgmean = full_df.loc[full_df['Statistic'] == 'Mean', 'Average_Private_Traditional'].values[0]
tradavgmedian = full_df.loc[full_df['Statistic'] == 'Median', 'Average_Private_Traditional'].values[0]
tradavgstd = full_df.loc[full_df['Statistic'] == 'STD', 'Average_Private_Traditional'].values[0]
tradavgtwentyfifth = full_df.loc[full_df['Statistic'] == '25th Percentile', 'Average_Private_Traditional'].values[0]
tradavgseventyfifth = full_df.loc[full_df['Statistic'] == '75th Percentile', 'Average_Private_Traditional'].values[0]
tradavgmin = full_df.loc[full_df['Statistic'] == 'Minimum', 'Average_Private_Traditional'].values[0]
tradavgmax = full_df.loc[full_df['Statistic'] == 'Maximum', 'Average_Private_Traditional'].values[0]

llmmean = full_df.loc[full_df['Statistic'] == 'Mean', 'LLM_Private'].values[0]
llmaedian = full_df.loc[full_df['Statistic'] == 'Median', 'LLM_Private'].values[0]
llmstd = full_df.loc[full_df['Statistic'] == 'STD', 'LLM_Private'].values[0]
llmtwentyfifth = full_df.loc[full_df['Statistic'] == '25th Percentile', 'LLM_Private'].values[0]
llmseventyfifth = full_df.loc[full_df['Statistic'] == '75th Percentile', 'LLM_Private'].values[0]
llmmin = full_df.loc[full_df['Statistic'] == 'Minimum', 'LLM_Private'].values[0]
llmmax = full_df.loc[full_df['Statistic'] == 'Maximum', 'LLM_Private'].values[0]

llmavgmean = full_df.loc[full_df['Statistic'] == 'Mean', 'Average_LLM_Private'].values[0]
llmavgmedian = full_df.loc[full_df['Statistic'] == 'Median', 'Average_LLM_Private'].values[0]
llmavgstd = full_df.loc[full_df['Statistic'] == 'STD', 'Average_LLM_Private'].values[0]
llmavgtwentyfifth = full_df.loc[full_df['Statistic'] == '25th Percentile', 'Average_LLM_Private'].values[0]
llmavgseventyfifth = full_df.loc[full_df['Statistic'] == '75th Percentile', 'Average_LLM_Private'].values[0]
llmavgmin = full_df.loc[full_df['Statistic'] == 'Minimum', 'Average_LLM_Private'].values[0]
llmavgmax = full_df.loc[full_df['Statistic'] == 'Maximum', 'Average_LLM_Private'].values[0]

thestats = [
    {
        'label': 'Non-Private Data',
        'mean': rawmean,
        'med': rawmedian,
        'q1': rawtwentyfifth,
        'q3': rawseventyfifth,
        'whislo': rawmin,
        'whishi': rawmax,
        'fliers': []
    }, 

    {
        'label': 'Private Traditional Data',
        'mean': tradmean,
        'med': tradmedian,
        'q1': tradtwentyfifth,
        'q3': tradseventyfifth,
        'whislo': tradmin,
        'whishi': tradmax,
        'fliers': []
    },
    {
        'label': 'Average Private Traditional Data',
        'mean': tradavgmean,
        'med': tradavgmedian,
        'q1': tradavgtwentyfifth,
        'q3': tradavgseventyfifth,
        'whislo': tradavgmin,
        'whishi': tradavgmax,
        'fliers': []
    },

    {
        'label': 'Private LLM Data',
        'mean': llmmean,
        'med': llmaedian,
        'q1': llmtwentyfifth,
        'q3': llmseventyfifth,
        'whislo': llmmin,
        'whishi': llmmax,
        'fliers': []
    },

    {
        'label': 'Private Average LLM Data',
        'mean': llmavgmean,
        'med': llmavgmedian,
        'q1': llmavgtwentyfifth,
        'q3': llmavgseventyfifth,
        'whislo': llmavgmin,
        'whishi': llmavgmax,
        'fliers': []
    }

]

fix, ax = plt.subplots(figsize=(10, 6))

ax.bxp(
    thestats,
    showmeans=True,
    meanline=True,
    patch_artist=True,
    boxprops=dict(facecolor='lightblue', color='blue'),
    medianprops=dict(color='red'),
    whiskerprops=dict(color='blue'),
    capprops=dict(color='blue'),
    flierprops=dict(markerfacecolor='red', markeredgecolor='red')
)

ax.set_title('Comparison of Non-Private and Private Data Statistics with PyDP')
ax.set_ylabel('Fare')

plt.show()