import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('sample_data.csv')
df = pd.DataFrame(data)

bins = np.arange(0, 11, 1)
df['score_bins'] = pd.cut(df['docking score'], bins)
target_scores = [-11, -10, -9, -2, -1]

resampled_data = pd.DataFrame()
for score in target_scores:
    subset = df[df['docking score'].between(score - 0.5, score + 0.5)]
    resampled_subset = subset.sample(n=2000, replace=True)
    resampled_data = pd.concat([resampled_data, resampled_subset])


df_all = pd.concat([resampled_data, df], axis=0)
plt.hist(df_all['docking score'], bins=20)
plt.show()
df_all.to_csv('resampled_data.csv', index=False)




