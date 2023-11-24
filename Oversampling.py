import pandas as pd
import numpy as np

data = {'docking score': np.random.uniform(1, 10, 1000)}  # 生成1000个随机的连续值
df = pd.DataFrame(data)


bins = np.arange(0, 11, 1)
df['score_bins'] = pd.cut(df['docking score'], bins)


target_scores = [-11, -10, -9, -2, -1]

resampled_data = pd.DataFrame()
for score in target_scores:
    subset = df[df['docking score'].between(score - 0.5, score + 0.5)]
    resampled_subset = subset.sample(n=1000, replace=True)  # your_resample_size是重采样的大小
    resampled_data = pd.concat([resampled_data, resampled_subset])


