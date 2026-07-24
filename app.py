import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_parquet("Task_000200_Put_PotLib_Lift13_CBG_lerobot_fail_example/meta/frame_reuse.parquet")

# 예: 에피소드별 target_frame_index 분포를 바이올린 플롯으로 시각화
plt.figure(figsize=(10, 5))
sns.violinplot(data=df, x="camera", y="target_frame_index")
plt.title("Camera vs Target Frame Index")
plt.show()