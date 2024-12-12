import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv("comp_results_ksom_cpu.csv").rename(columns={"time": "ksom CPU"}).set_index("dim")
df2 = pd.read_csv("comp_results_ksom_gpu.csv").rename(columns={"time": "ksom GPU"}).set_index("dim")
df3 = pd.read_csv("comp_results_quicksom_cpu.csv").rename(columns={"time": "quicksom CPU"}).set_index("dim")
df4 = pd.read_csv("comp_results_quicksom_gpu.csv").rename(columns={"time": "quicksom GPU"}).set_index("dim")
df5 = pd.read_csv("comp_results_minisom.csv").rename(columns={"time": "minisom"}).set_index("dim")
df6 = pd.read_csv("comp_results_sklearn_som.csv").rename(columns={"time": "sklearn_som"}).set_index("dim")


plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 14
df = pd.concat((df1, df2, df3, df4, df5, df6), axis=1)
ax = df.plot(colormap="Dark2", linewidth=1.5)
ax.set_xlabel("Dimension of data/SOM weights")
ax.set_ylabel("Training time in seconds")
plt.show()
