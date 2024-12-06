import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv("comp_results_ksom_cpu.csv").set_index("dim")
df2 = pd.read_csv("comp_results_ksom_gpu.csv").set_index("dim")
df3 = pd.read_csv("comp_results_minisom.csv").set_index("dim")

df = pd.concat((df1, df2, df3), axis=1)
df.plot()
plt.show()
