import pandas as pd

df = pd.read_fwf("C:/Users/danie/Downloads/clean_glass.txt")
df.to_csv('con.csv')
