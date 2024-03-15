import numpy as np
import pandas as pd
df=pd.read_csv('titanic.csv')
print(df.info(),"\n")

print(df.shape,"\n")
print(df.columns,"\n")
print(df.dtypes,"\n")

print(df.pivot_table(df,index=['sex','age'],aggfunc=np.sum),"\n")

print(df.groupby(['class','sex'])['survived'].aggregate('mean').unstack(),"\n")

print(df.groupby('sex')['survived'].mean(),'\n')

print(df.pivot_table('survived',['sex','age'],'class'),"\n")

age=pd.cut(df['age'],[0,10,30,60,80])
print(df.pivot_table('survived',['sex',age],'class'),"\n")

age=pd.cut(df['age'],3)
print(df.pivot_table('survived',['sex',age],'class'),"\n")

fare=pd.cut(df['fare'],2)
print(df.pivot_table('survived',['sex',age],['class',fare]),"\n")