import pandas as pd,matplotlib.pyplot as plt,seaborn as sns,numpy as np

df = pd.read_csv('heart_disease_uci.csv')

print(df.head())
print(df.info())
print(df.isnull().sum())
gender_column = {'Female':0,'Male':1}
cp_colum = {0:'Typical Angina',1:'Atypical Angina',2:'Non-anginal Pain',3:'Asymptomatic'}
fbs_column = {0:'No',1:'Yes'}
exang_column = {0:'No',1:'Yes'}
thal_column =  {'normal':0,'fixed defect':1,'reversable defect':2}
df['ca']=df['ca'].map(cp_colum)
df['sex'] = df['sex'].map(gender_column)
df['fbs'] = df['fbs'].map(fbs_column)
df['exang'] = df['exang'].map(exang_column)
df['thal'] = df['thal'].map(thal_column)
print(df.head())
df['AgeGroup'] = df['age'].apply(lambda x:'Adult' if x<40 else('Middle-Aged' if 40<=x<=60 else('Senior ' if 61<=x<=130 else 'kid or invalid')))
fig,axes = plt.subplots(2,3,figsize=(12,6))
sns.countplot(data=df,x='num',palette='Set2',ax=axes[0,0])
axes[0,0].set_title('Heart Diseases ')
sns.countplot(data=df,x='sex',hue='num',palette='Set2',ax=axes[0,1])
axes[0,1].set_title('gender vs heart diseases')
sns.countplot(data=df,x='cp',hue='num',palette='Set2',ax=axes[0,2])
axes[0,2].set_title('Chest Pain vs heart diseases')
sns.histplot(data=df,x='age',hue='num',kde=True,ax=axes[1,0])
axes[1,0].set_title('Age Distribution')
sns.boxplot(data=df,x='num',y='trestbps',ax=axes[1,1])
axes[1,1].set_title('heart diseases vs bp status')
sns.boxplot(data=df,x='num',y='chol',ax=axes[1,2])
axes[1,2].set_title('heart diseases vs cholestrol')
plt.tight_layout()
plt.savefig('plots/countplot_1.png')
plt.show()
print(df.groupby('thal')['num'].mean())
def get_risk_level(row):
    risk = 0
    if row['age'] > 50:
        risk+=1
    if row['trestbps']>130:
        risk+= 1
    if row['chol'] > 240:
        risk+=1
    
    if risk==3:
            return "High"
    if risk == 2:
        return "Medium"
    else:
        return 'Low'
df['RiskLevel'] = df.apply(get_risk_level,axis=1)
print(df['RiskLevel'])
risk_counts = df['RiskLevel'].value_counts()
plt.pie(risk_counts,labels=risk_counts.index,autopct="%1.1f%%",startangle=90)
plt.axis(True)
plt.show()

sns.lineplot(data=df,x='age',y='thalch',marker='x')
plt.title("Age vs MAX HEART RATE")
plt.show()

fig,axes = plt.subplots(1,3,figsize=(15,5))
sns.boxplot(data=df,x='age',y='trestbps',hue='RiskLevel',ax=axes[0])
axes[0].set_title('Age vs BP')
sns.boxplot(data=df,x='age',y='chol',hue='RiskLevel',ax=axes[1])
axes[1].set_title('Age vs Cholestrol')
sns.boxplot(data=df,x='age',y='thalch',hue='RiskLevel',ax=axes[2])
axes[2].set_title('Age vs Max Heart Rate')
plt.tight_layout()
plt.savefig('plots/boxplot_1.png')
plt.show()

pivoted  = df.groupby('age')[['trestbps','chol','thalch','oldpeak','num']].mean()
print(pivoted)
sns.heatmap(data=pivoted,annot=True,cmap='coolwarm')
plt.title('Comparision between age vs BP,cholestrol,MAX heart rate,oldpeak,Danger range')
plt.savefig('plots/heatmap_1.png')
plt.show()
