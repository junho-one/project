
# 2019 우주전파재난 예측 AI 경진대회

* 최종 7등 (WRMSE : 0.8099)



```python
import pandas as pd
import numpy as np
import math
import gmplot
import statistics
from math import *
import time 
from scipy.stats.stats import pearsonr
import seaborn as sns
import os 
import json
from sklearn.cluster import KMeans
import copy
from sklearn.neighbors import KDTree
from numpy import array
import glob
import datetime
```

# 1. 데이터 불러오기

## 1.1 train_x


```python
print("> 원본 트레이닝 X 데이터")
pd.read_csv("./train/ace_1999.csv").head(18)
```

    > 원본 트레이닝 X 데이터





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year      : year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>doy       : day-of-year.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>hr         : hour of day</td>
    </tr>
    <tr>
      <th>2</th>
      <td>min       : minutes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Np        : Proton Density (cm^-3).</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Tp         : Radial Component of proton temper...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Vp         : Proton Speed (km/s)</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Bgsm_x   : X-component of mag. field in GSM (nT).</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Bgsm_y   : Y-component of mag. field in GSM (nT).</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Bgsm_z   : Z-component of mag. field in GSM (nT).</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Bt          : &lt;|B|&gt; magnetic field magnitude (...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>A value of -9999.9 indicates bad or missing data.</td>
    </tr>
    <tr>
      <th>11</th>
      <td>year doy hr min Np Tp Vp B_gsm_x B_gsm_y B_gsm...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>BEGIN DATA</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1999   1  0  0      7.149  9.2352e+04    406.0...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1999   1  0  1      5.998  8.5859e+04    419.1...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1999   1  0  2      6.211  8.1547e+04    411.9...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1999   1  0  3      6.680  7.2308e+04    405.2...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1999   1  0  4  -9999.900 -9.9999e+03  -9999.9...</td>
    </tr>
  </tbody>
</table>
</div>




```python
print("> 매년 데이터 길이 확인\n")

total_len = 0
for year in range(1999,2014) :
    for dataPath in glob.glob("./train/ace_"+str(year)+".csv") :
        print( dataPath.split("/")[-1] , "길이 :" ,len(pd.read_csv(dataPath)))
        total_len += len(pd.read_csv(dataPath))
        
print("총 길이 : " , total_len)
```

    ace_1999.csv 길이 : 492764
    ace_2000.csv 길이 : 494114
    ace_2001.csv 길이 : 492766
    ace_2002.csv 길이 : 492764
    ace_2003.csv 길이 : 492763
    ace_2004.csv 길이 : 494113
    ace_2005.csv 길이 : 492764
    ace_2006.csv 길이 : 492764
    ace_2007.csv 길이 : 492764
    ace_2008.csv 길이 : 494112
    ace_2009.csv 길이 : 492759
    ace_2010.csv 길이 : 525613
    ace_2011.csv 길이 : 492762
    ace_2012.csv 길이 : 494113
    ace_2013.csv 길이 : 492764
    총 길이 :  7429699



```python
def split(x) :
    return x.split()
csv_list = []
for year in range(1999,2014) :
    print("태양데이터 " , year , "년")
    for dataPath in glob.glob("./train/ace_"+str(year)+".csv") :
        df = pd.read_csv(dataPath)
    
        df = df[11:]
        colsname = df.iloc[0][0].split()
        df = df.drop(df.index[:2])
        df = df.rename(columns = {df.columns[0] : 'all'})


        row_list = list(map(split,df['all'].values))


        sun_df = pd.DataFrame(row_list)
        sun_df.columns = colsname
        
        sun_df['date'] = sun_df['doy'].apply(lambda x : datetime.date(int(sun_df.year[0])-1,12,31) + datetime.timedelta(days = int(x)))
        
        sun_df['hr'] = sun_df['hr'].apply( lambda x : " " + str(x).zfill(2) )
        sun_df['min'] = sun_df['min'].apply( lambda x : ":" + str(x).zfill(2) )
        
        sun_df['date'] = sun_df['date'].apply(lambda x : str(x))
        
        sun_df['date'] += sun_df['hr']
        sun_df['date'] += sun_df['min']
        csv_list.append(sun_df)

merged_df = pd.concat(csv_list)
```

    태양데이터  1999 년
    태양데이터  2000 년
    태양데이터  2001 년
    태양데이터  2002 년
    태양데이터  2003 년
    태양데이터  2004 년
    태양데이터  2005 년
    태양데이터  2006 년
    태양데이터  2007 년
    태양데이터  2008 년
    태양데이터  2009 년
    태양데이터  2010 년
    태양데이터  2011 년
    태양데이터  2012 년
    태양데이터  2013 년



```python
merged_df.to_csv("./train/train_x.csv",  index=False)

print("> 15년 합친 트레이닝 데이터 길이 : ", len(merged_df))
merged_df.head(5)
```

    > 15년 합친 트레이닝 데이터 길이 :  7429504





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>doy</th>
      <th>hr</th>
      <th>min</th>
      <th>Np</th>
      <th>Tp</th>
      <th>Vp</th>
      <th>B_gsm_x</th>
      <th>B_gsm_y</th>
      <th>B_gsm_z</th>
      <th>Bt</th>
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1999</td>
      <td>1</td>
      <td>00</td>
      <td>:00</td>
      <td>7.149</td>
      <td>9.2352e+04</td>
      <td>406.00</td>
      <td>-2.174</td>
      <td>-2.598</td>
      <td>5.550</td>
      <td>6.630</td>
      <td>1999-01-01 00:00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1999</td>
      <td>1</td>
      <td>00</td>
      <td>:01</td>
      <td>5.998</td>
      <td>8.5859e+04</td>
      <td>419.12</td>
      <td>-1.245</td>
      <td>-0.140</td>
      <td>6.558</td>
      <td>6.796</td>
      <td>1999-01-01 00:01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1999</td>
      <td>1</td>
      <td>00</td>
      <td>:02</td>
      <td>6.211</td>
      <td>8.1547e+04</td>
      <td>411.99</td>
      <td>-2.003</td>
      <td>-1.198</td>
      <td>6.306</td>
      <td>6.802</td>
      <td>1999-01-01 00:02</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1999</td>
      <td>1</td>
      <td>00</td>
      <td>:03</td>
      <td>6.680</td>
      <td>7.2308e+04</td>
      <td>405.25</td>
      <td>-3.093</td>
      <td>-2.483</td>
      <td>5.545</td>
      <td>6.854</td>
      <td>1999-01-01 00:03</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1999</td>
      <td>1</td>
      <td>00</td>
      <td>:04</td>
      <td>-9999.900</td>
      <td>-9.9999e+03</td>
      <td>-9999.90</td>
      <td>-3.009</td>
      <td>-1.500</td>
      <td>5.908</td>
      <td>6.842</td>
      <td>1999-01-01 00:04</td>
    </tr>
  </tbody>
</table>
</div>



## 1.2 train_y


```python
print("> 원본 트레이닝 y 데이터")
pd.read_csv("./train/prev_train_y.csv").head(5)
```

    > 원본 트레이닝 y 데이터





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>kp_00h</th>
      <th>kp_03h</th>
      <th>kp_06h</th>
      <th>kp_09h</th>
      <th>kp_12h</th>
      <th>kp_15h</th>
      <th>kp_18h</th>
      <th>kp_21h</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1999-01-01</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1999-01-02</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1999-01-03</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1999-01-04</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1999-01-05</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = pd.read_csv("./train/prev_train_y.csv")

colsname = list(df.columns)[1:]
csv_list = []

for col in colsname :
    
    df_3hour = df[['date',col]]
    df_3hour['date'] = df_3hour['date'].astype('str')
    df_3hour['date'] = df_3hour['date'].apply(lambda x : x + " " + col[3:5] + ":00")
    
    df_3hour = df_3hour.rename({col : "kp"} , axis=1)
    
    csv_list.append(df_3hour)

merged_df = pd.concat(csv_list)

merged_df = merged_df.sort_values(by='date')

merged_df.to_csv("./train/train_y.csv",  index=False)


print("> 테스트 결과값 길이 :",len(merged_df))
merged_df.head()


```

    /usr/lib/python3/dist-packages/ipykernel_launcher.py:9: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      if __name__ == '__main__':
    /usr/lib/python3/dist-packages/ipykernel_launcher.py:10: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      # Remove the CWD from sys.path while we load stuff.


    > 테스트 결과값 길이 : 43832





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>kp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1999-01-01 00:00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1999-01-01 03:00</td>
      <td>2</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1999-01-01 06:00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1999-01-01 09:00</td>
      <td>2</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1999-01-01 12:00</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



# 2. 학습데이터 생성(데이터 가공 등)


원래 측정 데이터에 결측값이 없다면 7884000개가 있어야하지만, 초기 date 데이터에 결측값 존재 => 7429504개

1분 -> 20분으로 가면 7884000 -> 394200



```python
trn_x = pd.read_csv("./train/train_x.csv")
trn_y = pd.read_csv("./train/train_y.csv")
```

## 2.1 선형보간

## - X data

데이터 결측값이 -9999, -9999.9로 입력되어 있다.

1. B 관련 값과 Vp는 결측값이 별로 없어서 결측값들을 모두 삭제
2. Np와 Tp는 결측값을 Nan 값으로 치환
3. 1분->20분으로 변환. 
4. 남은 결측값들은 모두 선형 보간으로 채움


```python
trn_x = pd.read_csv("./train/train_x.csv")

trn_x = trn_x[~((trn_x.Bt <= -9999) & (trn_x.B_gsm_x <= -9999) & (trn_x.B_gsm_y <= -9999) & (trn_x.B_gsm_z <= -9999))]
trn_x = trn_x[~( trn_x.B_gsm_x <= -9999 )]
trn_x = trn_x[~( (trn_x.Vp == -9999.9) | (trn_x.Vp == -9999.9004))]

trn_x['Np'] = trn_x['Np'].replace(-9999.9 , np.nan)
trn_x['Np'] = trn_x['Np'].replace(-9999.9004 , np.nan)

trn_x['Tp'] = trn_x['Tp'].replace(-9999.9 , np.nan)
trn_x['Tp'] = trn_x['Tp'].replace(-9999.9004 , np.nan)

print("=> 총 데이터 개수 : " , len(trn_x))
print("Bt 만 <0 : " ,len(trn_x[(trn_x.Bt <= -9999)]))
print("B_gsm_x 만 <0 : " ,len(trn_x[(trn_x.B_gsm_x <= -9999)]))
print("B_gsm_y 만 <0 : " ,len(trn_x[(trn_x.B_gsm_y <= -9999)]))
print("B_gsm_z 만 <0 : " ,len(trn_x[(trn_x.B_gsm_z <= -9999)]))
print("Np 만 <0 : " ,len(trn_x[(trn_x.Np <= -9999)]))
print("Tp 만 <0 : " ,len(trn_x[(trn_x.Tp <= -9999)]))
print("Vp 만 <0 : " ,len(trn_x[(trn_x.Vp <= -9999)]))



del trn_x['year']
del trn_x['doy']
del trn_x['hr']
del trn_x['min']

trn_x['date'] = pd.to_datetime(trn_x['date'])
trn_x = trn_x.set_index("date")

trn_x = trn_x.resample('10Min').mean()

trn_x = trn_x.reset_index()
```

    => 총 데이터 개수 :  6903149
    Bt 만 <0 :  0
    B_gsm_x 만 <0 :  0
    B_gsm_y 만 <0 :  0
    B_gsm_z 만 <0 :  0
    Np 만 <0 :  0
    Tp 만 <0 :  0
    Vp 만 <0 :  0



```python
print("=> 1min -> 10min 후 결측치 개수")
print(trn_x.isna().sum())

trn_x = trn_x.interpolate(method='linear')
# trn_x = trn_x.interpolate(method='polynomial' , order=2)

print("=> 선형보간 후 결측치 개수")
print(trn_x.isna().sum())
```

    => 1min -> 10min 후 결측치 개수
    date            0
    Np         268942
    Tp         158057
    Vp          15210
    B_gsm_x     15210
    B_gsm_y     15210
    B_gsm_z     15210
    Bt          15210
    dtype: int64
    => 선형보간 후 결측치 개수
    date       0
    Np         0
    Tp         0
    Vp         0
    B_gsm_x    0
    B_gsm_y    0
    B_gsm_z    0
    Bt         0
    dtype: int64


## - Y data


* 3시간 -> 10분
* trn_y는 3시간 간격의 데이터 안의 결측값들은 선형보간으로 채워넣는다


```python
trn_y = pd.read_csv("./train/train_y.csv")

trn_y_bf = trn_y
trn_y_bf['date'] = pd.to_datetime(trn_y_bf['date'])
trn_y_bf = trn_y_bf.set_index('date')
trn_y_bf = trn_y_bf.resample('10Min').mean()

trn_y_bf = trn_y_bf.interpolate(method='linear')
trn_y_bf = trn_y_bf.reset_index()

trn_y_bf.kp = trn_y_bf.kp.apply(lambda x : round(x))

trn_y_bf = trn_y_bf[:-1]

trn_y_bf.head(7)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>kp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1999-01-01 00:00:00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1999-01-01 00:10:00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1999-01-01 00:20:00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1999-01-01 00:30:00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1999-01-01 00:40:00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1999-01-01 00:50:00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1999-01-01 01:00:00</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
train = copy.deepcopy(trn_x)
train_y = trn_y_bf
train['kp'] = train_y['kp']

train = train.loc[train.date <= '2013-12-31 21:00:00']
```


```python
del train['Np']
del train['Tp']
train.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>Vp</th>
      <th>B_gsm_x</th>
      <th>B_gsm_y</th>
      <th>B_gsm_z</th>
      <th>Bt</th>
      <th>kp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1999-01-01 00:00:00</td>
      <td>410.162500</td>
      <td>-2.200625</td>
      <td>-0.268250</td>
      <td>6.012375</td>
      <td>6.844500</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1999-01-01 00:10:00</td>
      <td>415.287000</td>
      <td>-1.886600</td>
      <td>3.638800</td>
      <td>5.200200</td>
      <td>6.766000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1999-01-01 00:20:00</td>
      <td>420.778889</td>
      <td>-1.166333</td>
      <td>3.629778</td>
      <td>5.375000</td>
      <td>6.852333</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1999-01-01 00:30:00</td>
      <td>414.110000</td>
      <td>-1.912000</td>
      <td>0.469500</td>
      <td>6.356875</td>
      <td>6.846125</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1999-01-01 00:40:00</td>
      <td>411.423000</td>
      <td>-1.812600</td>
      <td>0.598000</td>
      <td>5.812300</td>
      <td>6.885000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1999-01-01 00:50:00</td>
      <td>420.181250</td>
      <td>-0.733000</td>
      <td>-0.325500</td>
      <td>5.190500</td>
      <td>6.589000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1999-01-01 01:00:00</td>
      <td>394.442000</td>
      <td>-5.655800</td>
      <td>2.641100</td>
      <td>-0.921800</td>
      <td>6.493700</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1999-01-01 01:10:00</td>
      <td>390.040000</td>
      <td>-6.270750</td>
      <td>0.399375</td>
      <td>1.761500</td>
      <td>6.792375</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1999-01-01 01:20:00</td>
      <td>389.473333</td>
      <td>-5.932222</td>
      <td>0.110889</td>
      <td>2.625667</td>
      <td>6.809556</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1999-01-01 01:30:00</td>
      <td>402.895000</td>
      <td>-4.033400</td>
      <td>-2.301000</td>
      <td>4.824200</td>
      <td>7.029900</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
train.to_csv("./complete/train_date.csv" , index=False)
del train['date']
train.to_csv("./complete/10train.csv" , index=False)
train.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Vp</th>
      <th>B_gsm_x</th>
      <th>B_gsm_y</th>
      <th>B_gsm_z</th>
      <th>Bt</th>
      <th>kp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>410.162500</td>
      <td>-2.200625</td>
      <td>-0.268250</td>
      <td>6.012375</td>
      <td>6.844500</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>415.287000</td>
      <td>-1.886600</td>
      <td>3.638800</td>
      <td>5.200200</td>
      <td>6.766000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>420.778889</td>
      <td>-1.166333</td>
      <td>3.629778</td>
      <td>5.375000</td>
      <td>6.852333</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>414.110000</td>
      <td>-1.912000</td>
      <td>0.469500</td>
      <td>6.356875</td>
      <td>6.846125</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>411.423000</td>
      <td>-1.812600</td>
      <td>0.598000</td>
      <td>5.812300</td>
      <td>6.885000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>420.181250</td>
      <td>-0.733000</td>
      <td>-0.325500</td>
      <td>5.190500</td>
      <td>6.589000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>394.442000</td>
      <td>-5.655800</td>
      <td>2.641100</td>
      <td>-0.921800</td>
      <td>6.493700</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>390.040000</td>
      <td>-6.270750</td>
      <td>0.399375</td>
      <td>1.761500</td>
      <td>6.792375</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>389.473333</td>
      <td>-5.932222</td>
      <td>0.110889</td>
      <td>2.625667</td>
      <td>6.809556</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>402.895000</td>
      <td>-4.033400</td>
      <td>-2.301000</td>
      <td>4.824200</td>
      <td>7.029900</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



# 4. 모델 구성


* da-rnn 코드를 기반으로 내 데이터에 맞춰 수정

http://chandlerzuo.github.io/blog/2017/11/darnn



### 기존 모델의 단점

* 이전 x값을 볼 뿐만 아니라 y_history로 이전 y값을 보기 때문에 한번만 예측하는 것이 아닌 1년 예측을 하기에는 힘들다. 
* 현재 시간의 x와 바로 이전 시간의 x는 안보고 이전 T-2개 스텝의 x를 보고 예측한다.
> 만약 T=9를 넣는다면, [0][1][2][3][4][5][6][7] 을 보고 [9] 예측



### 개선

* y_history를 없애므로써 나의 task에 맞췄다.
* 현재 시간의 x까지 볼 수 있도록 수정해였다. ( + 디코더의 lstm 반복 횟수 수정 )
> 만약 T=7을 넣는다면, [0][1][2][3][4][5][6][7] 을 보고 [7] 예측



```python
train = pd.read_csv("./complete/10train.csv")
train_date = pd.read_csv("./complete/train_date.csv")
```

## 4.1 train-test split

* year에 값을 넣으면 그만큼만 테스트셋으로 찢어줌
> ex) year=2 이면 train 13년, test 2년


```python
def cal_testLen(df , year = 1):
    
    baseTime = str( datetime.datetime.strptime(str(2013 - year) + '-12-31 22:40:00' ,'%Y-%m-%d %H:%M:%S') )
    return len(df.loc[df.date >= baseTime])
```

## 4.2 Train Model


```python
from sklearn.metrics import accuracy_score
import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import matplotlib
import datetime as dt, itertools, pandas as pd, matplotlib.pyplot as plt, numpy as np
from sklearn.preprocessing import StandardScaler
import time
from sklearn.metrics import mean_squared_error



def WRMSE(predictions, answers):
    
    sum_answer = sum(answers)
    
    if sum_answer.data == 0:
        return torch.sqrt(Variable(torch.tensor(0.001, dtype=torch.float), requires_grad=True))
    
    weight = Variable(torch.tensor([answer/sum_answer for answer in answers], dtype=torch.float))

    loss = (weight * (predictions - answers).pow(2)).sum()
    return torch.sqrt(loss)

class encoder(nn.Module):
    def __init__(self, input_size, hidden_size, T):
        
        super(encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.T = T

        self.lstm_layer = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = 1)
        self.attn_linear = nn.Linear(in_features = 2 * hidden_size + T + 1, out_features = 1)

    def forward(self, input_data):
        
        input_weighted = Variable(input_data.data.new(input_data.size(0), self.T + 1, self.input_size).zero_())
        input_encoded = Variable(input_data.data.new(input_data.size(0), self.T + 1, self.hidden_size).zero_())
        
        hidden = self.init_hidden(input_data) 
        cell = self.init_hidden(input_data)
        
        for t in range(self.T + 1):
            x = torch.cat((hidden.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           cell.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           input_data.permute(0, 2, 1)), dim = 2)
            
            
            x = self.attn_linear(x.view(-1, self.hidden_size * 2 + self.T + 1))
            attn_weights = F.softmax(x.view(-1, self.input_size))
            
            weighted_input = torch.mul(attn_weights, input_data[:, t, :]) 
            
            self.lstm_layer.flatten_parameters()
            _, lstm_states = self.lstm_layer(weighted_input.unsqueeze(0), (hidden, cell))
            hidden = lstm_states[0]
            cell = lstm_states[1]
            
            input_weighted[:, t, :] = weighted_input
            input_encoded[:, t, :] = hidden
            
            
        return input_weighted, input_encoded

    def init_hidden(self, x):
        
        return Variable(x.data.new(1, x.size(0), self.hidden_size).zero_())



class decoder(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size, T):
        super(decoder, self).__init__()

        self.T = T
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        
        self.attn_layer = nn.Sequential(nn.Linear(2 * decoder_hidden_size + encoder_hidden_size, encoder_hidden_size),
                                         nn.Tanh(), nn.Linear(encoder_hidden_size, 1))
        self.lstm_layer = nn.LSTM(input_size = 1, hidden_size = decoder_hidden_size)
        self.fc = nn.Linear(encoder_hidden_size, 1)
        self.fc_final = nn.Linear(decoder_hidden_size + encoder_hidden_size, 1)

        self.fc.weight.data.normal_()

    def forward(self, input_encoded):
        
        hidden = self.init_hidden(input_encoded)
        cell = self.init_hidden(input_encoded)
        
        for t in range(self.T + 1):
        
            x = torch.cat((hidden.repeat(self.T + 1, 1, 1).permute(1, 0, 2),
                           cell.repeat(self.T + 1, 1, 1).permute(1, 0, 2), input_encoded), dim = 2)
            x = F.softmax(self.attn_layer(x.view(-1, 2 * self.decoder_hidden_size + self.encoder_hidden_size
                                                )).view(-1, self.T + 1)) 
            
            context = torch.bmm(x.unsqueeze(1), input_encoded)[:, 0, :]
         
            y_tilde = self.fc(context)
            
            self.lstm_layer.flatten_parameters()
            _, lstm_output = self.lstm_layer(y_tilde.unsqueeze(0), (hidden, cell))

            hidden = lstm_output[0]
            cell = lstm_output[1]
        
        y_pred = self.fc_final(torch.cat((hidden[0], context), dim = 1))
        
        return y_pred

    def init_hidden(self, x):
        return Variable(x.data.new(1, x.size(0), self.decoder_hidden_size).zero_())

class da_rnn:
    def __init__(self, file_data,  encoder_hidden_size = 64, decoder_hidden_size = 64, T = 18,
                 learning_rate = 0.01, batch_size = 256, parallel = True, debug = False,  test_length = 10) :
        self.T = T
        
        dat = pd.read_csv(file_data, nrows = 100 if debug else None)
        self.X = dat.loc[:, [x for x in dat.columns.tolist() if x != 'kp']].as_matrix()
        self.y = np.array(dat.kp)
        
        self.debug = debug
        
        self.y  = self.y.flatten()
        self.scalerX = StandardScaler()
        self.X = self.scalerX.fit_transform(self.X)
        
        self.batch_size = batch_size

        self.encoder = encoder(input_size = self.X.shape[1], hidden_size = encoder_hidden_size, T = T,
                              )
        self.decoder = decoder(encoder_hidden_size = encoder_hidden_size,
                               decoder_hidden_size = decoder_hidden_size,
                               T = T, )

        if parallel:
            self.encoder = nn.DataParallel(self.encoder)
            self.decoder = nn.DataParallel(self.decoder)

        self.encoder_optimizer = optim.Adam(params = filter(lambda p: p.requires_grad, self.encoder.parameters()),
                                           lr = learning_rate)
        self.decoder_optimizer = optim.Adam(params = filter(lambda p: p.requires_grad, self.decoder.parameters()),
                                           lr = learning_rate)
        
        # 3.Train 및 validation 셋 분리
        self.train_size = self.X.shape[0] - test_length

    def train(self, n_epochs = 10):
        
        iter_per_epoch = int(np.ceil(self.train_size * 1. / self.batch_size))
        self.iter_losses = np.zeros(n_epochs * iter_per_epoch)
        self.epoch_losses = np.zeros(n_epochs)

        self.loss_func = nn.MSELoss()
        n_iter = 0
        learning_rate = 1.
        
        for i in range(n_epochs):
            
            start = time.time()
            perm_idx = np.random.permutation(self.train_size - self.T)
            
            j = 0
            while j < self.train_size:
                batch_idx = perm_idx[j:(j + self.batch_size)]
                X = np.zeros((len(batch_idx), self.T + 1, self.X.shape[1]))
                y_target = self.y[batch_idx + self.T] 
                
                for k in range(len(batch_idx)):
            
                    X[k, :, :] = self.X[batch_idx[k] : (batch_idx[k] + self.T + 1), :]
                
                loss = self.train_iteration(X, y_target)
                
                self.iter_losses[int(i * iter_per_epoch + j / self.batch_size)] = loss

                j += self.batch_size
                n_iter += 1

                if n_iter % 10000 == 0 and n_iter > 0:
                    for param_group in self.encoder_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.9
                    for param_group in self.decoder_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.9

            self.epoch_losses[i] = np.mean(self.iter_losses[range(i * iter_per_epoch, (i + 1) * iter_per_epoch)])
           
            end = time.time()
            print("Epoch ",i,", loss: ",self.epoch_losses[i],".")
            print("걸린 시간 :",end-start)

            if i % 4 == 0 and i != 0:
                y_train_pred = self.predict(on_train = True)
                y_test_pred = self.predict(on_train = False)
                
                y_test_true = self.y[self.train_size:]
                y_train_true = self.y[:self.train_size - self.T + 1]
                
                y_train_pred = torch.from_numpy(y_train_pred).float()
                y_test_pred = torch.from_numpy(y_test_pred).float()
                y_train_true = torch.from_numpy(y_train_true).float()
                y_test_true = torch.from_numpy(y_test_true).float()
                
                Wrmse_train = WRMSE(y_train_pred , y_train_true)
                Wrmse_test = WRMSE(y_test_pred , y_test_true)
                
                print("Epoch::",i," TRAIN WRMSE " , Wrmse_train , "TEST WRMSE ", Wrmse_test)
            
    def train_iteration(self, X,  y_target):
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        input_weighted, input_encoded = self.encoder(Variable(torch.from_numpy(X).type(torch.FloatTensor)))
                
        y_pred = self.decoder(input_encoded)   
        y_true = Variable(torch.from_numpy(y_target).type(torch.FloatTensor) )
            
        y_pred = y_pred.flatten()
        
        loss = WRMSE(y_pred , y_true)
        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        
        return loss.data

    def predict(self, on_train = False , inverse_scailing = False):
        
        if on_train:
            y_pred = np.zeros(self.train_size - self.T + 1)
        else:
            y_pred = np.zeros(self.X.shape[0] - self.train_size)

        i = 0
        while i < len(y_pred):
            batch_idx = np.array(range(len(y_pred)))[i : (i + self.batch_size)]
            X = np.zeros((len(batch_idx), self.T + 1, self.X.shape[1]))

            for j in range(len(batch_idx)):
                if on_train:
                    X[j, :, :] = self.X[range(batch_idx[j], batch_idx[j] + self.T + 1), :]
                else:
                    X[j, :, :] = self.X[range(batch_idx[j] + self.train_size - self.T, batch_idx[j] + self.train_size + 1), :]
            _, input_encoded = self.encoder(Variable(torch.from_numpy(X).type(torch.FloatTensor)))
            y_pred[i:(i + self.batch_size)] = self.decoder(input_encoded).cpu().data.numpy()[:, 0]
            i += self.batch_size
        return y_pred
    
    
    def predict_test(self, on_train = False , inverse_scailing = False):
        
        if on_train:
            y_pred = np.zeros(self.train_size - self.T + 1)
        else:
            y_pred = np.zeros(self.X.shape[0] - self.train_size)

        i = 0
        while i < len(y_pred):
            batch_idx = np.array(range(len(y_pred)))[i : (i + self.batch_size)]
            X = np.zeros((len(batch_idx), self.T + 1, self.X.shape[1]))

            for j in range(len(batch_idx)):
                if on_train:
                    X[j, :, :] = self.X[range(batch_idx[j], batch_idx[j] + self.T + 1), :]
                else:
                    X[j, :, :] = self.X[range(batch_idx[j] + self.train_size - self.T, batch_idx[j] + self.train_size + 1), :]
            _, input_encoded = self.encoder(Variable(torch.from_numpy(X).type(torch.FloatTensor)))
            y_pred[i:(i + self.batch_size)] = self.decoder(input_encoded).cpu().data.numpy()[:, 0]
            i += self.batch_size
        return y_pred


    def predict_year(self ,tst_len, tst_path):
        
        y_pred = np.zeros(tst_len) # 26283
        
        tst_x = pd.read_csv(tst_path, nrows = 100 if self.debug else None)
        tst_x = tst_x.loc[:, [x for x in tst_x.columns.tolist() if x != 'kp']].as_matrix()
                
        tst_x = self.scalerX.fit_transform(tst_x)
        
        i = 0
        while i < len(y_pred):
            batch_idx = np.array(range(len(y_pred)))[i : (i + self.batch_size)]
            X = np.zeros((len(batch_idx), self.T + 1, tst_x.shape[1]))
            
            for j in range(len(batch_idx)):
            
                X[j, :, :] = tst_x[range(batch_idx[j], batch_idx[j] + self.T + 1), :]
            
            _, input_encoded = self.encoder(Variable(torch.from_numpy(X).type(torch.FloatTensor)))
            y_pred[i:(i + self.batch_size)] = self.decoder(input_encoded).cpu().data.numpy()[:, 0]
            i += self.batch_size
            
        return y_pred      
        

start = time.time()



test_length = cal_testLen(train_date,year = 1)

model = da_rnn(file_data = "./complete/10train.csv".format(),  parallel = False, learning_rate = .001 , test_length = test_length ,T=11)


model.train(n_epochs = 1)

torch.save(model, './model/final.pt')


end = time.time()

print("time:",end-start)


```

    /usr/lib/python3/dist-packages/ipykernel_launcher.py:125: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
    /usr/lib/python3/dist-packages/ipykernel_launcher.py:53: UserWarning: Implicit dimension choice for softmax has been deprecated. Change the call to include dim=X as an argument.
    /usr/lib/python3/dist-packages/ipykernel_launcher.py:100: UserWarning: Implicit dimension choice for softmax has been deprecated. Change the call to include dim=X as an argument.


    Epoch  0 , loss:  0.8016326714399005 .
    걸린 시간 : 414.12905526161194
    time: 415.26172852516174


    /usr/local/lib/python3.6/dist-packages/torch/serialization.py:251: UserWarning: Couldn't retrieve source code for container of type encoder. It won't be checked for correctness upon loading.
      "type " + obj.__name__ + ". It won't be checked "
    /usr/local/lib/python3.6/dist-packages/torch/serialization.py:251: UserWarning: Couldn't retrieve source code for container of type decoder. It won't be checked for correctness upon loading.
      "type " + obj.__name__ + ". It won't be checked "


## 4.3 Load Model

* 24 epoch를 돌린 모델 loading


```python
model = torch.load("./model/final_24.pt")
```

# 5. 모델 평가

## 5.1 Train

* accuracy와 wrmse로 평가


```python
import time
a = time.time()
train_pred = model.predict(on_train = True)

trn_pred = [train_pred[0]] + list(train_pred[list(range(8,len(train_pred),18))])
trn_pred = np.round(trn_pred)
print("걸린 시간 " , time.time() - a)
```

    /usr/lib/python3/dist-packages/ipykernel_launcher.py:53: UserWarning: Implicit dimension choice for softmax has been deprecated. Change the call to include dim=X as an argument.
    /usr/lib/python3/dist-packages/ipykernel_launcher.py:100: UserWarning: Implicit dimension choice for softmax has been deprecated. Change the call to include dim=X as an argument.


    걸린 시간  441.18391847610474



```python
trn_y = pd.read_csv("./train/train_y.csv")
trn_y = list(trn_y[:-2920].kp)
print(max(trn_pred))
print("train wrmse :", WRMSE(torch.tensor(trn_pred).float(),torch.tensor(list(trn_y)).float()))
print("train accuracy : ",accuracy_score(list(trn_pred),list(trn_y)))
```

    10.0
    train wrmse : tensor(0.7505)
    train accuracy :  0.42366542823621434


## 5.2 Test

* accuracy와 wrmse로 평가


```python
pred = model.predict_test()

pred = pred[list(range(8,len(train_date[-test_length:]),18))]

y = pd.read_csv("./train/train_y.csv")
tst_y = y[-2921:-1]
del tst_y['date']

print(len(pred),len(tst_y))

pred = np.round(pred)

print("test wrmse :",WRMSE(torch.tensor(pred).float(),torch.tensor(list(tst_y.kp)).float()))


print("test accuracy : ",accuracy_score(list(pred),list(tst_y.kp)))
```

    /usr/lib/python3/dist-packages/ipykernel_launcher.py:53: UserWarning: Implicit dimension choice for softmax has been deprecated. Change the call to include dim=X as an argument.
    /usr/lib/python3/dist-packages/ipykernel_launcher.py:100: UserWarning: Implicit dimension choice for softmax has been deprecated. Change the call to include dim=X as an argument.


    2920 2920
    test wrmse : tensor(0.8350)
    test accuracy :  0.35136986301369866


# 6. 2차대회 문제 불러오기 및 가공 

* 위에서 train, test set 가공했던 것과 같은 방식으로 진행


```python
import pandas as pd 
import datetime

df = pd.read_csv("./complete/final_test.csv")

df['date'] = df['doy'].apply(lambda x : datetime.date(2013,12,31) + datetime.timedelta(days = int(x)))

df['hr'] = df['hr'].apply( lambda x : " " + str(x).zfill(2) )
df['min'] = df['min'].apply( lambda x : ":" + str(x).zfill(2) )
        
df['date'] = df['date'].apply(lambda x : str(x))
        
df['date'] += df['hr']
df['date'] += df['min']

del df['doy']
del df['hr']
del df['min']

df.to_csv("./complete/final_test_x.csv",  index=False)
df.head(7)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Np</th>
      <th>Tp</th>
      <th>Vp</th>
      <th>B_gsm_x</th>
      <th>B_gsm_y</th>
      <th>B_gsm_z</th>
      <th>Bt</th>
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-9999.9004</td>
      <td>64552.640</td>
      <td>383.74547</td>
      <td>-0.419527</td>
      <td>5.537155</td>
      <td>-0.160227</td>
      <td>5.557684</td>
      <td>2014-01-01 00:00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-9999.9004</td>
      <td>64750.522</td>
      <td>382.48322</td>
      <td>-0.839191</td>
      <td>5.388721</td>
      <td>-0.513893</td>
      <td>5.480231</td>
      <td>2014-01-01 00:01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-9999.9004</td>
      <td>63637.725</td>
      <td>380.81279</td>
      <td>-1.072117</td>
      <td>5.481408</td>
      <td>-0.855559</td>
      <td>5.655600</td>
      <td>2014-01-01 00:02</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-9999.9004</td>
      <td>61602.074</td>
      <td>379.03508</td>
      <td>-1.294294</td>
      <td>5.478683</td>
      <td>-0.231456</td>
      <td>5.646056</td>
      <td>2014-01-01 00:03</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-9999.9004</td>
      <td>65951.954</td>
      <td>380.47019</td>
      <td>-0.656898</td>
      <td>5.529207</td>
      <td>-0.439082</td>
      <td>5.593700</td>
      <td>2014-01-01 00:04</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-9999.9004</td>
      <td>67784.638</td>
      <td>384.06392</td>
      <td>-0.482104</td>
      <td>5.508895</td>
      <td>-0.398991</td>
      <td>5.549438</td>
      <td>2014-01-01 00:05</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-9999.9004</td>
      <td>74699.698</td>
      <td>387.32278</td>
      <td>-0.113405</td>
      <td>5.473725</td>
      <td>0.266117</td>
      <td>5.487899</td>
      <td>2014-01-01 00:06</td>
    </tr>
  </tbody>
</table>
</div>




```python
trn_x = pd.read_csv("./complete/final_test_x.csv")

trn_x = trn_x[~((trn_x.Bt <= -9999) & (trn_x.B_gsm_x <= -9999) & (trn_x.B_gsm_y <= -9999) & (trn_x.B_gsm_z <= -9999))]
trn_x = trn_x[~( trn_x.B_gsm_x <= -9999 )]
trn_x = trn_x[~( (trn_x.Vp == -9999.9) | (trn_x.Vp == -9999.9004))]

trn_x['Np'] = trn_x['Np'].replace(-9999.9 , np.nan)
trn_x['Np'] = trn_x['Np'].replace(-9999.9004 , np.nan)

trn_x['Tp'] = trn_x['Tp'].replace(-9999.9 , np.nan)
trn_x['Tp'] = trn_x['Tp'].replace(-9999.9004 , np.nan)

print("=> 총 데이터 개수 : " , len(trn_x))
print("Bt 만 <0 : " ,len(trn_x[(trn_x.Bt <= -9999)]))
print("B_gsm_x 만 <0 : " ,len(trn_x[(trn_x.B_gsm_x <= -9999)]))
print("B_gsm_y 만 <0 : " ,len(trn_x[(trn_x.B_gsm_y <= -9999)]))
print("B_gsm_z 만 <0 : " ,len(trn_x[(trn_x.B_gsm_z <= -9999)]))
print("Np 만 <0 : " ,len(trn_x[(trn_x.Np <= -9999)]))
print("Tp 만 <0 : " ,len(trn_x[(trn_x.Tp <= -9999)]))
print("Vp 만 <0 : " ,len(trn_x[(trn_x.Vp <= -9999)]))


trn_x['date'] = pd.to_datetime(trn_x['date'])
trn_x = trn_x.set_index("date")

trn_x = trn_x.resample('10Min').mean()

trn_x = trn_x.reset_index()

print("=> 1min -> 10min 후 결측치 개수")
print(trn_x.isna().sum())

trn_x = trn_x.interpolate(method='linear')
# trn_x = trn_x.interpolate(method='polynomial' , order=2)

del trn_x['Np']
del trn_x['Tp']
print("=> 선형보간 후 결측치 개수")
print(trn_x.isna().sum())

del trn_x['date']


trn_x.to_csv("./complete/test_10m.csv" , index=False)
```

    => 총 데이터 개수 :  937060
    Bt 만 <0 :  0
    B_gsm_x 만 <0 :  0
    B_gsm_y 만 <0 :  0
    B_gsm_z 만 <0 :  0
    Np 만 <0 :  0
    Tp 만 <0 :  0
    Vp 만 <0 :  0
    => 1min -> 10min 후 결측치 개수
    date           0
    Np         46930
    Tp          3628
    Vp           826
    B_gsm_x      826
    B_gsm_y      826
    B_gsm_z      826
    Bt           826
    dtype: int64
    => 선형보간 후 결측치 개수
    date       0
    Vp         0
    B_gsm_x    0
    B_gsm_y    0
    B_gsm_z    0
    Bt         0
    dtype: int64


# 7. 2년치 예측값(kp) 산출


```python
# test_10m.csv
lasttst_x = pd.read_csv("./complete/test_10m.csv")
print(len(lasttst_x)) 
lasttst_x
z = model.predict_year(tst_len = len(lasttst_x)-11,  tst_path = "./complete/test_10m.csv")

zz = ([z[0]] * 11) + list(z)
print(len(zz)/18)
```

    105120


    /usr/lib/python3/dist-packages/ipykernel_launcher.py:287: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
    /usr/lib/python3/dist-packages/ipykernel_launcher.py:53: UserWarning: Implicit dimension choice for softmax has been deprecated. Change the call to include dim=X as an argument.
    /usr/lib/python3/dist-packages/ipykernel_launcher.py:100: UserWarning: Implicit dimension choice for softmax has been deprecated. Change the call to include dim=X as an argument.


    5840.0



```python
q = pd.DataFrame(zz , columns=['pred'])
real_pred = list(q.iloc[list(range(0,len(q),18))].pred)
real_pred = np.round(real_pred)
print(max(real_pred))
print("길이 :",len(real_pred))
```

    9.0
    길이 : 5840



```python
ans = np.array(real_pred)

ans = ans.reshape(-1,8)

ans = pd.DataFrame(ans)

print("=> 결과 저장")
ans.to_excel("./finalAns.xlsx" , index=False)
ans.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>4.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
