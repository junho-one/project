
# 타이타닉

## 78% accuracy

* passengerId = 승객번호
* survival 
* pclass = 티켓 클래스
* name
* sex = 성
* age = 나이
* sibsp = 배에 탄 형제나 배우자의 수
* parch = 배에 탄 부모나 자식의 수 
* ticket = 티켓 번호
* fare = 낸 돈
* cabin = 객실 번호 
* embarked = 승선 한 항구 ( C = Cherbourg, Q = Queenstown, S = Southampton)


```python
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
data = pd.read_csv('./data/train.csv')
X_test = pd.read_csv('./data/test.csv')
test_PassengerId = X_test['PassengerId']

```

# 1. 데이터 탐색


```python
data.head()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



## 1.1 데이터 타입 검사


```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
    PassengerId    891 non-null int64
    Survived       891 non-null int64
    Pclass         891 non-null int64
    Name           891 non-null object
    Sex            891 non-null object
    Age            714 non-null float64
    SibSp          891 non-null int64
    Parch          891 non-null int64
    Ticket         891 non-null object
    Fare           891 non-null float64
    Cabin          204 non-null object
    Embarked       889 non-null object
    dtypes: float64(2), int64(5), object(5)
    memory usage: 83.6+ KB


## 1.2 간단한 통계 정보 


```python
data.describe()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>714.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>446.000000</td>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>29.699118</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>32.204208</td>
    </tr>
    <tr>
      <th>std</th>
      <td>257.353842</td>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>14.526497</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>49.693429</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>223.500000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>20.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.910400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>446.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>668.500000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>38.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>891.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>



## 1.3 시각화


```python
plt.figure(figsize=(12, 9))
sns.heatmap(data.corr(), annot=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fc7f86f93c8>




![png](.//mdimage/output_10_1.png)



```python
numerical_features = data.select_dtypes(include=[np.number]).columns
categorical_features = data.select_dtypes(include=[np.object]).columns

print(numerical_features)
print(categorical_features)


sns.pairplot(data[numerical_features] ,   kind='reg', 
             diag_kws={'edgecolor':'r'} , plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.1}})
```

    Index(['PassengerId', 'Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare'], dtype='object')
    Index(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], dtype='object')


    /usr/local/lib/python3.6/dist-packages/numpy/lib/histograms.py:824: RuntimeWarning: invalid value encountered in greater_equal
      keep = (tmp_a >= first_edge)
    /usr/local/lib/python3.6/dist-packages/numpy/lib/histograms.py:825: RuntimeWarning: invalid value encountered in less_equal
      keep &= (tmp_a <= last_edge)





    <seaborn.axisgrid.PairGrid at 0x7fc7ee1d2198>




![png](.//mdimage/output_11_3.png)



```python
sns.countplot('Pclass',hue='Survived',data=data)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fc7ee1d2a58>




![png](.//mdimage/output_12_1.png)



```python
All_train = data

survived = All_train.loc[All_train['Survived']==1]
dead = All_train.loc[All_train['Survived']==0]

```


```python
plt.hist([survived['Pclass'] , dead['Pclass']])
plt.legend(['Survived' , 'DEad'])
plt.show()
```


![png](.//mdimage/output_14_0.png)



```python
plt.hist([survived['Pclass'], dead['Pclass']] , density = True)
plt.legend(['Survived' , 'DEad'])
plt.show()
```


![png](.//mdimage/output_15_0.png)



```python
df = pd.DataFrame({'SorD':All_train['Survived'],'PClass':All_train['Pclass']}) 
ct = pd.crosstab(df.SorD, df.PClass) 

ct.plot.bar(stacked=True) 
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fc7ec301ba8>




![png](.//mdimage/output_16_1.png)



```python
plt.hist([survived['Sex'] , dead['Sex']] )
plt.legend(['Survived' , 'Dead'])
plt.show()

```


![png](.//mdimage/output_17_0.png)



```python
plt.hist([survived['Fare'] , dead['Fare']] )
plt.legend(['Survived' , 'Dead'])
plt.show()
```


![png](.//mdimage/output_18_0.png)



```python
plt.hist([survived['Age'] , dead['Age']])
plt.legend(['Survived' , 'Dead'])
plt.show()
```


![png](.//mdimage/output_19_0.png)



```python
plt.hist([survived['Parch'] , dead['Parch']]  , density = True)
plt.legend(['Survived' , 'Dead'])
plt.show()
```


![png](.//mdimage/output_20_0.png)



```python
df = pd.DataFrame({'SorD':All_train['Survived'],'Embarked':All_train['Embarked']}) 
ct = pd.crosstab(df.SorD, df.Embarked) 

ct.plot.bar(stacked=True) 
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fc7e7fcc550>




![png](.//mdimage/output_21_1.png)



```python
# 테스트셋 X , y 분리
import copy

y_train = data.iloc[:,1]
X_train = copy.deepcopy(data)
X_train.drop("Survived" , axis= 1 , inplace=True)
```

# 2. 결측치 처리


## 2.1 결측치 확인


```python
print("> Train nan count")
print(X_train.isna().sum())
print()
print("> Test nan count")
print(X_test.isna().sum())
```

    > Train nan count
    PassengerId      0
    Pclass           0
    Name             0
    Sex              0
    Age            177
    SibSp            0
    Parch            0
    Ticket           0
    Fare             0
    Cabin          687
    Embarked         2
    dtype: int64
    
    > Test nan count
    PassengerId      0
    Pclass           0
    Name             0
    Sex              0
    Age             86
    SibSp            0
    Parch            0
    Ticket           0
    Fare             1
    Cabin          327
    Embarked         0
    dtype: int64


## 2.2 결측치 처리

### 2.2.1 Fare : 평균값으로 대체

*  값이 0이면 결측치라고 생각하고 처리
*  Pclass의 Fare 평균값으로 대체


```python
X_train.loc[X_train['Fare'] == 0]
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
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>179</th>
      <td>180</td>
      <td>3</td>
      <td>Leonard, Mr. Lionel</td>
      <td>male</td>
      <td>36.0</td>
      <td>0</td>
      <td>0</td>
      <td>LINE</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>263</th>
      <td>264</td>
      <td>1</td>
      <td>Harrison, Mr. William</td>
      <td>male</td>
      <td>40.0</td>
      <td>0</td>
      <td>0</td>
      <td>112059</td>
      <td>0.0</td>
      <td>B94</td>
      <td>S</td>
    </tr>
    <tr>
      <th>271</th>
      <td>272</td>
      <td>3</td>
      <td>Tornquist, Mr. William Henry</td>
      <td>male</td>
      <td>25.0</td>
      <td>0</td>
      <td>0</td>
      <td>LINE</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>277</th>
      <td>278</td>
      <td>2</td>
      <td>Parkes, Mr. Francis "Frank"</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>239853</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>302</th>
      <td>303</td>
      <td>3</td>
      <td>Johnson, Mr. William Cahoone Jr</td>
      <td>male</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>LINE</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>413</th>
      <td>414</td>
      <td>2</td>
      <td>Cunningham, Mr. Alfred Fleming</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>239853</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>466</th>
      <td>467</td>
      <td>2</td>
      <td>Campbell, Mr. William</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>239853</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>481</th>
      <td>482</td>
      <td>2</td>
      <td>Frost, Mr. Anthony Wood "Archie"</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>239854</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>597</th>
      <td>598</td>
      <td>3</td>
      <td>Johnson, Mr. Alfred</td>
      <td>male</td>
      <td>49.0</td>
      <td>0</td>
      <td>0</td>
      <td>LINE</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>633</th>
      <td>634</td>
      <td>1</td>
      <td>Parr, Mr. William Henry Marsh</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>112052</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>674</th>
      <td>675</td>
      <td>2</td>
      <td>Watson, Mr. Ennis Hastings</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>239856</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>732</th>
      <td>733</td>
      <td>2</td>
      <td>Knight, Mr. Robert J</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>239855</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>806</th>
      <td>807</td>
      <td>1</td>
      <td>Andrews, Mr. Thomas Jr</td>
      <td>male</td>
      <td>39.0</td>
      <td>0</td>
      <td>0</td>
      <td>112050</td>
      <td>0.0</td>
      <td>A36</td>
      <td>S</td>
    </tr>
    <tr>
      <th>815</th>
      <td>816</td>
      <td>1</td>
      <td>Fry, Mr. Richard</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>112058</td>
      <td>0.0</td>
      <td>B102</td>
      <td>S</td>
    </tr>
    <tr>
      <th>822</th>
      <td>823</td>
      <td>1</td>
      <td>Reuchlin, Jonkheer. John George</td>
      <td>male</td>
      <td>38.0</td>
      <td>0</td>
      <td>0</td>
      <td>19972</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_test.loc[X_test['Fare'] == 0]
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
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>266</th>
      <td>1158</td>
      <td>1</td>
      <td>Chisholm, Mr. Roderick Robert Crispin</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>112051</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>372</th>
      <td>1264</td>
      <td>1</td>
      <td>Ismay, Mr. Joseph Bruce</td>
      <td>male</td>
      <td>49.0</td>
      <td>0</td>
      <td>0</td>
      <td>112058</td>
      <td>0.0</td>
      <td>B52 B54 B56</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 0값을 nan값으로 만들고 값을 넣는다
X_train.loc[X_train['Fare'] == 0 , 'Fare'] = float('nan')

means = X_train.groupby('Pclass').Fare.mean()
X_train = X_train.set_index(['Pclass'])
X_train['Fare'] = X_train['Fare'].fillna(means)
X_train = X_train.reset_index()

print(means)

X_test.loc[X_test['Fare'] == 0 , 'Fare'] = float('nan')

X_test = X_test.set_index(['Pclass'])
X_test['Fare'] = X_test['Fare'].fillna(means)
X_test = X_test.reset_index()
```

    Pclass
    1    86.148874
    2    21.358661
    3    13.787875
    Name: Fare, dtype: float64


### 2.2.2 Cabin : 값의 유무로 대체

원래 데이터에 cabin이 있었다면 1, 없었다면 0


```python
# X_train.drop("Cabin" , axis = 1 , inplace=True)
# X_test.drop("Cabin" , axis = 1 , inplace=True)
X_train['Has_Cabin'] = X_train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
X_test['Has_Cabin'] = X_test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

X_train.drop("Cabin" , axis = 1 , inplace=True)
X_test.drop("Cabin" , axis = 1 , inplace=True)
```

### 2.2.3 Age : 평균값으로 대체

호칭의 평균 나이로 대체

* Mr. 성인 남성
* Mrs. 결혼을 한 여성
* Ms. 결혼 여부를 밝히고 싶지 않을 때 사용하는 여성 호칭
* Miss 결혼을 하지 않은 여성
* Master (Mstr.) 결혼을 하지 않은 남성. 주로 청소년 이하


```python
X_train.head()
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
      <th>Pclass</th>
      <th>PassengerId</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>Has_Cabin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>1</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>4</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>S</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>5</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>S</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 이름을 Mr, Mss 들 로만
f = lambda x : x.split(",")[1].split(".")[0].strip()
X_train['Name'] = X_train['Name'].apply(f)
X_test['Name'] = X_test['Name'].apply(f)

```


```python
print("> 호칭 별 사람 수")
X_train['Name'].value_counts()
```

    > 호칭 별 사람 수





    Mr              517
    Miss            182
    Mrs             125
    Master           40
    Dr                7
    Rev               6
    Col               2
    Mlle              2
    Major             2
    Mme               1
    the Countess      1
    Don               1
    Sir               1
    Lady              1
    Ms                1
    Capt              1
    Jonkheer          1
    Name: Name, dtype: int64




```python
print("> 이상치 확인")
X_train.loc[(X_train['Sex']=='male') & (X_train['Name'] =="Mrs")]
X_train.loc[(X_train['Sex']=='male') & (X_train['Name'] =="Miss")]
X_train.loc[(X_train['Sex']=='female') & (X_train['Name'] =="Mr")]
X_train.loc[(X_train['Sex']=='female') & (X_train['Name'] =="Master")]
```

    > 이상치 확인





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
      <th>Pclass</th>
      <th>PassengerId</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>Has_Cabin</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
# Age의 결측치들을 이름의 평균 나이로 넣어준다
means = X_train.groupby('Name').Age.mean()
print("> 호칭 별 나이 평균")
print(means)

X_train = X_train.set_index(['Name'])
X_train['Age'] = X_train['Age'].fillna(means)
X_train = X_train.reset_index()
#위의 코드는 Name을 기준으로 Age값의 평균을 구한다음 그 평균값들을 맞는 Name에 Nan값에 넣어주는 코드


X_test = X_test.set_index(['Name'])
X_test['Age'] = X_test['Age'].fillna(means)
X_test = X_test.reset_index()
```

    > 호칭 별 나이 평균
    Name
    Capt            70.000000
    Col             58.000000
    Don             40.000000
    Dr              42.000000
    Jonkheer        38.000000
    Lady            48.000000
    Major           48.500000
    Master           4.574167
    Miss            21.773973
    Mlle            24.000000
    Mme             24.000000
    Mr              32.368090
    Mrs             35.898148
    Ms              28.000000
    Rev             43.166667
    Sir             49.000000
    the Countess    33.000000
    Name: Age, dtype: float64



```python
X_test.head()
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
      <th>Name</th>
      <th>Pclass</th>
      <th>PassengerId</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>Has_Cabin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mr</td>
      <td>3</td>
      <td>892</td>
      <td>male</td>
      <td>34.5</td>
      <td>0</td>
      <td>0</td>
      <td>330911</td>
      <td>7.8292</td>
      <td>Q</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Mrs</td>
      <td>3</td>
      <td>893</td>
      <td>female</td>
      <td>47.0</td>
      <td>1</td>
      <td>0</td>
      <td>363272</td>
      <td>7.0000</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mr</td>
      <td>2</td>
      <td>894</td>
      <td>male</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>240276</td>
      <td>9.6875</td>
      <td>Q</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Mr</td>
      <td>3</td>
      <td>895</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>315154</td>
      <td>8.6625</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Mrs</td>
      <td>3</td>
      <td>896</td>
      <td>female</td>
      <td>22.0</td>
      <td>1</td>
      <td>1</td>
      <td>3101298</td>
      <td>12.2875</td>
      <td>S</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
print("> 평균값으로 모두 채운 것 확인\n")
X_train['Age'].fillna(0)
X_test['Age'].fillna(0)
print(X_train.loc[X_train['Age'] == 0])
print(X_train.loc[X_train['Age'] == 0])
```

    > 평균값으로 모두 채운 것 확인
    
    Empty DataFrame
    Columns: [Name, Pclass, PassengerId, Sex, Age, SibSp, Parch, Ticket, Fare, Embarked, Has_Cabin]
    Index: []
    Empty DataFrame
    Columns: [Name, Pclass, PassengerId, Sex, Age, SibSp, Parch, Ticket, Fare, Embarked, Has_Cabin]
    Index: []


### 2.2.4 Embarked : 최빈값으로 대체


```python
# Embarked에는 결측치가 2개 뿐이라서 최빈값으로 결측치를 채웠다.

X_train['Embarked'].fillna(X_train['Embarked'].mode()[0] , inplace=True)
```


```python
print(X_train.isna().sum())
X_train.head()
```

    Name           0
    Pclass         0
    PassengerId    0
    Sex            0
    Age            0
    SibSp          0
    Parch          0
    Ticket         0
    Fare           0
    Embarked       0
    Has_Cabin      0
    dtype: int64





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
      <th>Name</th>
      <th>Pclass</th>
      <th>PassengerId</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>Has_Cabin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mr</td>
      <td>3</td>
      <td>1</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Mrs</td>
      <td>1</td>
      <td>2</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Miss</td>
      <td>3</td>
      <td>3</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Mrs</td>
      <td>1</td>
      <td>4</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>S</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Mr</td>
      <td>3</td>
      <td>5</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>S</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(12, 9))
sns.heatmap(data.corr(), annot=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fc7e7fb0b38>




![png](.//mdimage/output_44_1.png)


# 3. 생사와 무관해 보이는 데이터 삭제 : PassengerId , Name , Ticket


```python
# 그래프도 그렇고 상관관계도 그렇고 PassengerId는 빼도 될거같다는 느낌이
X_train.drop("PassengerId" , axis = 1 , inplace=True)
X_test.drop("PassengerId" , axis = 1 , inplace=True)

# 이름도 Sex로인해 성별 구별하니 지워도 될듯
X_train.drop("Name" , axis = 1 , inplace=True)
X_test.drop("Name" , axis = 1 , inplace=True)


X_train.drop("Ticket" , axis= 1 , inplace=True)
X_test.drop("Ticket" , axis= 1 , inplace=True)
```

StandardScaler 나 MinMaxScaler가 이상치가 있는 경우 균형 잡힌 척도를 보장할 수 없다.
이상치 처리를 안했기 때문에 스케일링 작업을 하지 않았다.

# 4. Categorical variables 를 Numeric Variables 로 변환

## 4.1 Sex : 


```python
X_train['Sex'] = X_train['Sex'].map( {'male':1, 'female':0} )
X_test['Sex'] = X_test['Sex'].map( {'male':1, 'female':0} )

```

## 4.2 Embarked : one hot encoding


```python
# X_train
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)
X_train.head()

#a decision tree can be learned directly from categorical data with no data transform required (this depends on the specific implementation).
# 그럼 랜덤포레스트에서도 안바꿔줘도 상관없는건가?
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
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Has_Cabin</th>
      <th>Embarked_C</th>
      <th>Embarked_Q</th>
      <th>Embarked_S</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>1</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>1</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



# 5. 모델링

## 5.1 데이터셋 준비

* train set을 7 : 3 으로 나눠서 validiation set을 생성


```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

trnx, valx, trny, valy = train_test_split(X_train, y_train, test_size=0.3, random_state=510)
```

## 5.2 모델 선택

| <h2><center>모델</center></h2> | <h2><center>장점</center></h2> | <h2><center>단점</center></h2> | 
|-------------------|:------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------:| 
| **랜덤 포레스트** | 이상치에 강건(robust)하다 | 텍스트 데이터 같은 매우 차원이 높고 희소한 데이터에는 잘 작동하지 않는다 | 
| "  | 하이퍼 파라미터 조정 없이 기본 설정으로도 좋은 결과를 만들어줄 때가 많다 | 많은 트리를 만들수록 많은 메모리와 긴 훈련 시간을 요구한다 | 
| " | 데이터의 스케일을 맞출 필요가 없다 |  | 
| **그래디언트 부스팅** | 이상치에 강건(robust)하다 | 적절한 하이퍼 파라미터를 찾기 위한 튜닝시간이 길다 | 
| " | 하이퍼 파라미터를 잘 조정한다면 높은 정확도를 낸다 | 랜덤 포레스트보다 오버피팅의 위험이 더 높다 | 
| " | 데이터의 스케일을 맞출 필요가 없다 | |   
| **서포트 벡터 머신** | 데이터 특성이 몇개 되지 않더라도 복잡한 결정 경계를 만들 수 있다. | 데이터 전처리와 매개변수 설정에 따라 정확도가 다르기 때문에 신경을 많이 써야한다 (scailing) |

### 5.2.1 GradientBoosting


```python

param = {
    "loss":["deviance"],
    "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
    "min_samples_split": np.linspace(0.1, 0.5, 12),
    "min_samples_leaf": np.linspace(0.1, 0.5, 12),
    "max_depth":[3,5,8],
    "max_features":["log2","sqrt"],
    "criterion": ["friedman_mse",  "mae"],
    "subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
    "n_estimators":[10]
    }

CV_rfc = GridSearchCV(estimator=GradientBoostingClassifier(random_state = 42), param_grid=param, cv= 5)
CV_rfc.fit(trnx,trny)
CV_rfc.best_params_ , CV_rfc.best_score_
```


```python
forest = GradientBoostingClassifier(loss = "deviance" , learning_rate = 0.2 , criterion = 'friedman_mse' , max_depth = 3 , max_features = 'log2' , min_samples_leaf = 0.1 , min_samples_split = 0.31818181818181823 , n_estimators = 10 , subsample = 0.95)
forest.fit(trnx,trny)

print("train : {} ".format(forest.score(trnx,trny)))

print("validation : {} ".format(forest.score(valx,valy)))
```

    train : 0.7913322632423756 
    validation : 0.7761194029850746 


### 5.2.2 RandomForest


```python

param= {
          'n_estimators': [300, 400, 500, ],
          'max_features': ['auto', 'sqrt', 'log2'],
          'max_depth': [3,4,5,6,8 ]
 }

CV_rfc = GridSearchCV(estimator=RandomForestClassifier(random_state = 42), param_grid=param, cv= 5)
CV_rfc.fit(trnx,trny)
CV_rfc.best_params_ , CV_rfc.best_score_
```




    ({'max_depth': 8, 'max_features': 'auto', 'n_estimators': 500},
     0.8282504012841091)




```python

forest = RandomForestClassifier(n_estimators = 500, random_state = 42 , max_depth=8, max_features='auto')
forest.fit(trnx,trny)

print("train : {} ".format(forest.score(trnx,trny)))

print("validation : {} ".format(forest.score(valx,valy)))

```

    train : 0.9229534510433387 
    validation : 0.7910447761194029 


## 5.3 최종 모델 적용


```python
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators = 400, random_state = 42 , max_depth=8, max_features='auto')
forest.fit(X_train,y_train)
```




    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                           max_depth=8, max_features='auto', max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=400,
                           n_jobs=None, oob_score=False, random_state=42, verbose=0,
                           warm_start=False)




```python
result = forest.predict(X= X_test)

submission = pd.DataFrame({"PassengerId":test_PassengerId  , "Survived" : result})


submission[:10]
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
      <th>PassengerId</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>897</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>898</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>899</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>900</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>901</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
submission.to_csv("./data/result.csv" ,index=False)
```

# 5.4 특성 중요도


```python

print("특성 중요도 : \n{}".format(forest.feature_importances_))

feature_importances = pd.DataFrame(forest.feature_importances_,
                                   index = X_train.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)

feature_importances


# 랜덤포레스트의 장점 중 하나는 각 독립 변수의 중요도(feature importance)를 계산할 수 있다는 점이다.
# 포레스트 안에서 사용된 모든 노드에 대해 어떤 독립 변수를 사용하였고 그 노드에서 얻은 information gain을 구할 수 있으므로 
# 각각의 독립 변수들이 얻어낸 information gain의 평균을 비교하면 어떤 독립 변수가 중요한지를 비교할 수 있다.

```

    특성 중요도 : 
    [0.09971134 0.35675923 0.17315173 0.05117964 0.03891019 0.17154221
     0.06420081 0.01543297 0.00983961 0.01927227]





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
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Sex</th>
      <td>0.356759</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>0.173152</td>
    </tr>
    <tr>
      <th>Fare</th>
      <td>0.171542</td>
    </tr>
    <tr>
      <th>Pclass</th>
      <td>0.099711</td>
    </tr>
    <tr>
      <th>Has_Cabin</th>
      <td>0.064201</td>
    </tr>
    <tr>
      <th>SibSp</th>
      <td>0.051180</td>
    </tr>
    <tr>
      <th>Parch</th>
      <td>0.038910</td>
    </tr>
    <tr>
      <th>Embarked_S</th>
      <td>0.019272</td>
    </tr>
    <tr>
      <th>Embarked_C</th>
      <td>0.015433</td>
    </tr>
    <tr>
      <th>Embarked_Q</th>
      <td>0.009840</td>
    </tr>
  </tbody>
</table>
</div>




```python
data = pd.read_csv('./data/train.csv')

print("> Sex와 Age의 중요성 검증\n")

print("Pclass 1에 탄 20대 여성")
print(data.loc[(data['Sex'] == 'female') & (data['Pclass'] == 1 ) & (data['Age'] >=20) & ( data['Age'] < 30) ]['Survived'].value_counts())

print("Pclass 1에 탄 여성")
print(data.loc[(data['Sex'] == 'female') & (data['Pclass'] == 1 ) ]['Survived'].value_counts())


```

    > Sex와 Age의 중요성 검증
    
    Pclass 1에 탄 20대 여성
    1    15
    0     1
    Name: Survived, dtype: int64
    Pclass 1에 탄 여성
    1    91
    0     3
    Name: Survived, dtype: int64


# 6. 개선사항

## 6.1 Cabin

Cabin의 결측치를 잘 예측해서 넣으면 성능이 좋아질 것 같다.


```python
data = pd.read_csv('./data/train.csv')
X_test = pd.read_csv('./data/test.csv')

y_train = data.iloc[:,1]
X_train = copy.deepcopy(data)
```


```python

f = lambda x : str(x)[0]
X_train['Cabin'] = X_train.loc[pd.notnull(X_train['Cabin']) ,'Cabin'].apply(f)
X_test['Cabin'] = X_test['Cabin'].apply(f)


X_train.loc[pd.notnull(X_train['Cabin'])].head(10)

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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C</td>
      <td>C</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C</td>
      <td>S</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>McCarthy, Mr. Timothy J</td>
      <td>male</td>
      <td>54.0</td>
      <td>0</td>
      <td>0</td>
      <td>17463</td>
      <td>51.8625</td>
      <td>E</td>
      <td>S</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>1</td>
      <td>3</td>
      <td>Sandstrom, Miss. Marguerite Rut</td>
      <td>female</td>
      <td>4.0</td>
      <td>1</td>
      <td>1</td>
      <td>PP 9549</td>
      <td>16.7000</td>
      <td>G</td>
      <td>S</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>1</td>
      <td>1</td>
      <td>Bonnell, Miss. Elizabeth</td>
      <td>female</td>
      <td>58.0</td>
      <td>0</td>
      <td>0</td>
      <td>113783</td>
      <td>26.5500</td>
      <td>C</td>
      <td>S</td>
    </tr>
    <tr>
      <th>21</th>
      <td>22</td>
      <td>1</td>
      <td>2</td>
      <td>Beesley, Mr. Lawrence</td>
      <td>male</td>
      <td>34.0</td>
      <td>0</td>
      <td>0</td>
      <td>248698</td>
      <td>13.0000</td>
      <td>D</td>
      <td>S</td>
    </tr>
    <tr>
      <th>23</th>
      <td>24</td>
      <td>1</td>
      <td>1</td>
      <td>Sloper, Mr. William Thompson</td>
      <td>male</td>
      <td>28.0</td>
      <td>0</td>
      <td>0</td>
      <td>113788</td>
      <td>35.5000</td>
      <td>A</td>
      <td>S</td>
    </tr>
    <tr>
      <th>27</th>
      <td>28</td>
      <td>0</td>
      <td>1</td>
      <td>Fortune, Mr. Charles Alexander</td>
      <td>male</td>
      <td>19.0</td>
      <td>3</td>
      <td>2</td>
      <td>19950</td>
      <td>263.0000</td>
      <td>C</td>
      <td>S</td>
    </tr>
    <tr>
      <th>31</th>
      <td>32</td>
      <td>1</td>
      <td>1</td>
      <td>Spencer, Mrs. William Augustus (Marie Eugenie)</td>
      <td>female</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17569</td>
      <td>146.5208</td>
      <td>B</td>
      <td>C</td>
    </tr>
    <tr>
      <th>52</th>
      <td>53</td>
      <td>1</td>
      <td>1</td>
      <td>Harper, Mrs. Henry Sleeper (Myna Haxtun)</td>
      <td>female</td>
      <td>49.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17572</td>
      <td>76.7292</td>
      <td>D</td>
      <td>C</td>
    </tr>
  </tbody>
</table>
</div>




```python
C = X_train.loc[X_train['Cabin']=='C']
D = X_train.loc[X_train['Cabin']=='D']
E = X_train.loc[X_train['Cabin']=='E']
F = X_train.loc[X_train['Cabin']=='F']
G = X_train.loc[X_train['Cabin']=='G']
A = X_train.loc[X_train['Cabin']=='A']
B = X_train.loc[X_train['Cabin']=='B']
T = X_train.loc[X_train['Cabin']=='T']

```


```python
# plt.hist([A['Pclass'] , B['Pclass'] , C['Pclass'] , D['Pclass'] , E['Pclass'] , F['Pclass'] , G['Pclass'] , T['Pclass']] , density = True)
plt.figure(figsize=(12, 9))
plt.hist([A['Survived'] , B['Survived'] , C['Survived'] , D['Survived'] , E['Survived'] , F['Survived'] , G['Survived'] , T['Survived']] , density = True)

plt.legend(['A' , 'B', 'C', 'D', 'E', 'F', 'G', 'T'])
plt.show()


```


![png](.//mdimage/output_76_0.png)



```python
plt.hist([A['Pclass'] , B['Pclass'] , C['Pclass'] , D['Pclass'] , E['Pclass'] , F['Pclass'] , G['Pclass'] , T['Pclass']] , density = True)

plt.legend(['A' , 'B', 'C', 'D', 'E', 'F', 'G', 'T'])
plt.show()
```


![png](.//mdimage/output_77_0.png)



```python
plt.hist([A['Fare'] , B['Fare'] , C['Fare'] , D['Fare'] , E['Fare'] , F['Fare'] , G['Fare'] , T['Fare']] , density = True)

plt.legend(['A' , 'B', 'C', 'D', 'E', 'F', 'G', 'T'])
plt.show()
```


![png](.//mdimage/output_78_0.png)




## 6.2 Outliers

이상치 처리를 좀 더 신중하게 하면 성능이 좋아질 것 같다.


```python
Male_train = X_train.loc[X_train['Sex']=='male']
Female_train = X_train.loc[X_train['Sex']=='female']

Male_train.head()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>0</td>
      <td>3</td>
      <td>Moran, Mr. James</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>330877</td>
      <td>8.4583</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>McCarthy, Mr. Timothy J</td>
      <td>male</td>
      <td>54.0</td>
      <td>0</td>
      <td>0</td>
      <td>17463</td>
      <td>51.8625</td>
      <td>E</td>
      <td>S</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>0</td>
      <td>3</td>
      <td>Palsson, Master. Gosta Leonard</td>
      <td>male</td>
      <td>2.0</td>
      <td>3</td>
      <td>1</td>
      <td>349909</td>
      <td>21.0750</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>


