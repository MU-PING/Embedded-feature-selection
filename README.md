# Embedded-feature-selection
## 程式簡介
### 簡述
* 使用 Sklearn  套件實作 Feature Selection 中的 **Embedded methods【嵌入法】**

* Lasso.ipynb 以「Lasso」實作 Embedded methods

* Ridge.ipynb 以「Ridge」實作 Embedded methods

* train.csv 為 Boston 房價資料集，皆以其作為 Embedded methods 的範例資料集

|Attribute|Description|  
| -------- | -------- | 
|CRIM|per capita crime rate by town|
|ZN|proportion of residential land zoned for lots over 25,000 sq.ft|
|INDUS|proportion of non-retail business acres per town|
|CHAS|Charles River dummy variable (1 if tract bounds river; 0 otherwise)|
|NOX|nitric oxides concentration (parts per 10 million)|
|RM|average number of rooms per dwelling|
|AGE|proportion of owner-occupied units built prior to 1940|
|DIS|weighted distances to five Boston employment centres|
|RAD|index of accessibility to radial highways|
|TAX|full-value property-tax rate per $10,000|
|PTRATIO|pupil-teacher ratio by town|
|B|1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town|
|LSTAT|% lower status of the population|
|【**Label**】MEDV|Median value of owner-occupied homes in $1000's|
  
## Feature Selection
* 又稱為 variable selection、attribution selection 或 subset selection

* 指從資料集中選出最重要、最相關的特徵來給機器學習建立模型，大部分時候，這樣做可以增加機器學習的效能

* Feature selection 不等於 Dimensionality Reduction

### WHY
機器學習的實際應用中，特徵數量往往較多，可能存在不相關的特徵，特徵之間也可能存在相互依賴，容易導致：

* 特徵個數越多，分析特徵、訓練模型所需的時間也就越長。

* 特徵個數越多，容易引起【維度災難】
#### 維度災難
* 特徵維度超過一定界限後，分類器的效能隨著特徵維度的增加反而下降，如下圖
  ![image](https://user-images.githubusercontent.com/93152909/145701552-148a6354-f79c-4310-b047-619353903b76.png)
  > 原因往往是因為這些高維度特徵中含有「無關特徵」和「冗餘特徵」

* 無關特徵  
該特徵所提供的資訊對於當前學習任務無用，如對於「學生成績」而言，「學號」則是無關特徵。
* 冗餘特徵  
該特徵所包含的資訊能從其他特徵推演出來，如「面積」特徵，能從「長」和「寬」得出，則它是冗餘特徵。
### HOW
一般 Feature selection 的演算法分為三類：

* Filter methods

* Wrapper methods

* **Embedding methods【此篇介紹】**

## Embedded methods【嵌入法】
* 嵌入法是指在機器學習模型訓練的同時，執行特徵選擇

* 概念圖  
  ![image](https://user-images.githubusercontent.com/93152909/146382076-9f165415-c872-4665-bab5-4649ba62b721.png)

* 優點 - 結合  Wrapper methods 與 Filter methods 的優點

  * 與 Wrapper methods 優點相同，能偵測變數之間的相互影響
  
  * 與 Filter methods 優點相同，執行速度快，但相對正確率會比較高

  * 與 Filter methods 相比，特徵選擇的正確率較高
  
  * 有效的解決 Overfitting

* 目前主流的作法為 「L1 、L2 Regularization」，以分為下兩種演算法：

  * Lasso: Linear Regression with L1 Regularization
  
  * Ridge: Linear Regression with L2 Regularization
### Lasso: Linear Regression with L1 Regularization

```python
 # Lasso with cross-validation
 from sklearn.linear_model import LassoCV
```

* 全名：least absolute shrinkage and selection operator，又譯最小絕對值收斂和選擇算子、套索算法
* 𝐿𝑜𝑠𝑠′ 𝑓𝑢𝑛𝑐𝑡𝑖𝑜𝑛 - sklearn
  * 【**𝐿𝑜𝑠𝑠′**】： 加了 L1 的損失函數
  * 【**𝐿𝑜𝑠𝑠 𝑓𝑢𝑛𝑐𝑡𝑖𝑜𝑛**】： 原本的損失函數
  * 【**𝑟𝑒𝑔𝑢𝑙𝑎𝑟𝑖𝑧𝑎𝑡𝑖𝑜𝑛 𝑡𝑒𝑟𝑚**】： L1    
  
    <img src="https://user-images.githubusercontent.com/93152909/146800930-c302a9cf-d91d-4a8e-be8f-8479073d719a.png" width="500">
    
      * 𝑋：feature 
      * 𝑦：regression label
      * 𝑤：feature weights
      * 𝑚：numbers of data
      * 𝑝：numbers of feature
      * 𝑎𝑙𝑝ℎ𝑎：L1的有效性
### Ridge: Linear Regression with L2 Regularization

```python
 # Ridge with cross-validation
 from sklearn.linear_model import RidgeCV
```

* 𝐿𝑜𝑠𝑠′ 𝑓𝑢𝑛𝑐𝑡𝑖𝑜𝑛 - sklearn
  * 【**𝐿𝑜𝑠𝑠′**】： 加了 L2 的損失函數
  * 【**𝐿𝑜𝑠𝑠 𝑓𝑢𝑛𝑐𝑡𝑖𝑜𝑛**】： 原本的損失函數
  * 【**𝑟𝑒𝑔𝑢𝑙𝑎𝑟𝑖𝑧𝑎𝑡𝑖𝑜𝑛 𝑡𝑒𝑟𝑚**】： L2  
  
    <img src="https://user-images.githubusercontent.com/93152909/146800936-0630302e-e521-4d4e-b939-5399df703acc.png" width="500">
    
      * 𝑋：feature 
      * 𝑦：regression label
      * 𝑤：feature weights
      * 𝑚：numbers of data
      * 𝑝：numbers of feature
      * 𝑎𝑙𝑝ℎ𝑎：L2的有效性  
      
### L1 && L2 regularization
* 目的都是在 Loss function 中加入適當的【懲罰項】，讓模型不會過度收斂

* L1 & L2都能避免模型的參數同時出現極大正係數與極大負係數的情況

#### L1 regularization
* L1 會將不具影響力的變數之迴歸係數變成0，等於可以自動化的進行變數篩選(Feature selection)

* 變數篩選的同時可能也會犧牲掉模型的正確性，等同用「正確性」換「泛化性」

#### L2 regularization

* L2 會將不具影響力的變數之迴歸係數**逼近**為0(不會剛好等於0)，可以藉此降低資料集中的雜訊

* L2 會保留所有變數，模型可能還是會存在一些不重要的參數，多多少少影響模型的正確性

* L2 無法做到自動化的變數篩選，只能盡量將低不重要的變數之影響力

#### 如何選擇
實務上要選擇L1 或 L2， 就模型計算出來的 Loss 來看，L1 和 L2 所產生的最小 Loss 不會有太大差別，所以單純只看最小化 Loss 的結果來判斷要用哪種，其實結果是差不多的。不過就功能性來說，**當使用者的模型中具有過多的參數，想自動化把不重要的變數給移除，那應該要選擇 L1**；**如果想找到模型當中重要的參數可以透過 L2 來去辨別哪些參數是重要的，因為不重要參數的迴歸係數會趨近於0，但不會真的消失，所以可以根據迴歸係數的大小來得到重要參數的排名**
