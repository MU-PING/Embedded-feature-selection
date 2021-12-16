# embedded-feature-selection
## 程式簡介
### 簡述
* 使用 sklearn 實作 Feature Selection 中的 **Embedded methods【嵌入法】**

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

* 結合  Wrapper methods 與 Filter methods 的優點

* 概念圖  
  ![image](https://user-images.githubusercontent.com/93152909/146382076-9f165415-c872-4665-bab5-4649ba62b721.png)

* 優點

  * 與 Wrapper methods 優點相同，能偵測變數之間的相互影響
  
  * 與 Filter methods 優點相同，執行速度快，但相對正確率會比較高

  * 與 Filter methods 相比，特徵選擇的正確率較高
  
  * 有效的解決 Overfitting

* 目前主流的作法為 「L1 、L2 Regularization」，以分為下兩種演算法：

  * Lasso: Linear Regression with L1 Regularization
  
  * Ridge: Linear Regression with L2 Regularization
### Lasso: Linear Regression with L1 Regularization
* 全名：least absolute shrinkage and selection operator，又譯最小絕對值收斂和選擇算子、套索算法
* 𝐿𝑜𝑠𝑠′ 𝑓𝑢𝑛𝑐𝑡𝑖𝑜𝑛
  * 𝐿𝑜𝑠𝑠′ 𝑓𝑢𝑛𝑐𝑡𝑖𝑜𝑛： 表示加了 L1 的損失函數
  * 𝐿𝑜𝑠𝑠 𝑓𝑢𝑛𝑐𝑡𝑖𝑜𝑛： 表示原本的損失函數
  * 𝑅𝑒𝑔𝑢𝑙𝑎𝑟𝑖𝑧𝑎𝑡𝑖𝑜𝑛 𝑇𝑒𝑟𝑚： L1  
  * 𝑥：feature 
  * 𝑦：regression label
  * 𝑊：feature weights
  * 𝑁：numbers of data
  * 𝑀：numbers of feature
  * 𝜆：L1的有效性    
    <img src="https://user-images.githubusercontent.com/93152909/146438380-07f07a50-9c0c-4cc0-8e82-c532d5289886.png" width="650">

### Ridge: Linear Regression with L2 Regularization

* 𝐿𝑜𝑠𝑠′ 𝑓𝑢𝑛𝑐𝑡𝑖𝑜𝑛
  * 𝐿𝑜𝑠𝑠′ 𝑓𝑢𝑛𝑐𝑡𝑖𝑜𝑛： 表示加了 L2 的損失函數
  * 𝐿𝑜𝑠𝑠 𝑓𝑢𝑛𝑐𝑡𝑖𝑜𝑛： 表示原本的損失函數
  * 𝑅𝑒𝑔𝑢𝑙𝑎𝑟𝑖𝑧𝑎𝑡𝑖𝑜𝑛 𝑇𝑒𝑟𝑚： L2  
  * 𝑥：feature 
  * 𝑦：regression label
  * 𝑊：feature weights
  * 𝑁：numbers of data
  * 𝑀：numbers of feature
  * 𝜆：L2的有效性   
    <img src="https://user-images.githubusercontent.com/93152909/146438433-e849b573-6302-4fe9-bfc0-efbf7c8ce027.png" width="650">


## 參考
https://ithelp.ithome.com.tw/articles/10246876  
https://ithelp.ithome.com.tw/articles/10227654  
https://dasanlin888.pixnet.net/blog/post/476250317-%E6%AD%A3%E8%A6%8F%E5%8C%96%E8%BF%B4%E6%AD%B8(regularized-regression)  
https://allen108108.github.io/blog/2019/10/22/L1%20,%20L2%20Regularization%20%E5%88%B0%E5%BA%95%E6%AD%A3%E5%89%87%E5%8C%96%E4%BA%86%E4%BB%80%E9%BA%BC%20_/  
https://medium.com/ai%E5%8F%8D%E6%96%97%E5%9F%8E/learning-model-l1-l2-regularization%E5%B7%AE%E7%95%B0-8d7fc089b35c  
https://www.cnblogs.com/zingp/p/10375691.html
> 備份於 Reference 資料夾中
