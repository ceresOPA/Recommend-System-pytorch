# Recommend-System-pytorch（经典推荐系统算法的Pytorch实现）
The code of recommendation system algorithms(models) implemented by Pytorch. (Updating)

> 本仓库代码主要参考自jc_Lee大佬的[Recommend-System-tf2.0](https://github.com/jc-LeeHub/Recommend-System-tf2.0)，由于本人对tensorflow并不熟悉，而且也更习惯于使用torch，因此就有了写一套pytorch版代码的想法，也是加深自己对推荐系统算法理解的过程。关于算法理论的解读建议看《深度学习推荐系统》这本书和jc_Lee大佬的知乎文章，感觉都是很不错的。

## Models List（算法列表）

|  Model | Paper |                                                                                                                                 
| :----: | :------- | 
|  [FM](https://github.com/ceresOPA/Recommend-System-pytorch/tree/main/FM) | [ICDM 2010] [Fast Context-aware Recommendationswith Factorization Machines](https://www.ismll.uni-hildesheim.de/pub/pdfs/Rendle_et_al2011-Context_Aware.pdf)  
|  [FFM](https://github.com/ceresOPA/Recommend-System-pytorch/tree/main/FFM) | [RecSys 2016] [Field-aware Factorization Machines for CTR Prediction](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf)           |
|  [DeepFM](https://github.com/ceresOPA/Recommend-System-pytorch/tree/main/DeepFM)   | [IJCAI 2017] [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](http://www.ijcai.org/proceedings/2017/0239.pdf)                      |
|  [xDeepFM](https://github.com/ceresOPA/Recommend-System-pytorch/tree/main/xDeepFM) | [KDD 2018] [xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems](https://arxiv.org/pdf/1803.05170.pdf)                     |

## Reference（参考资料）
  - [Recommend-System-tf2.0](https://github.com/jc-LeeHub/Recommend-System-tf2.0)
  - [《深度学习推荐系统》](https://book.douban.com/subject/35013197/)
  - http://shomy.top/2018/12/31/factorization-machine/