# dml
## Distance Metric Learning Algorithm

Implementation of MPCK-MEANS algorithm (**M**etric Learning with **P**airwise **C**onstraints **K-Means**)

The original article: *Bilenko, Mikhail, Sugato Basu, and Raymond J. Mooney.* **"Integrating constraints and metric learning in semi-supervised clustering."** *Proceedings of the twenty-first international conference on Machine learning. ACM, 2004.*
([pdf](http://research.microsoft.com/en-us/um/people/mbilenko/papers/04-semi-icml.pdf))

By integrating this algorithm to an existed interactive semi-supervised image clustering, we can improve [the result](https://minhvvu.github.io/demo/stage1/index.html).
Here is the abstract of out work - *Viet Minh Vu, Hien Phuong Lai, Muriel Visani* **Towards an approach using metric learning for interactive semi-supervised clustering of images.**  *KSE 2016: 357-362* ([pdf](https://minhvvu.github.io/demo/article/KSE_paper89_approach-metric-learning_2016.pdf)):

> The problem of unsupervised and semi-supervised clustering is extensively studied in machine learning. In order to involve user in image data clustering, we proposed a new approach for interactive semi-supervised clustering that translates user feedback (expressed at the level of individual images) into pairwise constraints between groups of images, these groups being formed thanks to the underlying hierarchical clustering solution and user feedback. Recently, the need for appropriate measures of distance or similarity between data led to the emergence of distance metric learning approaches. In this paper, we propose a method incorporating metric learning in the existing system to improve performance and reduce the computational time. Our preliminary experiments performed on the Wang dataset show that metric learning methods improve the performances and computational time of the existing system.

*Dependences:*

1. CMake

2. C++11

3. [Eigen Library:](http://eigen.tuxfamily.org/index.php?title=Main_Page)


*System Design:*
![system diagram](https://github.com/minhvvu/dml/blob/master/systemDesign.png "Class Diagram")


*How to extend?*

Inheritate the existed classes and add your code to into virtual functions which are called automatically.

