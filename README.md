- # CG Project F 大作业

  [TOC]

  ## 安装

  ### 安装需求

  - 在安装之前请确认使用`python 3.5`或者`python 3.6`的版本运行。**同时安装`tensorflow 2.0.0`和`tensorflow-gpu 2.0.0`**。本代码并不支持`tensforlow 1.x`的版本，请注意。如果没有安装，可以使用`pip`进行安装

    ```shell
    pip install numpy
    pip install tensorflow==2.0.0
    pip install tensorflow-gpu==2.0.0
    ```

  - 如果有在GPU上运行的需求，请遵循Tensorflow的需求安装CUDA Runtime以及CUDNN。具体可以参照[网址](https://www.tensorflow.org/install/gpu)钟的Software requirements需求进行安装

  - 一些legacy的API当中会有其他额外的需求，例如在`legacy/pointfly.py`当中有`import transforms3d`，为了保证顺利运行，请安装：

    ```shell
    pip install transofrms3d
    pip install matplotlib
    ```

  ### 数据集安装

  数据集请从网上下载并解压，并**放置在和源代码根目录同级别的目录下**，结构大致如下：

  ```
  ./Data ---> 数据集
  ./CG-Final-Project-F ---> 源代码
  ./Models ---> 模型存放的位置
  ```

  目前的制作好的数据集只有ModelNet，同学们可以自行把数据集做好然后放进Data当中。

  ### 测试

  可以通过下面代码测试模型是否能够正常的运转

  ```shell
  python3 CG-Final-Project-F/run.py conf/modelnet40/point_cnn.pyconf
  ```

  若不能正常运转，请检查：

  - Python版本是否正确
  - Tensorflow以及版本是否为2.0.0
  - （如果使用GPU）CUDA和CUDNN是否安装在正确的位置上，CUDA Runtime是否被正确加入PATH或者LD_LIBRARY_PATH的环境变量中，并且版本是否正确
  - 数据集的位置是否放置正确

  若还是不能运转，请告知我

  

  ## 大作业要求

  在本次的大作业中，你需要寻找一篇有关Pointcloud Classification以及Segmentation的一篇文章，例如[Dynamic graph cnn for learning on point clouds](https://arxiv.org/abs/1801.07829), [Mento Carlo Point cloud classificaiton](https://arxiv.org/abs/1806.01759)。**并在本框架对文章提出的算法进行实习，同时对比和测试算法和原代码以及论文的准确率比较**。

  ### 实现难点

  注意，在复现的过程当中，**你需要做的并不是copy and paste**的操作，而是在理解论文的基础上**参考原来的代码在本文框架上进行实现**。总体来说，你可能会遇到下面这些困难：

  - Tensorflow的版本不兼容的问题，例如有些代码使用Tensorflow 1.x进行开发的，而我们的代码是基于Tensorflow 2.0.0。由于Tensforflow从1.x到2.0的过程中会有很多op不兼容的问题，这个需要自己解决。
  - 原本的代码都是使用`tf.Session`并手写计算图的方式实现的。而我们是基于Tensorflow Keras Layer，这个需要自己转换。
  - 原本的代码当中会包含一些数据的处理，你需要把