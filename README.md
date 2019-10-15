- # CG Project F 大作业

  

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

首先，**你需要把CG-Final-Project-F加入到PYTHONPATH的环境变量当中**，可以通过下面代码测试模型是否能够正常的运转

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
- 原本的代码当中会包含一些数据的处理，你需要把这部分给去除并且适配到我们的pipeline上去

### 步骤

在目前的Demo代码中包含了一个[PointCNN](https://arxiv.org/pdf/1801.07791.pdf)的网络架构。你可以在`conf/point_cnn.pyconf`以及`layers/xconv`中找到相应的代码。本次大作业需要做的事情会和这个类似，你需要：

- 将论文中模型架构变成Tensorflow的Keras Layer（参考：`layers/xconv.py`）
- 准备数据集，并为特定的数据集写读入的函数（参考：`utils/datasetutil.py`）
- 为你的模型准备一个Configuration File（参考：`conf/point_cnn.pyconf`）
- 测试你的代码run得起来：`python3 CG-Final-Project-F/run.py conf/${dataset_name}/${your_custom_config}`

具体的步骤可以参考`conf/point_cnn.pyconf`以及`layres/xconv`，**建议通过亲自run一下代码以及打断点的方式看一下模型是如何运行起来的**



## 合作

本项目需要**每个小组在[网址](https://github.com/vividgithub/CG-Final-Project-F)上**建立一个自己的**小组分支**，**你的Commit以及分支上的代码会作为最后的考核标准**。同时，我也会维护`master`分支，`master`分支会不定期的更新README以及push一些公用的Utility。如果同学有很好的想法或者可以复用的东西也希望能够发pull request给我，我会合并到master分支上。

鉴于这需要使用到一些`git`的基础知识，如果你对这部分不熟悉的话，你可以参考git官网上的[官方教程](https://git-scm.com/book/en/v2)。在这里，我简要的把一些基础需要用到的知识列出。

- **【重要】在使用Git之前，请把你的Github账号告知我，我会将你加入本项目的合作者，这样你可以自由的`push`代码**。

- **如无必要，请必要尝试更改master分支；如果你有很好的idea或者可以复用的Utility想要和大家分享，请发Pull request给我**。

- 你也可以在小组的分支外建立个人的分支，个人分支可以选择不push到remote端，但是小组的分支需要push上来作为考核的标准
- 建议**隔三差五的push一遍，好让助教follow你们的进度状况，不要等到学期末在一口气push上来**

### 常用Git命令

- 将代码clone下来使用

  ```
  git clone https://github.com/vividgithub/CG-Final-Project-F
  ```

- 在git上建立自己的分支（这里以F1小组为例）：

  ```
  cd CG-Final-Project-F
  git checkout -b F1
  ```

- 当你写了一些代码，需要把代码Commit。你可以通过`git status`

  ```
  cd CG-Final-Project-F
  git add .
  git commit -m "Implement a greate function" 
  ```

- 把自己的代码push到远程上：

  ```
  cd CG-Final-Project-F
  git push orgin F1
  ```

- 当发现master更新以后，想要合并master分支：

  ```
  cd CG-Final-Project-F
  git fetch origin
  git merge master
  ```

## FAQ

#### 如何把一个从一个config生成自己自定义的Layer或者函数

一般来说，你需要使用到`utils/confutil`里面的`register_conf`以及`object_from_conf`函数，`register_conf`是一个函数/类装饰器，用来告诉python`这个类可以由config生成`。具体可以参见下面的代码： 

```python
from utils.confutil import register_conf, object_from_conf

@register_conf(name="layer-myawesomelayer", scope="layer", conf_func="self")
class MyAwesomeLayer(tf.keras.layer.Layer):
  # 注意我们一般要使用**kwargs来避免额外的参数
  def __init__(self, key, value1, value2, value2, **kwargs):
    ...
    
conf = {
  "name": "layer-myawesomelayer",
  "key": "a key",
  "value1": "a value",
  "value2": "another value"
}

my_awesome_layer = object_from_conf(conf, scope="layer")
print(type(my_awesome_layer)) # MyAwesomeLayer
```

更详细的用法可以参考`PointCNN`其中的实现代码