# Encrypted Image Retrieval 加密图像检索

#### 介绍

该库侧重于原始开源软件的设计和开发，描述了实验中的技术等，存储了代码，并将进行更新和维护。随着云计算的发展，越来越多的资源受限的数据所有者倾向于将他们的图像存储在云中。考虑到安全和隐私，图片在上传前应该加密。因此，我们设计了一种高效的保护隐私的图像检索方案。我们的方案首先采用迁移学习技术，使用预训练的深度学习模型提取图像特征，以提高检索精度。然后使用K近邻算法来加密这些特征，在不损害数据隐私的情况下提高检索效率。我们的项目主要提供四种相互独立的功能：（1）图像加密；（2）图像特征提取；（3）图像特征加密；（4）图像检索。

#### 软件架构
软件架构如图所示：![输入图片说明](frame.png)


#### 安装教程

本软件使用前的准备步骤：下载安装python，需要打开 PyCharm 的官方网站 https://www.jetbrains.com/pycharm/download/并根据电脑操作系统（Windows、macOS 或者 Linux）选择对应的版本安装python。
之后需要安装anaconda+CUDA+cuDNN+PyTorch。
下载数据集corel-1k：下载地址https://download.csdn.net/download/weixin_46323807/90546450?spm= 1001.2014.3001.5501。 

#### 使用说明

1. 本软件相关代码于https://gitee.com/liangchenjuezizi/original-open-source-software中下载。基于AES的图像加密算法于image_en_AES _files.py执行，基于混沌的图像加密算法于image_en_chaos _files.py执行，只需把目标文件夹换为本地文件夹即可。
2. CLD图像特征提取于CLD_corel-1k.py中执行，EHD图像特征提取于EHD_corel-1k.py中执行，VGG16提取图像特征于子目录deep_learning的VGG16_corel-1k.py中执行。子目录deep_learning中的其余py文件用于训练卷积神经网络，避免过拟合和和权重衰退。
3. 图像特征加密于子目录user下的KNN_owner.py和KNN_search_user.py中执行，其中KNN_owner.py是图像所有者对整个图像数据的特征向量进行加密，KNN_ search_user.py是查询用户根据图像所有者分发的密钥对查询图像特征进行加密。
4. 图像检索于子目录user下的Server_GUI.py文件中执行，需要先拿到所有的加密图像特征和查询图像特征再进行点积操作，即可得到检索结构。
5. 本软件隐私保护图像检索功能能够被第三方程序引用或可编译为可执行软件，具备用户交互功能。使用者进入Server_GUI.py所在命令行或者终端下，执行python Server_GUI.py --query_count 5 --want_top 3语句并输入查询图像编号即可，其中，数字5和3可根据实际情况进行更改，分别表示查询相似图像的次数和返回相似图像的个数。
6. 子目录image_eample存储原始图像以及使用基于AES和基于混沌的图像加密算法的结果数据示例。

#### 项目声明 Project Statement
本项目的名称、作者及单位:

项目名称：Encrypted Image Retrieval 加密图像检索
项目作者：Liang Jing
作者单位：暨南大学网络空间安全学院


