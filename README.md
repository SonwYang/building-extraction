# building-extraction-based-on-deep-learning
# 前言

语义分割在基于高分影像的建筑提取中不能区分不同的建筑个体，因为在建筑密集区其通常将同一个类的对象紧密地打包成一个连接的组件，即使是优秀的语义分割网络模型也不可避免地会产生这样的结果。而实例分割可以较好地解决这个问题，精准到建筑个体的分割结果在绘制地图、城市规划以及人口估计中将会大有可为。

![image-20200514161921651](https://i.loli.net/2020/05/14/9fhBbeiRd6NTagC.png)

# 要解决的问题

1. 目前，在基于深度学习技术对高分遥感影像进行建筑提取的研究中，通常是将其归类为二分类的任务，即将所有的像素视为建筑和非建筑这两类，而对网络模型区分建筑物个体的能力关注较少；
2. 在建筑密集区进行遥感影像分类后，易出现“房屋粘连”现象，即预测后的边界通常不够友好，而且同一个类的对象被紧密地打包成一个连接的组件，因此需要研究相应的影像后处理方法来优化分割结果。

# 技术路线

![image-20200514162328700](https://i.loli.net/2020/05/14/WoJa7CBqY49GRzf.png)

# HA U-Net模型结构

![image-20200514162902586](https://i.loli.net/2020/05/14/blQ7y1hTmxXzG6P.png)

# 基于权重映射的影像分类流程

![image-20200514163432391](https://i.loli.net/2020/05/14/wnv86TGWkPCKigD.png)

# 基于分水岭算法的影像后处理

对所获得的概率分布图进行双阈值操作，分别获取内部标记和外部标记，其中高阈值对应内部标记，低阈值对应外部标记，然后用分水岭算法进行处理（对于无房屋粘连区域不进行处理），保留分水线，最后将分水线叠加到预测结果上。

![image-20200514163531289](https://i.loli.net/2020/05/14/ZxluaiLJ7nCTqGw.png)

# 提取效果概览

![image-20200514162815984](https://i.loli.net/2020/05/14/yBCZMfUs2OA4xmQ.png)

# 相关指标

| Methods\Metrics                    | IOU   | Kappa | 实例化F1 |
| ---------------------------------- | ----- | ----- | -------- |
| U-Net                              | 73.19 | 82.16 | 85.76    |
| HA U-Net+IWM                       | 75.32 | 83.71 | 89.36    |
| HA U-Net+IWM+ Watershed (0.5, 0.9) | 75.28 | 83.69 | 89.90    |

# Reference

[基于分水岭算法的影像后处理](https://spark-in.me/post/playing-with-dwt-and-ds-bowl-2018)

[weight mapping](https://arxiv.org/pdf/1802.07465.pdf)

[attention U-Net](https://arxiv.org/pdf/1804.03999.pdf)

[HNN](https://www.semanticscholar.org/paper/A-Holistically-Nested-U-Net%3A-Surgical-Instrument-on-Yu-Wang/52b9f2f06a15cc67422fe03f6af7541b3dc717f8)

[实例化F1](https://tianchi.aliyun.com/competition/entrance/231767/information)

# PS
如果你不熟悉深度学习在提取建筑物的使用，你可以尝试运行下这个kaggle上的[notebook](https://www.kaggle.com/yangpeng1995/building-extraction-in-deep-learning/data)。
