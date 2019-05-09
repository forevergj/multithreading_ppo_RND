# multithreading_ppo_RND
  运用的强化学习算法是多线程PPO算法，可选择是否添加自适应抖动、批训练探索回报，只在gym倒立摆的游戏环境实验过，效果和原本的多线程ppo算法差不多。（pytorch）
  
  PPO算法可以选择Method，clip or kl_divergence，输入可以是图片（把n_features删掉即可），或者ram特征。
  
  RND参考文献：https://arxiv.org/abs/1810.12894
  
  程序实现的细节没有仔细参考原文。
