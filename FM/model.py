import torch
import torch.nn as nn


# 模型构建
class FM_Layer(nn.Module):
    def __init__(self, feature_nums=10, k=5):
        """
        feature_nums: 特征个数
        k: 特征对应的隐向量维度大小
        """
        super(FM_Layer, self).__init__()
        self.feature_nums = feature_nums
        self.k = k
        self.linear = nn.Linear(self.feature_nums, 1)   # 前两项线性层(w和b)
        self.V = nn.Parameter(torch.randn(self.feature_nums, self.k))   # 交互矩阵
        nn.init.uniform_(self.V, -0.1, 0.1)

    def fm_layer(self, x):
        linear_part = self.linear(x)
        interaction_part_1 = torch.mm(x, self.V)
        interaction_part_1 = torch.pow(interaction_part_1, 2)
        interaction_part_2 = torch.mm(torch.pow(x, 2), torch.pow(self.V, 2))
        part = torch.sum(interaction_part_2 - interaction_part_1, 1, keepdim=False)
        output = linear_part.reshape(-1,) + 0.5 * part
        return output
    
    def forward(self, x):
        x = self.fm_layer(x)
        return x


class FM(nn.Module):
    """
    这里进行分类任务, 因此加了个Sigmoid, 用以表示输出概率, 训练中损失函数使用BCELoss(二分交叉熵损失函数)
    如果是进行回归任务, 如预测评分, 则不加Sigmoid, 直接输出, 损失函数可采用MSELoss
    """
    def __init__(self, feature_nums, k):
        super().__init__()
        self.fm = nn.Sequential(
            FM_Layer(feature_nums, k),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fm(x)
        return x