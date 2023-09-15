import torch
import torch.nn as nn
import torch.nn.functional as F

# 模型构建
class FFM_Layer(nn.Module):
    def __init__(self, feature_columns, k=5):
        """
        feature_columns: 特征列
        k: 特征对应的隐向量维度大小
        """
        super(FFM_Layer, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.k = k
        self.feature_nums = sum([feat['feat_onehot_dim'] for feat in self.sparse_feature_columns]) \
                           + len(self.dense_feature_columns)
        self.field_nums = len(self.dense_feature_columns) + len(self.sparse_feature_columns)

        self.linear = nn.Linear(self.feature_nums, 1)   # 前两项线性层(w和b)
        self.V = nn.Parameter(torch.randn(self.feature_nums, self.field_nums, self.k))   # 交互矩阵，在FM的基础上增加了field域
        nn.init.uniform_(self.V, -0.1, 0.1)


    def ffm_layer(self, x):
        dense_inputs, sparse_inputs = x[:, :13], x[:, 13:].to(dtype=torch.int64)
        sparse_x = torch.concat([ F.one_hot(sparse_inputs[:, i], num_classes=self.sparse_feature_columns[i]['feat_onehot_dim'])
                                  for i in range(sparse_inputs.shape[1])], axis=1)
        x = torch.concat([dense_inputs, sparse_x], axis=-1)

        linear_part = self.linear(x)

        inter_part = 0
        # 每维特征先与对应的 [field_num, k] 相乘得到Vij*X
        field_f = torch.matmul(x, self.V.reshape(self.V.size(0), -1)).reshape(-1, self.V.size(1), self.V.size(2))
        # 域之间两两相乘，
        for i in range(self.field_nums):
            for j in range(i+1, self.field_nums):
                inter_part += torch.sum(
                    torch.multiply(field_f[:, i], field_f[:, j]), # [None, 8]
                    dim=1, keepdim=True
                )

        output = linear_part + inter_part

        return output
    
    def forward(self, x):
        x = self.ffm_layer(x)
        return x

class FFM(nn.Module):
    def __init__(self, feature_columns, k):
        super(FFM, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.ffm = nn.Sequential(
            FFM_Layer(feature_columns, k),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        output = self.ffm(inputs).reshape(-1, )
        return output