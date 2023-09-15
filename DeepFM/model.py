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
    

class DeepFM(nn.Module):
    def __init__(self, feature_columns, feature_nums, k):
        super(DeepFM, self).__init__()

        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.embed_layers = {
            'embed_' + str(i): nn.Embedding(feat['feat_onehot_dim'], feat['embed_dim'])
             for i, feat in enumerate(self.sparse_feature_columns)
        }

        self.fm = nn.Sequential(
            FM_Layer(feature_nums, k),
            nn.Sigmoid()
        )
        self.fc = nn.Sequential(
            nn.Linear(feature_nums, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, dense_inputs, sparse_inputs):
        # embedding，就是和词嵌入是一样的
        sparse_embed = torch.concat([self.embed_layers[f'embed_{i}'](sparse_inputs[:, i])
                                  for i in range(sparse_inputs.shape[1])], axis=1)
        
        x = torch.concat([dense_inputs, sparse_embed], axis=-1)

        fm_output = self.fm(x)
        fc_output = self.fc(x).reshape(-1,)

        output = 0.5*(fm_output + fc_output)

        return output