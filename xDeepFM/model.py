import torch
import torch.nn as nn

class CIN(nn.Module):
    """
    CIN这部分按照原tf代码的思路和自己的理解实现了下, 可能存在一定的偏差 
    然后有部分地方是写死的, 后面还要再改改
    """
    def __init__(self, cin_size=[26, 128, 128]):
        super(CIN, self).__init__()
        self.cin_size = cin_size  # 每层的矩阵个数
        self.cin_W = nn.ModuleList(
            [nn.Conv3d(in_channels=1, out_channels=cin_size[i+1], kernel_size=(cin_size[i], cin_size[0], 1), stride=1) for i in range(len(cin_size)-1)])

    def forward(self, inputs):
        k = inputs.shape[-1]
        res_list = [inputs]
        X0 = torch.split(inputs, 1, dim=-2)
        for i, size in enumerate(self.cin_size[1:]):
            Xi = torch.split(res_list[-1], 1, dim=-2)
            x = []
            for i_i in range(len(Xi)):
                tmp = []
                for j_i in range(len(X0)):
                    tmp.append(Xi[i_i]*X0[j_i])
                tmp = torch.hstack(tmp).unsqueeze(1)
                # print(tmp.shape)
                x.append(tmp)
                
            x = torch.concatenate(x, dim=1).unsqueeze(1)
            x =  self.cin_W[i](x) # 用卷积实现三维矩阵的压缩
            x = x.reshape(-1, size, x.size(-1))
            res_list.append(x)

        res_list = res_list[1:]   # 去掉 X0
        res = torch.concatenate(res_list, dim=1)     # (Batch, field_num[1]+...+field_num[n], k)
        output = torch.sum(res, dim=-1)  # (Batch, field_num[1]+...+field_num[n])
        return output


class xDeepFM(nn.Module):
    def __init__(self, feature_columns, feature_nums, cin_size):
        super(xDeepFM, self).__init__()

        self.feature_nums = feature_nums

        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.embed_layers = [nn.Embedding(feat['feat_onehot_dim'], feat['embed_dim'])
                                    for feat in self.sparse_feature_columns]
        
        self.linear =  nn.Linear(self.feature_nums, 1) # 一阶项
        
        self.dense_layer = nn.Sequential(
            nn.Linear(221, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU()
        )

        self.cin_layer = nn.Sequential(
            CIN(cin_size),
            nn.Linear(256, 1),
            nn.ReLU()
        )

        self.out_layer = nn.Sequential(
            nn.Sigmoid()
        )

    def forward(self, inputs):
        dense_inputs, sparse_inputs = inputs[:, :13], inputs[:, 13:].int()

        # linear
        linear_out = self.linear(inputs)

        emb = [self.embed_layers[i](sparse_inputs[:, i]).unsqueeze(1) for i in range(sparse_inputs.shape[1])] # [n, field, k]
        emb = torch.concatenate(emb, dim=1) # 与DeepFM和DeepCross不同，采用纵向拼接，而不是横向拼接

        # CIN
        cin_out = self.cin_layer(emb)

        # dense
        emb = emb.reshape(-1, emb.shape[1]*emb.shape[2])
        emb = torch.concat([dense_inputs, emb], axis=1)
        dense_out = self.dense_layer(emb)

        output = self.out_layer(linear_out + cin_out + dense_out).reshape(-1, )
        return output