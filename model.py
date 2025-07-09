import torch
import torch.nn as nn
from torch.nn import functional as F



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class SelfAttention(nn.Module):
    def __init__(self, in_dims=2, d_model=64, num_heads=4):
        super(SelfAttention, self).__init__()
        self.embedding = nn.Linear(in_dims, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)

        self.edge_query = nn.Linear(d_model//num_heads, d_model//num_heads)
        self.edge_key = nn.Linear(d_model//num_heads, d_model//num_heads)

        self.C_edge_query = nn.Linear(d_model//num_heads, d_model//num_heads)
        self.C_edge_key = nn.Linear(d_model//num_heads, d_model//num_heads)

        self.scaled_factor = torch.sqrt(torch.Tensor([d_model])).cuda()
        self.softmax = nn.Softmax(dim=-1)
        self.num_heads = num_heads
    def split_heads(self, x):
        # x [batch_size seq_len d_model]
        x = x.reshape(x.shape[0], -1, self.num_heads, x.shape[-1] // self.num_heads).contiguous()
        return x.permute(0, 2, 1, 3)  # [batch_size nun_heads seq_len depth]
    def forward(self, x, edge_inital, C_edge_inital, G, C_G):
        # batch_size seq_len 2
        assert len(x.shape) == 3
        embeddings = self.embedding(x)  # batch_size seq_len d_model
        query = self.query(embeddings)  # batch_size seq_len d_model
        key = self.key(embeddings)      # batch_size seq_len d_model
        query = self.split_heads(query)  # B num_heads seq_len d_model
        key = self.split_heads(key)  # B num_heads seq_len d_model

        edge_query = self.edge_query(edge_inital)  # batch_size 4 seq_len d_model
        edge_key = self.edge_key(edge_inital)      # batch_size 4 seq_len d_model
        div = torch.sum(G, dim=1)[:, None, :, None]
        Gquery = query + torch.einsum('bmn,bhnc->bhmc', G.transpose(-2, -1), edge_query) / div  # q [batch, num_agent, heads, 64/heads]
        Gkey = key + torch.einsum('bmn,bhnc->bhmc', G.transpose(-2, -1), edge_key) / div
        g_attention = torch.matmul(Gquery, Gkey.permute(0, 1, 3, 2))  # (batch_size, num_heads, seq_len, seq_len)
        g_attention = self.softmax(g_attention / self.scaled_factor)

        C_edge_query = self.C_edge_query(C_edge_inital)  # batch_size 4 seq_len d_model
        C_edge_key = self.C_edge_key(C_edge_inital)      # batch_size 4 seq_len d_model
        C_div = torch.sum(C_G, dim=1)[:, None, :, None]
        CGquery = query + torch.einsum('bmn,bhnc->bhmc', C_G.transpose(-2, -1), C_edge_query) / C_div  # q [batch, num_agent, heads, 64/heads]
        CGkey = key + torch.einsum('bmn,bhnc->bhmc', C_G.transpose(-2, -1), C_edge_key) / C_div
        Cg_attention = torch.matmul(CGquery, CGkey.permute(0, 1, 3, 2))  # (batch_size, num_heads, seq_len, seq_len)
        Cg_attention = self.softmax(Cg_attention / self.scaled_factor)

        return g_attention, Cg_attention



class Edge_inital(nn.Module):
    def __init__(self, in_dims=2, d_model=64):
        super(Edge_inital, self).__init__()
        self.x_embedding = nn.Linear(in_dims, d_model//4)
        self.edge_embedding = nn.Linear(d_model//4, d_model//4)
    def forward(self, x, G):
        assert len(x.shape) == 3
        embeddings = self.x_embedding(x)  # batch_size seq_len d_model
        div = torch.sum(G, dim=-1)[:, :, None]
        edge_init = self.edge_embedding(torch.matmul(G, embeddings) / div)  # T N d_model
        edge_init = edge_init.unsqueeze(1).repeat(1, 4, 1, 1)
        return edge_init



class AsymmetricConvolution(nn.Module):
    def __init__(self, in_cha, out_cha):
        super(AsymmetricConvolution, self).__init__()
        self.conv1 = nn.Conv2d(in_cha, out_cha, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.conv2 = nn.Conv2d(in_cha, out_cha, kernel_size=(1, 3), padding=(0, 1))
        self.shortcut = lambda x: x
        if in_cha != out_cha:
            self.shortcut = nn.Sequential(nn.Conv2d(in_cha, out_cha, 1, bias=False))
        self.activation = nn.PReLU()
    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.activation(self.conv2(x) + self.conv1(x))
        return x + shortcut



class DConvolution(nn.Module):
    def __init__(self, in_cha, out_cha):
        super(DConvolution, self).__init__()
        self.asymmetric_convolutions = nn.ModuleList()
        for i in range(4):
            self.asymmetric_convolutions.append(AsymmetricConvolution(in_cha, out_cha))
        self.asymmetric_convolutions.append(AsymmetricConvolution(in_cha*5, out_cha))
    def forward(self, x):
        x0 = self.asymmetric_convolutions[0](x)
        x1 = self.asymmetric_convolutions[1](x0)
        x2 = self.asymmetric_convolutions[2](x1)
        x3 = self.asymmetric_convolutions[3](x2)
        x4 = self.asymmetric_convolutions[4](torch.cat([x, x0, x1, x2, x3], dim=1))
        return x4



class S_Branch(nn.Module):
    def __init__(self):
        super(S_Branch, self).__init__()
        self.tcns = nn.Sequential(nn.Conv2d(8, 8, 1, padding=0),
            nn.PReLU())
        self.Dconvolutions = DConvolution(4, 4)
        self.activation = nn.Sigmoid()
    def forward(self, x):
        temporal_x = x.permute(1, 0, 2, 3)  # x (num_heads T N N)
        temporal_x = self.tcns(temporal_x) + temporal_x
        x = temporal_x.permute(1, 0, 2, 3)
        threshold = self.activation(self.Dconvolutions(x))
        x_zero = torch.zeros_like(x, device='cuda')
        Sparse_x = torch.where(x > threshold, x, x_zero)
        return Sparse_x



class T_Branch(nn.Module):
    def __init__(self):
        super(T_Branch, self).__init__()
        self.Dconvolutions = DConvolution(4, 4)
        self.activation = nn.Sigmoid()
    def forward(self, x):
        threshold = self.activation(self.Dconvolutions(x))
        x_zero = torch.zeros_like(x, device='cuda')
        Sparse_x = torch.where(x > threshold, x, x_zero)
        return Sparse_x



class BinaryThresholdFunctionType(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        # 前向传播：应用自适应二值化阈值
        ctx.save_for_backward(input, threshold)
        return (input > 0).float()  # 阈值化操作
    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播：提供近似梯度
        input, threshold = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_threshold = None  # 默认不计算阈值的梯度
        # 对输入张量应用梯度近似
        grad_input[torch.abs(input) > threshold] = 0
        return grad_input, grad_threshold  # 返回梯度



class BinaryThreshold(nn.Module):
    def __init__(self):
        super(BinaryThreshold, self).__init__()
    def forward(self, input, threshold):
        return BinaryThresholdFunctionType.apply(input, threshold)



class STAdaptiveGroupEstimator(nn.Module):
    def __init__(self, in_dims=2):
        super().__init__()
        self.ste = BinaryThreshold()
        self.multi_output = nn.Sequential(nn.Linear(in_dims, 8),
            nn.PReLU(),
            nn.Linear(8, 16))
        self.th = nn.Parameter(torch.Tensor([0.7]))
    def forward(self, node_features):
        # node_features = (T N 2)
        node_features = self.multi_output(node_features)  # node_features (T N 16)
        temp = F.normalize(node_features, p=2, dim=2)  # temp [batch, num_agent, 64]
        corr_mat = torch.matmul(temp, temp.permute(0, 2, 1))  # corr_mat [batch, num_agent, num_agent]
        G = self.ste((corr_mat - self.th.clamp(-0.9999, 0.9999)), self.th.clamp(-0.9999, 0.9999))  # G [batch, num_agent, num_agent]
        return G



class SparseWeightedAdjacency(nn.Module):
    def __init__(self, s_in_dims=2, t_in_dims=3, embedding_dims=64, dropout=0,):
        super(SparseWeightedAdjacency, self).__init__()
        # AdaptiveGroupEstimator
        self.S_Group = STAdaptiveGroupEstimator(in_dims=2)
        # edge_inital
        self.S_edge_inital = Edge_inital(s_in_dims, embedding_dims)
        self.C_S_edge_inital = Edge_inital(s_in_dims, embedding_dims)
        # dense interaction
        self.S_group_attention = SelfAttention(s_in_dims, embedding_dims)

        self.g_S_branch = S_Branch()
        self.C_g_S_branch = S_Branch()
    def forward(self, graph, identity):
        assert len(graph.shape) == 3
        spatial_graph = graph[:, :, 1:]  # (T N 2)

        S_G = self.S_Group(spatial_graph)  # (T N N)

        #### 用于保证值不为零 #####
        spatial_scaling = torch.ones_like(S_G) * 1e-6
        C_S_G = 1 - S_G + spatial_scaling

        S_E = self.S_edge_inital(spatial_graph, S_G)  # (T 4 N 16)
        C_S_E = self.C_S_edge_inital(spatial_graph, C_S_G)  # (T 4 N 16)

        G_S, C_G_S = self.S_group_attention(spatial_graph, S_E, C_S_E, S_G, C_S_G)  # (T num_heads N N)

        G_S = self.g_S_branch(G_S)  # (T num_heads N N)
        C_G_S = self.C_g_S_branch(C_G_S)  # (T num_heads N N)

        G_S = G_S + identity[0].unsqueeze(1)
        C_G_S = C_G_S + identity[0].unsqueeze(1)
        return G_S, C_G_S, S_G, C_S_G, S_E, C_S_E



class GraphConvolution(nn.Module):
    def __init__(self, in_dims=2, embedding_dims=16, dropout=0):
        super(GraphConvolution, self).__init__()
        self.edge_value = nn.Linear(embedding_dims, in_dims)
        self.C_edge_value = nn.Linear(embedding_dims, in_dims)

        self.g_embedding = nn.Linear(in_dims, embedding_dims, bias=False)
        self.Cg_embedding = nn.Linear(in_dims, embedding_dims, bias=False)

        self.activation = nn.PReLU()
        self.dropout = dropout
    def forward(self, graph, g_adjacency, Cg_adjacency, G, C_G, edge_inital, C_edge_inital):
        # graph=[T, 1, N, 2](seq_len 1 num_p 2)
        div = torch.sum(G, dim=1)[:, None, :, None]
        edge = self.edge_value(edge_inital)
        value = graph + torch.einsum('bmn,bhnc->bhmc', G.transpose(-2, -1), edge) / div
        g_gcn_features = self.g_embedding(torch.matmul(g_adjacency, value))

        C_div = torch.sum(C_G, dim=1)[:, None, :, None]
        C_edge = self.C_edge_value(C_edge_inital)
        C_value = graph + torch.einsum('bmn,bhnc->bhmc', C_G.transpose(-2, -1), C_edge) / C_div
        Cg_gcn_features = self.Cg_embedding(torch.matmul(Cg_adjacency, C_value))

        gcn_features = F.dropout(self.activation(g_gcn_features + Cg_gcn_features), p=self.dropout)
        return gcn_features  # [batch_size num_heads seq_len hidden_size]



class SparseGraphConvolution(nn.Module):
    def __init__(self, in_dims=16, embedding_dims=16, dropout=0):
        super(SparseGraphConvolution, self).__init__()
        self.dropout = dropout

        self.spatial_gcn = GraphConvolution(in_dims, embedding_dims)
        self.temporal_gcn = GraphConvolution(in_dims, embedding_dims)
    def forward(self, graph, G_S, C_G_S, S_G, C_S_G, S_E, C_S_E):
        # graph [1 seq_len num_pedestrians  3]
        graph = graph[:, :, :, 1:]
        spa_graph = graph.permute(1, 0, 2, 3)  # (seq_len 1 num_p 2)

        spatial_features = self.spatial_gcn(spa_graph, G_S, C_G_S, S_G, C_S_G, S_E, C_S_E)
        spatial_features = spatial_features.permute(2, 0, 1, 3)  # spatial_features [N, T, heads, 16]

        return spatial_features  # [N, T, heads, 16]



class TrajectoryModel(nn.Module):
    def __init__(self,embedding_dims=64, number_gcn_layers=1, dropout=0,obs_len=8, pred_len=12, n_tcn=5, num_heads=4):
        super(TrajectoryModel, self).__init__()
        self.number_gcn_layers = number_gcn_layers
        self.n_tcn = n_tcn
        self.dropout = dropout
        # sparse graph learning
        self.sparse_weighted_adjacency_matrices = SparseWeightedAdjacency()

        # graph convolution
        self.stsgcn = SparseGraphConvolution(in_dims=2, embedding_dims=embedding_dims // num_heads, dropout=dropout)

        self.tcns = nn.ModuleList()
        self.tcns.append(nn.Sequential(nn.Conv2d(obs_len, pred_len, 3, padding=1),
            nn.PReLU()))
        for j in range(1, self.n_tcn):
            self.tcns.append(nn.Sequential(nn.Conv2d(pred_len, pred_len, 3, padding=1),
                nn.PReLU()))

        self.output = nn.Linear(embedding_dims // num_heads, 2)
        self.multi_output = nn.Sequential(nn.Conv2d(num_heads, 16, 1, padding=0),
            nn.PReLU(),
            nn.Conv2d(16, 20, 1, padding=0),)
    def forward(self, graph, identity):
        # graph 1 obs_len N 3   # obs_traj 1 obs_len N 2
        G_S, C_G_S, S_G, C_S_G, S_E, C_S_E \
            = self.sparse_weighted_adjacency_matrices(graph.squeeze(), identity)

        # gcn_representation = [N, T, heads, 16]
        gcn_representation = self.stsgcn(graph, G_S, C_G_S, S_G, C_S_G, S_E, C_S_E)

        features = self.tcns[0](gcn_representation)
        for k in range(1, self.n_tcn):
            features = F.dropout(self.tcns[k](features) + features, p=self.dropout)

        prediction = self.output(features)   # prediction=[N, Tpred, nums, 2]
        prediction = self.multi_output(prediction.permute(0, 2, 1, 3))   # prediction=[N, 20, Tpred, 2]

        return prediction.permute(1, 2, 0, 3).contiguous()



