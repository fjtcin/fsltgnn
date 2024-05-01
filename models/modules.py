import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryLoss:
    def __init__(self):
        self.loss1 = nn.BCEWithLogitsLoss()
        self.loss2 = nn.BCELoss()

    def __call__(self, input, target):
        assert input.dim() == 2, f"Wrong dimension {input.dim()} of input!"
        if input.size(1) == 1:
            input = input.squeeze(1)
            return self.loss1(input, target)
        elif input.size(1) == 2:
            input = input.softmax(dim=1)[:, 1]
            return self.loss2(input, target)
        else:
            raise ValueError(f"Wrong shape {input.shape} of input!")


class TimeEncoder(nn.Module):

    def __init__(self, time_dim: int, parameter_requires_grad: bool = True):
        """
        Time encoder.
        :param time_dim: int, dimension of time encodings
        :param parameter_requires_grad: boolean, whether the parameter in TimeEncoder needs gradient
        """
        super(TimeEncoder, self).__init__()

        self.time_dim = time_dim
        # trainable parameters for time encoding
        self.w = nn.Linear(1, time_dim)
        self.w.weight = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim, dtype=np.float32))).reshape(time_dim, -1))
        self.w.bias = nn.Parameter(torch.zeros(time_dim))

        if not parameter_requires_grad:
            self.w.weight.requires_grad = False
            self.w.bias.requires_grad = False

    def forward(self, timestamps: torch.Tensor):
        """
        compute time encodings of time in timestamps
        :param timestamps: Tensor, shape (batch_size, seq_len)
        :return:
        """
        # Tensor, shape (batch_size, seq_len, 1)
        timestamps = timestamps.unsqueeze(dim=2)

        # Tensor, shape (batch_size, seq_len, time_dim)
        output = torch.cos(self.w(timestamps))

        return output


class TransformerTimeEncoder(nn.Module):

    def __init__(self, time_dim):
        self.time_dim = time_dim

    def positional_encoding(self, times: torch.Tensor):
        positions = torch.arange(self.time_dim, device=times.device)
        div_term = torch.pow(10000.0, -2 * positions / self.time_dim)
        encoded = times * div_term
        encoded[:, 0::2] = torch.sin(encoded[:, 0::2])  # apply sin to even indices in the tensor; 2i
        encoded[:, 1::2] = torch.cos(encoded[:, 1::2])  # apply cos to odd indices in the tensor; 2i+1
        return encoded


class MergeLayer(nn.Module):

    def __init__(self, input_dim1: int, input_dim2: int, hidden_dim: int, output_dim: int):
        """
        Merge Layer to merge two inputs via: input_dim1 + input_dim2 -> hidden_dim -> output_dim.
        :param input_dim1: int, dimension of first input
        :param input_dim2: int, dimension of the second input
        :param hidden_dim: int, hidden dimension
        :param output_dim: int, dimension of the output
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim1 + input_dim2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()

    def forward(self, input_1: torch.Tensor, input_2: torch.Tensor):
        """
        merge and project the inputs
        :param input_1: Tensor, shape (*, input_dim1)
        :param input_2: Tensor, shape (*, input_dim2)
        :return:
        """
        # Tensor, shape (*, input_dim1 + input_dim2)
        x = torch.cat([input_1, input_2], dim=1)
        # Tensor, shape (*, output_dim)
        h = self.fc2(self.act(self.fc1(x)))
        return h


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(input_dim * 2, input_dim)
        self.fc2 = nn.Linear(input_dim, output_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_1: torch.Tensor, input_2: torch.Tensor, times=None):
        x = torch.cat([input_1, input_2], dim=1)
        return self.fc2(self.dropout(self.act(self.fc1(x))))

    def prototypical_encoding(self, model):
        pass


class LinkPredictor(nn.Module):
    def __init__(self, prompt_dim, lamb, dropout):
        super().__init__()
        self.lamb = lamb
        self.prompts = nn.Parameter(torch.ones(1, prompt_dim))
        self.time_encoder = TimeEncoder(time_dim=prompt_dim)
        self.mlp = MLP(input_dim=prompt_dim, output_dim=1, dropout=dropout)

    def forward(self, input_1, input_2, times):
        src = torch.cat([input_1, input_1], dim=1)
        dst = torch.cat([input_2, input_2], dim=1)
        p = self.prompts + self.time_encoder(torch.from_numpy(times).unsqueeze(1).float().to(src.device)).squeeze(1) * self.lamb
        src = F.normalize(src * p)
        dst = F.normalize(dst * p)
        return self.mlp(src, dst)


class EdgeClassifier(nn.Module):
    def __init__(self, args, train_data, train_idx_data_loader, prompt_dim, mlp, lamb):
        super().__init__()
        self.lamb = lamb
        self.args = args
        self.train_data = train_data
        self.train_idx_data_loader = train_idx_data_loader
        self.num_classes = train_data.labels.max().item() + 1
        self.prompts = nn.Parameter(torch.ones(1, prompt_dim))
        self.time_encoder = TimeEncoder(time_dim=prompt_dim)
        self.mlp = mlp

    def forward(self, input_1: torch.Tensor, input_2: torch.Tensor, times: np.ndarray):
        features = torch.cat([input_1, input_2], dim=1)
        p = self.prompts + self.time_encoder(torch.from_numpy(times).unsqueeze(1).float().to(self.args.device)).squeeze(1) * self.lamb
        src = F.normalize(features * p)
        dst = self.prototypical_edges
        res = self.mlp(src.repeat_interleave(dst.size(0), dim=0), dst.repeat(src.size(0), 1)).reshape(src.size(0), dst.size(0))
        return res

    def prototypical_encoding(self, model):
        self.prototypical_edges = torch.zeros(self.num_classes, self.prompts.shape[1], device=self.args.device)
        # count_nodes_for_each_label = torch.zeros(self.num_classes, 1).to(self.args.device)

        for batch_idx, train_data_indices in enumerate(self.train_idx_data_loader):
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids, batch_labels = \
                    self.train_data.src_node_ids[train_data_indices], self.train_data.dst_node_ids[train_data_indices], self.train_data.node_interact_times[train_data_indices], \
                    self.train_data.edge_ids[train_data_indices], self.train_data.labels[train_data_indices]

            with torch.no_grad():
                if self.args.model_name in ['TGAT', 'CAWN', 'TCL']:
                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model.compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                            dst_node_ids=batch_dst_node_ids,
                                                                            node_interact_times=batch_node_interact_times,
                                                                            num_neighbors=self.args.num_neighbors)
                elif self.args.model_name in ['JODIE', 'DyRep', 'TGN']:
                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model.compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                            dst_node_ids=batch_dst_node_ids,
                                                                            node_interact_times=batch_node_interact_times,
                                                                            edge_ids=batch_edge_ids,
                                                                            edges_are_positive=True,
                                                                            num_neighbors=self.args.num_neighbors)
                elif self.args.model_name in ['GraphMixer']:
                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model.compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                            dst_node_ids=batch_dst_node_ids,
                                                                            node_interact_times=batch_node_interact_times,
                                                                            num_neighbors=self.args.num_neighbors,
                                                                            time_gap=self.args.time_gap)
                elif self.args.model_name in ['DyGFormer']:
                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model.compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                            dst_node_ids=batch_dst_node_ids,
                                                                            node_interact_times=batch_node_interact_times)
                else:
                    raise ValueError(f"Wrong value for model_name {self.args.model_name}!")

            batch_edge_embeddings = torch.hstack((batch_src_node_embeddings, batch_dst_node_embeddings))
            p = self.prompts + self.time_encoder(torch.from_numpy(batch_node_interact_times).unsqueeze(1).float().to(self.args.device)).squeeze(1) * self.lamb
            batch_edge_embeddings = F.normalize(batch_edge_embeddings * p)

            batch_labels = torch.from_numpy(batch_labels).to(self.args.device)
            mask = torch.zeros(self.num_classes, batch_labels.size(0), device=self.args.device)
            mask.scatter_(0, batch_labels.unsqueeze(0), 1)
            sum_features_for_each_label = mask @ batch_edge_embeddings

            # count_nodes_for_each_label += torch.sum(mask, dim=1, keepdim=True)
            self.prototypical_edges += sum_features_for_each_label

        self.prototypical_edges = F.normalize(self.prototypical_edges)


class EdgeClassifierLearnable(nn.Module):
    def __init__(self, num_classes, prompt_dim, mlp, lamb):
        super().__init__()
        self.lamb = lamb
        self.prompts = nn.Parameter(torch.ones(1, prompt_dim))
        self.time_encoder = TimeEncoder(time_dim=prompt_dim)
        self.prototypical_edges = nn.Parameter(torch.rand(num_classes, prompt_dim))
        self.mlp = mlp

    def forward(self, input_1: torch.Tensor, input_2: torch.Tensor, times: np.ndarray):
        features = torch.cat([input_1, input_2], dim=1)
        p = self.prompts + self.time_encoder(torch.from_numpy(times).unsqueeze(1).float().to(features.device)).squeeze(1) * self.lamb
        src = F.normalize(features * p)
        dst = F.normalize(self.prototypical_edges)
        res = self.mlp(src.repeat_interleave(dst.size(0), dim=0), dst.repeat(src.size(0), 1)).reshape(src.size(0), dst.size(0))
        return res

    def prototypical_encoding(self, model):
        pass


class MultiHeadAttention(nn.Module):

    def __init__(self, node_feat_dim: int, edge_feat_dim: int, time_feat_dim: int,
                 num_heads: int = 2, dropout: float = 0.1):
        """
        Multi-head Attention module.
        :param node_feat_dim: int, dimension of node features
        :param edge_feat_dim: int, dimension of edge features
        :param time_feat_dim: int, dimension of time features (time encodings)
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        """
        super(MultiHeadAttention, self).__init__()

        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.time_feat_dim = time_feat_dim
        self.num_heads = num_heads

        self.query_dim = node_feat_dim + time_feat_dim
        self.key_dim = node_feat_dim + edge_feat_dim + time_feat_dim

        assert self.query_dim % num_heads == 0, "The sum of node_feat_dim and time_feat_dim should be divided by num_heads!"

        self.head_dim = self.query_dim // num_heads

        self.query_projection = nn.Linear(self.query_dim, num_heads * self.head_dim, bias=False)
        self.key_projection = nn.Linear(self.key_dim, num_heads * self.head_dim, bias=False)
        self.value_projection = nn.Linear(self.key_dim, num_heads * self.head_dim, bias=False)

        self.scaling_factor = self.head_dim ** -0.5

        self.layer_norm = nn.LayerNorm(self.query_dim)

        self.residual_fc = nn.Linear(num_heads * self.head_dim, self.query_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, node_features: torch.Tensor, node_time_features: torch.Tensor, neighbor_node_features: torch.Tensor,
                neighbor_node_time_features: torch.Tensor, neighbor_node_edge_features: torch.Tensor, neighbor_masks: np.ndarray):
        """
        temporal attention forward process
        :param node_features: Tensor, shape (batch_size, node_feat_dim)
        :param node_time_features: Tensor, shape (batch_size, 1, time_feat_dim)
        :param neighbor_node_features: Tensor, shape (batch_size, num_neighbors, node_feat_dim)
        :param neighbor_node_time_features: Tensor, shape (batch_size, num_neighbors, time_feat_dim)
        :param neighbor_node_edge_features: Tensor, shape (batch_size, num_neighbors, edge_feat_dim)
        :param neighbor_masks: ndarray, shape (batch_size, num_neighbors), used to create mask of neighbors for nodes in the batch
        :return:
        """
        # Tensor, shape (batch_size, 1, node_feat_dim)
        node_features = torch.unsqueeze(node_features, dim=1)

        # Tensor, shape (batch_size, 1, node_feat_dim + time_feat_dim)
        query = residual = torch.cat([node_features, node_time_features], dim=2)
        # shape (batch_size, 1, num_heads, self.head_dim)
        query = self.query_projection(query).reshape(query.shape[0], query.shape[1], self.num_heads, self.head_dim)

        # Tensor, shape (batch_size, num_neighbors, node_feat_dim + edge_feat_dim + time_feat_dim)
        key = value = torch.cat([neighbor_node_features, neighbor_node_edge_features, neighbor_node_time_features], dim=2)
        # Tensor, shape (batch_size, num_neighbors, num_heads, self.head_dim)
        key = self.key_projection(key).reshape(key.shape[0], key.shape[1], self.num_heads, self.head_dim)
        # Tensor, shape (batch_size, num_neighbors, num_heads, self.head_dim)
        value = self.value_projection(value).reshape(value.shape[0], value.shape[1], self.num_heads, self.head_dim)

        # Tensor, shape (batch_size, num_heads, 1, self.head_dim)
        query = query.permute(0, 2, 1, 3)
        # Tensor, shape (batch_size, num_heads, num_neighbors, self.head_dim)
        key = key.permute(0, 2, 1, 3)
        # Tensor, shape (batch_size, num_heads, num_neighbors, self.head_dim)
        value = value.permute(0, 2, 1, 3)

        # Tensor, shape (batch_size, num_heads, 1, num_neighbors)
        attention = torch.einsum('bhld,bhnd->bhln', query, key)
        attention = attention * self.scaling_factor

        # Tensor, shape (batch_size, 1, num_neighbors)
        attention_mask = torch.from_numpy(neighbor_masks).to(node_features.device).unsqueeze(dim=1)
        attention_mask = attention_mask == 0
        # Tensor, shape (batch_size, self.num_heads, 1, num_neighbors)
        attention_mask = torch.stack([attention_mask for _ in range(self.num_heads)], dim=1)

        # Tensor, shape (batch_size, self.num_heads, 1, num_neighbors)
        # note that if a node has no valid neighbor (whose neighbor_masks are all zero), directly set the masks to -np.inf will make the
        # attention scores after softmax be nan. Therefore, we choose a very large negative number (-1e10 following TGAT) instead of -np.inf to tackle this case
        attention = attention.masked_fill(attention_mask, -1e10)

        # Tensor, shape (batch_size, num_heads, 1, num_neighbors)
        attention_scores = self.dropout(torch.softmax(attention, dim=-1))

        # Tensor, shape (batch_size, num_heads, 1, self.head_dim)
        attention_output = torch.einsum('bhln,bhnd->bhld', attention_scores, value)

        # Tensor, shape (batch_size, 1, num_heads * self.head_dim), where num_heads * self.head_dim is equal to node_feat_dim + time_feat_dim
        attention_output = attention_output.permute(0, 2, 1, 3).flatten(start_dim=2)

        # Tensor, shape (batch_size, 1, node_feat_dim + time_feat_dim)
        output = self.dropout(self.residual_fc(attention_output))

        # Tensor, shape (batch_size, 1, node_feat_dim + time_feat_dim)
        output = self.layer_norm(output + residual)

        # Tensor, shape (batch_size, node_feat_dim + time_feat_dim)
        output = output.squeeze(dim=1)
        # Tensor, shape (batch_size, num_heads, num_neighbors)
        attention_scores = attention_scores.squeeze(dim=2)

        return output, attention_scores


class TransformerEncoder(nn.Module):

    def __init__(self, attention_dim: int, num_heads: int, dropout: float = 0.1):
        """
        Transformer encoder.
        :param attention_dim: int, dimension of the attention vector
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        """
        super(TransformerEncoder, self).__init__()
        # use the MultiheadAttention implemented by PyTorch
        self.multi_head_attention = nn.MultiheadAttention(embed_dim=attention_dim, num_heads=num_heads, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

        self.linear_layers = nn.ModuleList([
            nn.Linear(in_features=attention_dim, out_features=4 * attention_dim),
            nn.Linear(in_features=4 * attention_dim, out_features=attention_dim)
        ])
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(attention_dim),
            nn.LayerNorm(attention_dim)
        ])

    def forward(self, inputs_query: torch.Tensor, inputs_key: torch.Tensor = None, inputs_value: torch.Tensor = None,
                neighbor_masks: np.ndarray = None):
        """
        encode the inputs by Transformer encoder
        :param inputs_query: Tensor, shape (batch_size, target_seq_length, self.attention_dim)
        :param inputs_key: Tensor, shape (batch_size, source_seq_length, self.attention_dim)
        :param inputs_value: Tensor, shape (batch_size, source_seq_length, self.attention_dim)
        :param neighbor_masks: ndarray, shape (batch_size, source_seq_length), used to create mask of neighbors for nodes in the batch
        :return:
        """
        if inputs_key is None or inputs_value is None:
            assert inputs_key is None and inputs_value is None
            inputs_key = inputs_value = inputs_query
        # note that the MultiheadAttention module accept input data with shape (seq_length, batch_size, input_dim), so we need to transpose the input
        # transposed_inputs_query, Tensor, shape (target_seq_length, batch_size, self.attention_dim)
        # transposed_inputs_key, Tensor, shape (source_seq_length, batch_size, self.attention_dim)
        # transposed_inputs_value, Tensor, shape (source_seq_length, batch_size, self.attention_dim)
        transposed_inputs_query, transposed_inputs_key, transposed_inputs_value = inputs_query.transpose(0, 1), inputs_key.transpose(0, 1), inputs_value.transpose(0, 1)

        if neighbor_masks is not None:
            # Tensor, shape (batch_size, source_seq_length)
            neighbor_masks = torch.from_numpy(neighbor_masks).to(inputs_query.device) == 0

        # Tensor, shape (batch_size, target_seq_length, self.attention_dim)
        hidden_states = self.multi_head_attention(query=transposed_inputs_query, key=transposed_inputs_key,
                                                  value=transposed_inputs_value, key_padding_mask=neighbor_masks)[0].transpose(0, 1)
        # Tensor, shape (batch_size, target_seq_length, self.attention_dim)
        outputs = self.norm_layers[0](inputs_query + self.dropout(hidden_states))
        # Tensor, shape (batch_size, target_seq_length, self.attention_dim)
        hidden_states = self.linear_layers[1](self.dropout(F.relu(self.linear_layers[0](outputs))))
        # Tensor, shape (batch_size, target_seq_length, self.attention_dim)
        outputs = self.norm_layers[1](outputs + self.dropout(hidden_states))

        return outputs
