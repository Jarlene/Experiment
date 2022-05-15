from turtle import forward
from typing import List, Callable, Union, Any, Type, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal

class FeaturesLinear(torch.nn.Module):

    def __init__(self, field_dims, output_dim=1):
        super().__init__()
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.fc(x), dim=1) + self.bias


class FeaturesEmbedding(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)


class FieldAwareFactorizationMachine(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.num_fields = len(field_dims)
        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(sum(field_dims), embed_dim) for _ in range(self.num_fields)
        ])
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        for embedding in self.embeddings:
            torch.nn.init.xavier_uniform_(embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        xs = [self.embeddings[i](x) for i in range(self.num_fields)]
        ix = list()
        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                ix.append(xs[j][:, i] * xs[i][:, j])
        ix = torch.stack(ix, dim=1)
        return ix


class FactorizationMachine(torch.nn.Module):

    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix


class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim, hiden_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for dim in hiden_dims:
            layers.append(torch.nn.Linear(input_dim, dim))
            layers.append(torch.nn.BatchNorm1d(dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, input_dim)``
        """
        return self.mlp(x)


class InnerProductNetwork(torch.nn.Module):

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        num_fields = x.shape[1]
        row, col = list(), list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                row.append(i), col.append(j)
        return torch.sum(x[:, row] * x[:, col], dim=2)


class OuterProductNetwork(torch.nn.Module):

    def __init__(self, num_fields, embed_dim, kernel_type='mat'):
        super().__init__()
        num_ix = num_fields * (num_fields - 1) // 2
        if kernel_type == 'mat':
            kernel_shape = embed_dim, num_ix, embed_dim
        elif kernel_type == 'vec':
            kernel_shape = num_ix, embed_dim
        elif kernel_type == 'num':
            kernel_shape = num_ix, 1
        else:
            raise ValueError('unknown kernel type: ' + kernel_type)
        self.kernel_type = kernel_type
        self.kernel = torch.nn.Parameter(torch.zeros(kernel_shape))
        torch.nn.init.xavier_uniform_(self.kernel.data)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        num_fields = x.shape[1]
        row, col = list(), list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                row.append(i), col.append(j)
        p, q = x[:, row], x[:, col]
        if self.kernel_type == 'mat':
            kp = torch.sum(p.unsqueeze(1) * self.kernel, dim=-1).permute(0, 2, 1)
            return torch.sum(kp * q, -1)
        else:
            return torch.sum(p * q * self.kernel.unsqueeze(0), -1)


class CrossNetwork(torch.nn.Module):

    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.w = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, 1, bias=False) for _ in range(num_layers)
        ])
        self.b = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros((input_dim,))) for _ in range(num_layers)
        ])

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        x0 = x
        for i in range(self.num_layers):
            xw = self.w[i](x)
            x = x0 * xw + self.b[i] + x
        return x


class AttentionalFactorizationMachine(torch.nn.Module):

    def __init__(self, embed_dim, attn_size, dropouts):
        super().__init__()
        self.attention = torch.nn.Linear(embed_dim, attn_size)
        self.projection = torch.nn.Linear(attn_size, 1)
        self.fc = torch.nn.Linear(embed_dim, 1)
        self.dropouts = dropouts

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        num_fields = x.shape[1]
        row, col = list(), list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                row.append(i), col.append(j)
        p, q = x[:, row], x[:, col]
        inner_product = p * q
        attn_scores = F.relu(self.attention(inner_product))
        attn_scores = F.softmax(self.projection(attn_scores), dim=1)
        attn_scores = F.dropout(attn_scores, p=self.dropouts[0], training=self.training)
        attn_output = torch.sum(attn_scores * inner_product, dim=1)
        attn_output = F.dropout(attn_output, p=self.dropouts[1], training=self.training)
        return self.fc(attn_output)


class CompressedInteractionNetwork(torch.nn.Module):

    def __init__(self, input_dim, cross_layer_sizes, split_half=True):
        super().__init__()
        self.num_layers = len(cross_layer_sizes)
        self.split_half = split_half
        self.conv_layers = torch.nn.ModuleList()
        prev_dim, fc_input_dim = input_dim, 0
        for i in range(self.num_layers):
            cross_layer_size = cross_layer_sizes[i]
            self.conv_layers.append(torch.nn.Conv1d(input_dim * prev_dim, cross_layer_size, 1,
                                                    stride=1, dilation=1, bias=True))
            if self.split_half and i != self.num_layers - 1:
                cross_layer_size //= 2
            prev_dim = cross_layer_size
            fc_input_dim += prev_dim
        self.fc = torch.nn.Linear(fc_input_dim, 1)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        xs = list()
        x0, h = x.unsqueeze(2), x
        for i in range(self.num_layers):
            x = x0 * h.unsqueeze(1)
            batch_size, f0_dim, fin_dim, embed_dim = x.shape
            x = x.view(batch_size, f0_dim * fin_dim, embed_dim)
            x = F.relu(self.conv_layers[i](x))
            if self.split_half and i != self.num_layers - 1:
                x, h = torch.split(x, x.shape[1] // 2, dim=1)
            else:
                h = x
            xs.append(x)
        return self.fc(torch.sum(torch.cat(xs, dim=1), 2))


class AnovaKernel(torch.nn.Module):

    def __init__(self, order, reduce_sum=True):
        super().__init__()
        self.order = order
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        batch_size, num_fields, embed_dim = x.shape
        a_prev = torch.ones((batch_size, num_fields + 1, embed_dim), dtype=torch.float).to(x.device)
        for t in range(self.order):
            a = torch.zeros((batch_size, num_fields + 1, embed_dim), dtype=torch.float).to(x.device)
            a[:, t+1:, :] += x[:, t:, :] * a_prev[:, t:-1, :]
            a = torch.cumsum(a, dim=1)
            a_prev = a
        if self.reduce_sum:
            return torch.sum(a[:, -1, :], dim=-1, keepdim=True)
        else:
            return a[:, -1, :]


class CrossLayer(torch.nn.Module):
    def __init__(self,
                 feature_nums,  # 需要交叉的两个tensor的特征数
                 emb_size=8,
                 w_channels=1,
                 use_mask=True,  # 第一层交叉是特征自己与自己交叉，需要mask重复向量，后续不需要mask
                 use_bn=True,
                 **kwargs):
        super(CrossLayer, self).__init__(**kwargs)
        self.w_channels = w_channels
        self.use_bn = use_bn
        self.feature_num0 = feature_nums[0]
        self.feature_num1 = feature_nums[1]
        self.emb_size = emb_size
        self.use_mask = use_mask

        self.W = torch.nn.Parameter(torch.zeros(1, 1, self.w_channels, self.emb_size, self.emb_size), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.W)
        self.register_parameter('W', self.W)


        ones = torch.ones(self.feature_num1, self.feature_num0)
        ones = torch.tril(ones, diagonal=-1)
        if self.use_mask:
            self.mask = ones
            self.mask = torch.unsqueeze(self.mask, dim=0)
            self.mask = torch.unsqueeze(self.mask, dim=-1)
            self.mask = torch.unsqueeze(self.mask, dim=-1)
            self.mask = self.mask == 1
            self.mask = torch.nn.Parameter(self.mask, requires_grad=False)
        
        self.interaction_num = torch.sum(ones).numpy().astype(np.int).tolist()
        if self.use_bn:
            self.bn = torch.nn.BatchNorm2d(self.interaction_num)

    
    def forward(self, xi, xj):
        v_x_1 = torch.unsqueeze(xi, dim=1)  # [batch, 1, feature_num0, emb]
        v_x_2 = torch.unsqueeze(xj, dim=2)  # [batch, feature_num1, 1, emb]
        v_x_1 = torch.unsqueeze(v_x_1, dim=-2)  # [batch, 1, feature_num0, 1, emb]
        v_x_2 = torch.unsqueeze(v_x_2, dim=-1)  # [batch, feature_num1, 1, emb, 1]
        raw_cross = v_x_1 * v_x_2  # [batch, feature_num1, feature_num0, emb, emb]
        if self.use_mask:            
            self.mask = self.mask.to(xi.device)
            mask_cross = torch.masked_select(raw_cross, self.mask)
            mask_cross = torch.reshape(mask_cross, (-1, self.interaction_num, self.emb_size, self.emb_size))
            # shape mask be explicitly set for eager mode.
            # [batcsh, n*(n-1)/2, emb, emb]
        else:
            mask_cross = torch.reshape(raw_cross, [-1, self.interaction_num, self.emb_size, self.emb_size])

        
        if self.use_bn:
            mask_cross = self.bn(mask_cross)
        
        
        # broadcast feature map to w_channel
        # [batch, interaction_num, w_channel,  emb, emb)
        mask_cross = torch.unsqueeze(mask_cross, dim=2)
        mask_cross = torch.repeat_interleave(mask_cross, self.w_channels, dim=2)

        # step 3. optional structures
        mask_cross = mask_cross * self.W  # [batch, interaction_num, w_channel, emb, emb]
        return torch.transpose(mask_cross, 1, 2)   # [batch, w_channel, interaction_num, emb, emb]

class FuseLayer(torch.nn.Module):

    def __init__(self,
                 feature_nums,  # 需要交叉的两个tensor的特征数
                 w_channels=1,
                 use_bn=True,
                 **kwargs):
        super(FuseLayer, self).__init__(**kwargs)
        self.use_bn = use_bn
        self.w_channels = w_channels
        self.use_bn = use_bn
        self.feature_num0 = feature_nums[0]
        self.feature_num1 = feature_nums[1]
        ones = torch.ones(self.feature_num1, self.feature_num0)
        ones = torch.tril(ones, diagonal=-1)

        self.interaction_num = torch.sum(ones).numpy().astype(np.int).tolist()
        if use_bn:
            self.bn = torch.nn.BatchNorm3d(self.w_channels)

        self.W = torch.nn.Parameter(torch.zeros(1, self.w_channels, self.interaction_num,  1, 1), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.W)



    def forward(self, inputs): 
        if self.use_bn:
            inputs_bn = self.bn(inputs)
        else:
            inputs_bn = inputs
        # step 2. add weight
        z = inputs_bn * self.W
        z = torch.sum(z, dim=-1)
        z = torch.sum(z, dim=2)
        return z


class InteractionLayer(torch.nn.Module):
    def __init__(self, 
                field_nums,  
                emb_size,                  
                use_bn=True,
                use_atten = False,
                attn_size: Optional[int] = None) -> None:
        super().__init__()
        self.feature_num0 = field_nums[0]
        self.feature_num1 = field_nums[1]
        self.emb_size = emb_size
        self.use_bn = use_bn
        self.use_atten = use_atten
        self.row, self.col = list(), list()
        for i in range(self.feature_num0):
            for j in range(i + 1, self.feature_num1):
                self.row.append(i), self.col.append(j)
        self.interaction_num = len(self.row)
        if self.use_bn:
            self.bn = torch.nn.BatchNorm1d(self.interaction_num)
        self.W = torch.nn.Parameter(torch.zeros(1, self.interaction_num, self.emb_size))
        torch.nn.init.xavier_uniform_(self.W)

        if self.use_atten:
            if attn_size is None:
                attn_size = self.emb_size
            self.attention = torch.nn.Linear(emb_size, attn_size)
            self.projection = torch.nn.Linear(attn_size, 1)

    def forward(self, xi, xj):
        p = xi[:,self.row,:]
        q = xj[:,self.col,:]
        out = p * q
        if self.use_bn:
            out = self.bn(out)
        out = out * self.W
        if self.use_atten:
            attn_scores = F.relu(self.attention(out))
            attn_scores = F.softmax(self.projection(attn_scores), dim=1)
            out = attn_scores * out
        return out

class FusionLayer(torch.nn.Module):
    
    def __init__(self,
                 field_nums,  # 需要交叉的两个tensor的特征数
                 use_bn=False,
                 **kwargs):
        super(FusionLayer, self).__init__(**kwargs)
        self.use_bn = use_bn
        self.use_bn = use_bn
        self.feature_num0 = field_nums[0]
        self.feature_num1 = field_nums[1]
        self.use_bn = use_bn
        ones = torch.ones(self.feature_num1, self.feature_num0)
        ones = torch.tril(ones, diagonal=-1)

        self.interaction_num = torch.sum(ones).numpy().astype(np.int).tolist()
        self.W = torch.nn.Parameter(torch.zeros(1, self.interaction_num, 1), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.W)
        if use_bn:
            self.bn = torch.nn.BatchNorm1d(self.interaction_num)

    def forward(self, x):
        if self.use_bn:
            inputs_bn = self.bn(x)
        else:
            inputs_bn = x

        z = inputs_bn * self.W
        z = torch.sum(z, dim=1)
        return z


class SparseLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()


class DenseLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()


class MoE(torch.nn.Module):

    def __init__(self, 
                 experts_nums,
                 input_size,
                 expert_model: torch.nn.Module,
                 topK : int = 4,
                 gate_type : str = 'noise',
                 ) -> None:
        super(MoE, self).__init__()
        self.experts_nums = experts_nums
        self.expert_model = expert_model
        self.topK = topK
        self.gate_type = gate_type
        self.experts = torch.nn.ModuleList([expert_model for i in range(self.num_experts)])

        self.w_gate = torch.nn.Parameter(torch.zeros(input_size, experts_nums), requires_grad=True)
        self.w_noise = torch.nn.Parameter(torch.zeros(input_size, experts_nums), requires_grad=True)


        self.softplus = torch.nn.Softplus()
        self.softmax = torch.nn.Softmax(dim = 1)
        self.normal = Normal(torch.tensor([0.0]), torch.tensor([1.0]))

        assert(self.topK <= self.num_experts)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """

        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()
        threshold_positions_if_in = torch.arange(batch) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat,0 , threshold_positions_if_out), 1)
        # is each value currently in the top k.
        prob_if_in = self.normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = self.normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def gate(self, x):
        clean_logits = x @ self.w_gate
        if self.gate_type == 'noise':
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = (self.softplus(raw_noise_stddev) + 1e-2)
            noisy_logits = clean_logits + ( torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits
        
        top_logits, top_indices = logits.topk(min(self.topK + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.topK]
        top_k_indices = top_indices[:, :self.topK]
        top_k_gates = self.softmax(top_k_logits)
        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        if self.gate_type == 'noise' and self.topK < self.num_experts:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x):
        gates, load = self.noisy_top_k_gating(x)
        # importance = gates.sum(0)
        # loss = self.cv_squared(importance) + self.cv_squared(load)
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        _, _expert_index = sorted_experts.split(1, dim=1)
        _batch_index = sorted_experts[index_sorted_experts[:, 1],0]
        _part_sizes = list((gates > 0).sum(0).numpy())
        gates_exp = gates[_batch_index.flatten()]
        _nonzero_gates = torch.gather(gates_exp, 1, _expert_index)
        inp_exp = x[_batch_index].squeeze(1)
        expert_inputs = torch.split(inp_exp, _part_sizes, dim=0)
        gates = torch.split(_nonzero_gates, _part_sizes, dim=0)
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        stitched = torch.cat(expert_outputs, 0).exp()
        stitched = stitched.mul(_nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_outputs[-1].size(1), requires_grad=True)
        combined = zeros.index_add(0, _batch_index, stitched.float())
        combined[combined == 0] = np.finfo(float).eps
        y = combined.log()
        return y

