import torch
import torch.nn as nn
import torch.nn.functional as F

debug_on = False
device = None


class Spatial_Attention_layer(nn.Module):
    """
    compute spatial attention scores
    """
    def __init__(self, num_of_vertices, num_of_features, num_of_timesteps):
        """
        Compute spatial attention scores
        :param num_of_vertices: int
        :param num_of_features: int
        :param num_of_timesteps: int
        """
        super(Spatial_Attention_layer, self).__init__()

        global device
        self.W_1 = torch.randn(num_of_timesteps, requires_grad=True).to(device)
        self.W_2 = torch.randn(num_of_features, num_of_timesteps, requires_grad=True).to(device)
        self.W_3 = torch.randn(num_of_features, requires_grad=True).to(device)
        self.b_s = torch.randn(1, num_of_vertices, num_of_vertices, requires_grad=True).to(device)
        self.V_s = torch.randn(num_of_vertices, num_of_vertices, requires_grad=True).to(device)

    def forward(self, x):
        """
        Parameters
        ----------
        x: tensor, x^{(r - 1)}_h,
           shape is (batch_size, N, C_{r-1}, T_{r-1})

           initially, N == num_of_vertices (V)

        Returns
        ----------
        S_normalized: tensor, S', spatial attention scores
                      shape is (batch_size, N, N)

        """
        # get shape of input matrix x
        # batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        # The shape of x could be different for different layer, especially the last two dimensions

        # compute spatial attention scores
        # shape of lhs is (batch_size, V, T)
        lhs = torch.matmul(torch.matmul(x, self.W_1), self.W_2)

        # shape of rhs is (batch_size, T, V)
        # rhs = torch.matmul(self.W_3, x.transpose((2, 0, 3, 1)))
        rhs = torch.matmul(x.permute((0, 3, 1, 2)), self.W_3)  # do we need to do transpose??

        # shape of product is (batch_size, V, V)
        product = torch.matmul(lhs, rhs)

        S = torch.matmul(self.V_s, torch.sigmoid(product + self.b_s))

        # normalization
        S = S - torch.max(S, 1, keepdim=True)[0]
        exp = torch.exp(S)
        S_normalized = exp / torch.sum(exp, 1, keepdim=True)
        return S_normalized


class cheb_conv_with_SAt(nn.Module):
    """
    K-order chebyshev graph convolution with Spatial Attention scores
    """

    def __init__(self, num_of_filters, K, cheb_polynomials, num_of_features):
        """
        Parameters
        ----------
        num_of_filters: int

        num_of_features: int, num of input features

        K: int, up K - 1 order chebyshev polynomials
                will be used in this convolution

        """
        super(cheb_conv_with_SAt, self).__init__()
        self.K = K
        self.num_of_filters = num_of_filters
        self.cheb_polynomials = cheb_polynomials

        global device
        self.Theta = torch.randn(self.K, num_of_features, num_of_filters, requires_grad=True).to(device)

    def forward(self, x, spatial_attention):
        """
        Chebyshev graph convolution operation

        Parameters
        ----------
        x: mx.ndarray, graph signal matrix
           shape is (batch_size, N, F, T_{r-1}), F is the num of features

        spatial_attention: mx.ndarray, shape is (batch_size, N, N)
                           spatial attention scores

        Returns
        ----------
        mx.ndarray, shape is (batch_size, N, self.num_of_filters, T_{r-1})

        """
        (batch_size, num_of_vertices,
         num_of_features, num_of_timesteps) = x.shape

        global device

        outputs = []
        for time_step in range(num_of_timesteps):
            # shape is (batch_size, V, F)
            graph_signal = x[:, :, :, time_step]
            output = torch.zeros(batch_size, num_of_vertices,
                                 self.num_of_filters).to(device)  # do we need to set require_grad=True?
            for k in range(self.K):
                # shape of T_k is (V, V)
                T_k = self.cheb_polynomials[k]

                # shape of T_k_with_at is (batch_size, V, V)
                T_k_with_at = T_k * spatial_attention

                # shape of theta_k is (F, num_of_filters)
                # theta_k = self.Theta.data()[k]
                theta_k = self.Theta[k]

                # shape is (batch_size, V, F)
                # rhs = nd.batch_dot(T_k_with_at.transpose((0, 2, 1)),  # why do we need to transpose?
                #                    graph_signal)
                rhs = torch.matmul(T_k_with_at.permute((0, 2, 1)),
                                   graph_signal)

                output = output + torch.matmul(rhs, theta_k)
            # outputs.append(output.expand_dims(-1))
            outputs.append(torch.unsqueeze(output, -1))
        return F.relu(torch.cat(outputs, dim=-1))


class Temporal_Attention_layer(nn.Module):
    """
    compute temporal attention scores
    """

    def __init__(self, num_of_vertices, num_of_features, num_of_timesteps):
        """
        Temporal Attention Layer
        :param num_of_vertices: int
        :param num_of_features: int
        :param num_of_timesteps: int
        """
        super(Temporal_Attention_layer, self).__init__()

        global device
        self.U_1 = torch.randn(num_of_vertices, requires_grad=True).to(device)
        self.U_2 = torch.randn(num_of_features, num_of_vertices, requires_grad=True).to(device)
        self.U_3 = torch.randn(num_of_features, requires_grad=True).to(device)
        self.b_e = torch.randn(1, num_of_timesteps, num_of_timesteps, requires_grad=True).to(device)
        self.V_e = torch.randn(num_of_timesteps, num_of_timesteps, requires_grad=True).to(device)

    def forward(self, x):
        """
        Parameters
        ----------
        x: torch.tensor, x^{(r - 1)}_h
                       shape is (batch_size, V, C_{r-1}, T_{r-1})

        Returns
        ----------
        E_normalized: torch.tensor, S', spatial attention scores
                      shape is (batch_size, T_{r-1}, T_{r-1})

        """
        # _, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        # N == batch_size
        # V == num_of_vertices
        # C == num_of_features
        # T == num_of_timesteps

        # compute temporal attention scores
        # shape of lhs is (N, T, V)
        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U_1),
                           self.U_2)

        # shape is (batch_size, V, T)
        # rhs = torch.matmul(self.U_3, x.transpose((2, 0, 1, 3)))
        rhs = torch.matmul(x.permute((0, 1, 3, 2)), self.U_3)  # Is it ok to switch the position?

        product = torch.matmul(lhs, rhs)  # wd: (batch_size, T, T)

        # (batch_size, T, T)
        E = torch.matmul(self.V_e, torch.sigmoid(product + self.b_e))

        # normailzation
        E = E - torch.max(E, 1, keepdim=True)[0]
        exp = torch.exp(E)
        E_normalized = exp / torch.sum(exp, 1, keepdim=True)
        return E_normalized


class ASTGCN_block(nn.Module):
    def __init__(self, num_of_vertices, num_of_features, num_of_timesteps, backbone):
        """
        Parameters
        ----------
        backbone: dict, should have 6 keys,
                        "K",
                        "num_of_chev_filters",
                        "num_of_time_filters",
                        "time_conv_kernel_size",  # wd: never used?? Actually there is no such key in backbone...
                        "time_conv_strides",
                        "cheb_polynomials"
        """
        super(ASTGCN_block, self).__init__()

        K = backbone['K']
        num_of_chev_filters = backbone['num_of_chev_filters']
        num_of_time_filters = backbone['num_of_time_filters']
        time_conv_strides = backbone['time_conv_strides']
        cheb_polynomials = backbone["cheb_polynomials"]

        self.SAt = Spatial_Attention_layer(num_of_vertices, num_of_features, num_of_timesteps)

        self.cheb_conv_SAt = cheb_conv_with_SAt(
            num_of_filters=num_of_chev_filters,
            K=K,
            cheb_polynomials=cheb_polynomials,
            num_of_features=num_of_features)

        self.TAt = Temporal_Attention_layer(num_of_vertices, num_of_features, num_of_timesteps)

        self.time_conv = nn.Conv2d(
            in_channels=num_of_chev_filters,
            out_channels=num_of_time_filters,
            kernel_size=(1, 3),
            stride=(1, time_conv_strides),
            padding=(0, 1))

        self.residual_conv = nn.Conv2d(
            in_channels=num_of_features,
            out_channels=num_of_time_filters,
            kernel_size=(1, 1),
            stride=(1, time_conv_strides))

        self.ln = nn.LayerNorm(num_of_time_filters)

    def forward(self, x):
        """
        Parameters
        ----------
        x: torch.tensor, shape is (batch_size, N, C_{r-1}, T_{r-1})

        Returns
        ----------
        torch.tensor, shape is (batch_size, N, num_of_time_filters, T_{r-1})

        """
        (batch_size, num_of_vertices,
         num_of_features, num_of_timesteps) = x.shape

        # shape of temporal_At: (batch_size, num_of_timesteps, num_of_timesteps)
        temporal_At = self.TAt(x)

        x_TAt = torch.matmul(x.reshape(batch_size, -1, num_of_timesteps),
                             temporal_At) \
            .reshape(batch_size, num_of_vertices,
                     num_of_features, num_of_timesteps)

        # cheb gcn with spatial attention
        # (batch_size, num_of_vertices, num_of_vertices)
        spatial_At = self.SAt(x_TAt)

        # (batch_size, num_of_vertices, num_of_features, num_of_timesteps)
        spatial_gcn = self.cheb_conv_SAt(x, spatial_At)

        # convolution along time axis
        # (batch_size, num_of_vertices, num_of_time_filters, num_of_timesteps)
        time_conv_output = (self.time_conv(spatial_gcn.permute(0, 2, 1, 3))
                            .permute(0, 2, 1, 3))

        # residual shortcut
        # (batch_size, num_of_vertices, num_of_time_filters, num_of_timesteps)
        x_residual = (self.residual_conv(x.permute(0, 2, 1, 3))
                      .permute(0, 2, 1, 3))

        # (batch_size, num_of_vertices, num_of_time_filters, num_of_timesteps)
        relued = F.relu(x_residual + time_conv_output)
        return self.ln(relued.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)


class ASTGCN_submodule(nn.Module):
    def __init__(self, num_for_prediction, backbones, num_of_vertices, num_of_features, num_of_timesteps):
        """
        submodule to deal with week, day, and hour individually.
        :param num_for_prediction: int
        :param backbones: dict
        :param num_of_vertices: int
        :param num_of_features: int
        :param num_of_timesteps: list of int. It includes the num_of_timestep of the input layer, and also of the next layer
        """
        super(ASTGCN_submodule, self).__init__()

        all_num_of_features = [num_of_features, backbones[0]["num_of_time_filters"]]
        self.blocks = nn.Sequential(*[ASTGCN_block(num_of_vertices,
                                                   all_num_of_features[idx],
                                                   num_of_timesteps[idx],
                                                   backbone)
                                      for idx, backbone in enumerate(backbones)])

        # in_channels: num_of_timestemps, i.e. num_of_weeks/days/hours * args.points_per_hour
        # input(batch, C_in, H_in, W_in)  => output(batch, C_out, H_out, W_out)
        # when padding is 0, and kernel_size[0] is 0, H_out == H_in  --> num_of_vertices
        # out_height = floor((height + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1)/stride[0]) + 1
        # out_width = floor((width + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1)/stride[1]) + 1
        self.final_conv = nn.Conv2d(in_channels=num_of_timesteps[-1],
                                    out_channels=num_for_prediction,
                                    kernel_size=(1, backbones[-1]['num_of_time_filters']))

        global device
        self.W = torch.randn(num_of_vertices, num_for_prediction, requires_grad=True).to(device)

    def forward(self, x):
        """
        Parameters
        ----------
        x: torch.tensor,
           shape is (batch_size, num_of_vertices,
                     num_of_features, num_of_timesteps)

        Returns
        ----------
        torch.tensor, shape is (batch_size, num_of_vertices, num_for_prediction)

        """
        x = self.blocks(x)
        # the output x's shape: (batch_size, num_of_vertices, num_of_time_filters, num_of_timesteps)

        # the shape of final_conv()'s output: (batch, num_for_prediction, num_of_vertices, num_of_features_out)
        module_output = (self.final_conv(x.permute((0, 3, 1, 2)))
                         [:, :, :, -1].permute((0, 2, 1)))
        # TODO: why choose the last one of the feature dimension?
        # _, num_of_vertices, num_for_prediction = module_output.shape

        #    (batch_size, num_of_vertices, num_for_prediction)
        #  *             (num_of_vertices, num_for_prediction)
        # => (batch_size, num_of_vertices, num_for_prediction)
        return module_output * self.W


class ASTGCN(nn.Module):
    def __init__(self, num_for_prediction, all_backbones, num_of_vertices, num_of_features, num_of_timesteps, _device):
        """
        Parameters
        ----------
        num_for_prediction: int
            how many time steps will be forecasting

        all_backbones: list[list],
                       3 backbones for "hour", "day", "week" submodules.
                       "week", "day", "hour" (in order)

        num_of_vertices: int
            The number of vertices in the graph

        num_of_features: int
            The number of features of each measurement

        num_of_timesteps: 2D array, shape=(3, 2)
            The timestemps for each time scale (week, day, hour).
            Each row is [input_timesteps, output_timesteps].
        """
        super(ASTGCN, self).__init__()

        global device
        device = _device

        if debug_on:
            print("ASTGCN model:")
            print("num for prediction: ", num_for_prediction)
            print("num of vertices: ", num_of_vertices)
            print("num of features: ", num_of_features)
            print("num of timesteps: ", num_of_timesteps)

        self.submodules = nn.ModuleList([ASTGCN_submodule(num_for_prediction,
                                                          backbones,
                                                          num_of_vertices,
                                                          num_of_features,
                                                          num_of_timesteps[idx])
                                         for idx, backbones in enumerate(all_backbones)])

    def forward(self, x_list):
        """
        Parameters
        ----------
        x_list: list[torch.tensor],
                shape of each element is (batch_size, num_of_vertices,
                                        num_of_features, num_of_timesteps)

        Returns
        ----------
        Y_hat: torch.tensor,
               shape is (batch_size, num_of_vertices, num_for_prediction)
        """
        if debug_on:
            for x in x_list:
                print('Shape of input to the model:', x.shape)

        if len(x_list) != len(self.submodules):
            raise ValueError("num of submodule not equals to "
                             "length of the input list")

        num_of_vertices_set = {i.shape[1] for i in x_list}
        if len(num_of_vertices_set) != 1:
            raise ValueError("Different num_of_vertices detected! "
                             "Check if your input data have same size"
                             "at axis 1.")

        batch_size_set = {i.shape[0] for i in x_list}
        if len(batch_size_set) != 1:
            raise ValueError("Input values must have same batch size!")

        submodule_outputs = []
        for idx, submodule in enumerate(self.submodules):
            submodule_result = submodule(x_list[idx])
            submodule_result = torch.unsqueeze(submodule_result, dim=-1)
            submodule_outputs.append(submodule_result)

        submodule_outputs = torch.cat(submodule_outputs, dim=-1)

        return torch.sum(submodule_outputs, dim=-1)
