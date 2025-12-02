
import torch
from torch import nn
from typing import Tuple


class PFGRUCell(torch.nn.Module):
    """
    Based on:
    Particle Filter Recurrent Neural Networks (2020)
    https://github.com/Yusufma03/pfrnns
    """
    """
    This is the base class for the PF-RNNs. We implement the shared functions here, including
        1. soft-resampling
        2. reparameterization trick
        3. obs_extractor o_t(x_t)
        4. control_extractor u_t(x_t)

        All particles in PF-RNNs are processed in parallel to benefit from GPU parallelization.
    """
    # TODO document differences with original implementation
    def __init__(
            self,
            input_size,
            hidden_size,
            num_particles,
            resamp_alpha):
        """
        :param input_size: the size of input x_t
        :param hidden_size: the size of the hidden particle h_t^i
        :param num_particles: number of particles for a PF-RNN
        :param resamp_alpha: the control parameter \alpha for soft-resampling.
        We use the importance sampling with a proposal distribution q(i) = \alpha w_t^i + (1 - \alpha) (1 / K)
        """
        super().__init__()

        self.input_size = input_size
        self.h_dim = hidden_size
        self.num_particles = num_particles
        self.resamp_alpha = resamp_alpha
        self.fc_z = nn.Linear(self.h_dim + self.input_size, self.h_dim)
        self.fc_r = nn.Linear(self.h_dim + self.input_size, self.h_dim)
        self.fc_n = nn.Linear(self.h_dim + self.input_size, self.h_dim * 2)
        self.fc_obs = nn.Linear(self.input_size + self.h_dim, 1)
        self.batch_norm = nn.BatchNorm1d(self.num_particles)

    def resampling(self, particles, prob):
        """
        The implementation of soft-resampling. We implement soft-resampling in a batch-manner.

        :param particles: \{(h_t^i, c_t^i)\}_{i=1}^K for PF-LSTM and \{h_t^i\}_{i=1}^K for PF-GRU.
                        each tensor has a shape: [num_particles * batch_size, h_dim]
        :param prob: weights for particles in the log space. Each tensor has a shape: [num_particles * batch_size, 1]
        :return: resampled particles and weights according to soft-resampling scheme.
        """
        resamp_prob = self.resamp_alpha * torch.exp(prob) + (1 - self.resamp_alpha) * 1 / self.num_particles
        resamp_prob = resamp_prob.view(self.num_particles, -1)
        indices = torch.multinomial(resamp_prob.transpose(0, 1),
                                    num_samples=self.num_particles, replacement=True)
        batch_size = indices.size(0)
        indices = indices.transpose(1, 0).contiguous()
        offset = torch.arange(batch_size, dtype=torch.long, device=indices.device).unsqueeze(0)
        indices = offset + indices * batch_size
        flatten_indices = indices.view(-1, 1).squeeze()

        particles_new = particles[flatten_indices]

        prob_new = torch.exp(prob.view(-1, 1)[flatten_indices])
        prob_new = prob_new / (self.resamp_alpha * prob_new + (1 - self.resamp_alpha) / self.num_particles)
        
        # TODO here is where I get problems
        if not torch.all(prob_new > 0.):
            breakpoint()

        prob_new = torch.log(prob_new).view(self.num_particles, -1, 1)
        prob_new = prob_new - torch.logsumexp(prob_new, dim=0, keepdim=True)
        prob_new = prob_new.view(-1, 1)

        return particles_new, prob_new

    def reparameterize(self, mu, var):
        """
        Reparameterization trick

        :param mu: mean
        :param var: variance
        :return: new samples from the Gaussian distribution
        """
        std = torch.nn.functional.softplus(var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, input: torch.Tensor, hx: Tuple[torch.Tensor, torch.Tensor]):
        # hx is composed of hidden state (h0) and log(weights) (p0)
        h0, p0 = hx

        # update gate
        z = torch.sigmoid(self.fc_z(torch.cat((h0, input), dim=1)))
        # reset gate
        r = torch.sigmoid(self.fc_r(torch.cat((h0, input), dim=1)))
        # memory
        n = self.fc_n(torch.cat((r * h0, input), dim=1))

        mu_n, var_n = torch.split(n, self.h_dim, dim=1)
        n = self.reparameterize(mu_n, var_n)

        # batch norm and relu replace hyperbolic tangent function
        n = n.view(self.num_particles, -1, self.h_dim).transpose(0, 1).contiguous()
        n = self.batch_norm(n)
        n = n.transpose(0, 1).contiguous().view(-1, self.h_dim)
        n = nn.functional.leaky_relu(n)

        # update of hidden state
        h1 = (1 - z) * n + z * h0

        # particle weight update
        att = torch.cat((h1, input), dim=1)
        p1 = self.fc_obs(att) + p0

        # normalize log of weights
        p1 = p1.view(self.num_particles, -1, 1)
        p1 = p1 - torch.logsumexp(p1, dim=0, keepdim=True)

        # soft-resampling
        h1, p1 = self.resampling(h1, p1)

        return h1, p1


class PFGRU(torch.nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_particles: int,
            num_layers: int,
            dropout_prob: float,
            resamp_alpha: float,
            device: str):
        super().__init__()

        self._num_layers = num_layers
        self._hidden_size = hidden_size
        self._num_particles = num_particles
        self._device = device

        self._rnn_cells = torch.nn.ModuleList()
        self._rnn_cells.append(
            PFGRUCell(
                input_size=input_size,
                hidden_size=hidden_size,
                num_particles=num_particles,
                resamp_alpha=resamp_alpha))
        for _ in range(num_layers - 1):
            self._rnn_cells.append(
                PFGRUCell(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    num_particles=num_particles,
                    resamp_alpha=resamp_alpha))

        # self._dropout = nn.Dropout(dropout_prob)

        self.to(device)

    def forward(self, input: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]):

        no_batch = False
        if input.ndim == 2:
            # add batch dimension
            input = input[..., None, :]
            no_batch = True

        # input has shape (seq_len, batch, input_size) (standard PyTorch)
        # hidden is tuple with shapes:
        #   (num_layers, batch_size * num_particles, hidden_size)
        #   (num_layers, batch_size * num_particles, 1)

        # repeat the batch dimension when using PF-RNN
        # TODO separate batch and particles dimensions
        input = input.repeat(1, self._num_particles, 1)

        # split into hidden state and log(weights)
        hidden, probs = hidden
        # unbind tensor and create list to prevent in-place operations, which
        # lead to autograd errors
        hidden = list(hidden.unbind(0))
        probs = list(probs.unbind(0))

        seq_len = input.shape[0]
        # outputs of shape (seq_len, batch, ?)
        output = []

        for step in range(seq_len):
            layer_input = input[step]
            for layer, rnn_cell in enumerate(self._rnn_cells):
                hidden[layer], probs[layer] = rnn_cell(layer_input, (hidden[layer], probs[layer]))

                # hidden[layer] = self._dropout(hidden[layer])

                # calculate mean hidden-state
                layer_input = hidden[layer] * torch.exp(probs[layer])

            output.append(layer_input)

        output = torch.stack(output, dim=0)
        # separate batch and particles dimensions
        # output has shape (seq_len, num_particles, batch, hidden_size)
        output = output.reshape([seq_len, self._num_particles, -1, self._hidden_size])

        # swap the first axes to have shape (num_particles, seq_length, ...)
        output = torch.swapaxes(output, 0, 1)

        # remove batch dimension if required
        if no_batch:
            output = torch.squeeze(output, dim=2)

        hidden = torch.stack(hidden, dim=0)
        probs = torch.stack(probs, dim=0)

        return output, (hidden, probs)


class UncertaintyPFGRU(torch.nn.Module):
    """
    TODO:
        - enable inference on individual predictions (like the other networks)
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            output_size: int,
            num_layers: int,
            num_particles: int,
            dropout_prob: float,
            resamp_alpha: float,
            device: str):

        super().__init__()

        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._num_particles = num_particles
        self._device = device

        self._pfgru = PFGRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_particles=num_particles,
            num_layers=num_layers,
            dropout_prob=dropout_prob,
            resamp_alpha=resamp_alpha,
            device=device)

        self.reset_parameters()

        self.to(device)

    def forward(self, input: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]):

        preds, hidden = self._pfgru(input, hidden)
        # preds have shape (num_particles, seq_len, batch, output_size)

        # calculate mean and variance of particles
        # Paper's code used sum over particles instead of mean.
        # However, this was before activation and linear layers.
        # Also, when we train on individual predictions the sum of outputs is scaled wrongly
        # TODO maybe with elbo loss we can use the sum?
        # output_sum = torch.sum(preds, dim=0)
        output_var, output_mean = torch.var_mean(preds, dim=0, unbiased=False)

        return output_mean, output_var, preds, hidden

    def reset_parameters(self):
        for layer in self.modules():
            if isinstance(layer, torch.nn.Linear):
                # reset weights *and* biases
                layer.reset_parameters()
            elif isinstance(layer, torch.nn.BatchNorm1d):
                # reset weights *and* biases
                layer.reset_parameters()

    def init_hidden(self, batch_size=None):
        # use random initial values for more diversity
        # func = torch.zeros
        func = torch.rand
        if batch_size is None:
            batch_size = 1
        h0 = func(
            (self._num_layers, batch_size * self._num_particles, self._hidden_size), 
            device=self._device)
        p0 = np.log(1 / self._num_particles) * torch.ones(
            (self._num_layers, batch_size * self._num_particles, 1),
            device=self._device)
        hidden = (h0, p0)

        return hidden
