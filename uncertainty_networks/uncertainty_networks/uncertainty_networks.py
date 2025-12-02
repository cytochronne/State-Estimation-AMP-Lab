
from uncertainty_networks.pfrnn import PFGRU

import copy
import numpy as np
import torch
from torch import Tensor
from typing import Literal, Tuple, Sequence


class MonteCarloDropout(torch.nn.modules.dropout._DropoutNd):
    def __init__(self, p: float):
        super().__init__(p, False)

    def forward(self, input: Tensor) -> Tensor:
        # always have dropout enabled
        return torch.nn.functional.dropout(input, self.p, True, False)


class UncertaintyMLP(torch.nn.Module):
    """
    References:
    - "Dropout as a Bayesian Approximation" https://arxiv.org/abs/1506.02142
    - "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles" https://arxiv.org/abs/1612.01474
    - https://xuwd11.github.io/Dropout_Tutorial_in_PyTorch/#5-dropout-as-bayesian-approximation

    Implementation notes:
    - with MC Dropout network should be larger than without
    - weights of ensemble models should be initialized with different values
    - loss should be applied to individual predictions of the models
    - order of training data should be randomized
    - weight regularization did not help
    - MC Dropout paper used model precision derived from data, which was omitted here
    - Deep ensembles paper outputted parameters of gaussian to form GMM, which was omitted here
    """

    def __init__(
            self,
            input_size: int,
            hidden_sizes: Sequence[int],
            output_size: int,
            dropout_prob: float,
            num_passes: int,
            num_models: int,
            initialization: Literal["rl", "sl"],
            activation: torch.nn.Module,
            device: str):

        super().__init__()

        self._output_size = output_size
        self._num_passes = num_passes
        self._num_models = num_models
        self._initialization = initialization
        self._device = device

        assert initialization in ["rl", "sl"]

        # create model
        model = []
        for hidden_size in hidden_sizes:
            model.append(torch.nn.Linear(input_size, hidden_size))
            model.append(activation())
            if dropout_prob > 0:
                model.append(MonteCarloDropout(dropout_prob))
            input_size = hidden_size
        model.append(torch.nn.Linear(input_size, self._output_size))
        model = torch.nn.Sequential(*model)

        # create ensemble
        self._models = torch.nn.ModuleList()
        for _ in range(self._num_models):
            self._models.append(copy.deepcopy(model))

        # re-initialize parameters to ensure diversity in each model of the ensemble
        self.reset_parameters()

        self.to(device)

    @torch.jit.ignore
    @torch.jit.export
    def reset_parameters(self) -> None:
        for layer in self.modules():
            if not isinstance(layer, torch.nn.Linear):
                continue
            if self._initialization == "rl":
                # use orthogonal initialization for weights
                torch.nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                # set biases to zero
                torch.nn.init.zeros_(layer.bias)
                # as described in: 
                # https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
            else:
                # use default layer initialization
                layer.reset_parameters()

    @torch.jit.export
    def forward(self, input: Tensor, shared_input: bool=True) -> Tensor:
        # input shape:
        #   (batch, input_size)                         when shared_input is True
        #   (num_models*num_passes, batch, input_size)  when shared_input is False
        # batch dimension can be composed of several dimensions, e.g. (sequence, batch) for RNNs.

        # if shared_input is False, then run inference on individual predictions
        # can be used to propagate individual predictions through many modules
        if shared_input:
            batch_dims = input.shape[:-1]
        else:
            batch_dims = input.shape[1:-1]
            assert input.shape[0] == self._num_models*self._num_passes

        output = []
        # iterate over ensemble models
        for i, model in enumerate(self._models):
            # iterate over passes of single MC dropout model
            for j in range(self._num_passes):
                if shared_input:
                    model_input = input
                else:
                    model_input = input[i*self._num_passes + j]

                output.append(model(model_input))

        output = torch.stack(output, dim=0)
        assert output.shape == (self._num_models*self._num_passes,) + batch_dims + (self._output_size,)

        return output


class UncertaintyGRU(torch.nn.Module):
    """
    References:
    - "Dropout as a Bayesian Approximation" https://arxiv.org/abs/1506.02142
    - "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles" https://arxiv.org/abs/1612.01474
    - https://xuwd11.github.io/Dropout_Tutorial_in_PyTorch/#5-dropout-as-bayesian-approximation

    Implementation notes:
    (see UncertaintyMLP module)
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            dropout_prob: float,
            num_passes: int,
            num_models: int,
            initialization: Literal["rl", "sl"],
            device: str):

        super().__init__()

        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._num_passes = num_passes
        self._num_models = num_models
        self._initialization = initialization
        self._device = device

        assert initialization in ["rl", "sl"]

        # create model
        gru = torch.nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_prob)

        # create ensemble
        self._models = torch.nn.ModuleList()
        for _ in range(self._num_models):
            self._models.append(copy.deepcopy(gru))

        # re-initialize parameters to ensure diversity in each model of the ensemble
        self.reset_parameters()

        self.to(device)

    @torch.jit.ignore
    @torch.jit.export
    def reset_parameters(self) -> None:
        for module in self.modules():
            if not isinstance(module, torch.nn.GRU):
                continue
            if self._initialization == "rl":
                for name, param in module.named_parameters():
                    if "weight" in name:
                        torch.nn.init.orthogonal_(param, 1.0)
                    elif "bias" in name:
                        torch.nn.init.zeros_(param)
            else:
                # use default layer initialization
                module.reset_parameters()

    @torch.jit.export
    def forward(self, input: Tensor, hidden: Tensor, shared_input: bool=True) -> Tuple[Tensor, Tensor]:
        # input shape:
        # (seq_len, batch, input_size)                         when shared_input is True
        # (num_models*num_passes, seq_len, batch, input_size)  when shared_input is False
        # hidden shape:
        # (num_models*num_passes, num_layers, batch_size, hidden_size)

        # If shared_input is False, then run inference on individual predictions
        # Can be used to propagate individual predictions through many modules
        if shared_input:
            assert input.ndim == 3
        else:
            assert input.ndim == 4
            assert input.shape[0] == self._num_models*self._num_passes

        output = []
        hidden_output = []
        # iterate over Ensemble models
        for i, model in enumerate(self._models):
            # iterate over passes of single MC Dropout model
            for j in range(self._num_passes):
                if shared_input:
                    model_input = input
                else:
                    model_input = input[i*self._num_passes + j]

                res = model(model_input, hidden[i*self._num_passes + j])
                output.append(res[0])
                hidden_output.append(res[1])

        output = torch.stack(output, dim=0)
        hidden_output = torch.stack(hidden_output, dim=0)

        assert output.shape == (self._num_models*self._num_passes,) + input.shape[-3:-1] + (self._hidden_size,)
        assert hidden_output.shape == hidden.shape

        return output, hidden_output

    @torch.jit.export
    def init_hidden(self, batch_size: int) -> Tensor:

        shape = (self._num_models*self._num_passes, self._num_layers, batch_size, self._hidden_size)

        # use random initial values for more diversity in uncertainty
        # some sources claim zeros is better
        # (https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
        init_func = torch.zeros
        # init_func = torch.rand
        hidden = init_func(shape, device=self._device)
        return hidden


class UncertaintyNetwork(torch.nn.Module):
    """
    Uncertainty module consisting of MLP + GRU + MLP for convenience.
    """

    def __init__(
            self,
            input_size: Tuple[int, int, int],
            hidden_size: Tuple[Sequence[int], int, Sequence[int]],
            output_size: Tuple[int, int],
            num_layers: int,
            dropout_prob: float,
            num_passes: int,
            num_models: int,
            initialization: Literal["rl", "sl"],
            device: str):

        super().__init__()

        activation = torch.nn.LeakyReLU

        self.mlp1 = UncertaintyMLP(
            input_size=input_size[0],
            hidden_sizes=hidden_size[0],
            output_size=output_size[0],
            dropout_prob=dropout_prob,
            num_passes=num_passes,
            num_models=num_models,
            initialization=initialization,
            activation=activation,
            device=device)

        self.rnn = UncertaintyGRU(
            input_size=input_size[1],
            hidden_size=hidden_size[1],
            num_layers=num_layers,
            dropout_prob=dropout_prob,
            num_passes=num_passes,
            num_models=num_models,
            initialization=initialization,
            device=device)

        self.mlp2 = UncertaintyMLP(
            input_size=input_size[2],
            hidden_sizes=hidden_size[2],
            output_size=output_size[1],
            dropout_prob=dropout_prob,
            num_passes=num_passes,
            num_models=num_models,
            initialization=initialization,
            activation=activation,
            device=device)

    @torch.jit.export
    def reset_parameters(self) -> None:
        self.mlp1.reset_parameters()
        self.rnn.reset_parameters()
        self.mlp2.reset_parameters()

    @torch.jit.export
    def forward(self, input: Tensor, hidden: Tensor) -> Tuple[Tensor, Tensor]:

        # include sequence dimension if not present
        added_seq_dim = False
        if input.ndim == 2:
            input = input[None, ...]
            added_seq_dim = True
        assert input.ndim == 3
        assert input.shape[-2] == hidden.shape[-2]

        preds = self.mlp1(input)
        preds, hidden = self.rnn(preds, hidden=hidden, shared_input=False)
        preds = self.mlp2(preds, shared_input=False)

        # remove sequence length dimension only if it was not present
        if added_seq_dim:
            preds = torch.squeeze(preds, dim=-3)

        return preds, hidden

    @torch.jit.export
    def init_hidden(self, batch_size: int) -> Tensor:
        return self.rnn.init_hidden(batch_size)
