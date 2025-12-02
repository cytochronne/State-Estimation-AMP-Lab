"""
Example usage of uncertainty networks for estimation of model and data uncertainties.
"""
from uncertainty_networks import UncertaintyMLP

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import BatchSampler, RandomSampler, TensorDataset


# reproducibility
torch.manual_seed(0)
np.random.seed(0)


def train(X_train, y_train, model, epochs, batch_size, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0009)
    X_train = torch.Tensor(X_train).to(device)
    y_train = torch.Tensor(y_train).to(device)
    dataset = TensorDataset(X_train, y_train)
    sampler = BatchSampler(RandomSampler(dataset), batch_size, False)
    model.train()
    model.to(device)

    for epoch in range(epochs):
        epoch_loss = 0
        for batch_indices in sampler:
            optimizer.zero_grad()
            X_batch, y_batch = dataset[batch_indices]

            output = model(X_batch)
            # split output of network into predictions and data uncertainties
            preds, data_logvar = torch.split(output, [1, 1], dim=-1)

            # enforce minimum variance of 1e-6 for numerical stability
            logvar_clip = torch.maximum(data_logvar, torch.log(torch.tensor(1e-6)))
            # squared loss weighted by variance + variance regularization
            loss = 0.5*torch.mean(torch.exp(-logvar_clip) * (preds - y_batch).square() + logvar_clip)
            loss = loss.mean()

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print("Epoch: {}, Loss: {:.4f}".format(epoch, epoch_loss))


def test(X_test, y_test, model, device):

    with torch.no_grad():
        X_test = torch.Tensor(X_test).to(device)
        y_test = torch.Tensor(y_test).to(device)
        model.eval()
        model.to(device)
        output = model(X_test)
        # split output of network into predictions and data uncertainties
        preds, data_logvar = torch.split(output, [1, 1], dim=-1)
        # final data uncertainty is the mean of all predicted data uncertainties
        data_var = torch.mean(torch.exp(data_logvar), dim=0)
        # compute model uncertainty from predictions
        model_var, mean = torch.var_mean(preds, dim=0, unbiased=False)
        var = model_var + data_var
        loss = (mean - y_test).square().mean()

        return (
            mean.to("cpu").numpy(),
            var.to("cpu").numpy(),
            model_var.to("cpu").numpy(),
            data_var.to("cpu").numpy(),
            preds.to("cpu").numpy(),
            loss.to("cpu").numpy())


# Training
# parameters
main_shape = (64, 64, 64, 64)
input_size = 1
output_size = 2  # consists of prediction size + data uncertainty size
# TODO
device = "cpu"
epochs = 50
batch_size = 10
num_std_plot = 1
use_jit = True
noise_std = 0.5

# generate dataset
X = np.linspace(-20, 20, 10000).reshape(-1, 1)
# apply disturances to data
missing_region = (np.array([[0.0, 0.1], [0.9, 1.0]])*X.shape[0]).astype(int)
noisy_region = (np.array([[0.6, 0.8]])*X.shape[0]).astype(int)
missing_mask = np.full(X.shape[0], False, dtype=bool)
for region in missing_region:
    missing_mask[region[0]:region[1]] = True
noisy_mask = np.full(X.shape[0], False, dtype=bool)
for region in noisy_region:
    noisy_mask[region[0]:region[1]] = True
def y_func(x):
    return np.sin(0.7*x+1) + 0.5*np.cos(1.2*x-2)
y = y_func(X)

X_train = X.copy()
y_train = y.copy()
y_train[noisy_mask] += noise_std*np.random.randn(*y[noisy_mask].shape)
X_train = X_train[~missing_mask]
y_train = y_train[~missing_mask]

# train Deep Ensamble with MC Dropout
model = UncertaintyMLP(
    input_size=input_size,
    hidden_sizes=main_shape,
    output_size=output_size,
    dropout_prob=0.1,
    num_passes=5,
    num_models=4,
    initialization="sl",
    activation=torch.nn.LeakyReLU,
    device=device)
model = torch.jit.script(model) if use_jit else model

train(X_train, y_train, model, epochs, batch_size, device)
mean, var, model_var, data_var, preds, loss = test(X, y, model, device)

# Plotting
stdx = num_std_plot*np.sqrt(var)
model_stdx = num_std_plot*np.sqrt(model_var)
data_stdx = num_std_plot*np.sqrt(data_var)
fig = plt.figure(dpi=300, figsize=(6, 9), constrained_layout=True)
axs = fig.subplots(6, sharex=True, sharey=True)
# ground truth
axs[0].set_title("Ground Truth Data")
# show gt on all plots
for i, ax in enumerate(axs):
    ax.grid()
    if i == 1:
        continue
    linestyle = "--" if i != 0 else None
    use_label = True if i == 0 else False
    ax.plot(np.where(np.logical_or(noisy_mask, missing_mask), np.nan, X.flatten()), np.where(np.logical_or(noisy_mask, missing_mask), np.nan, y.flatten()), color="tab:blue", linestyle=linestyle, label="Data" if use_label else None)
    ax.plot(np.where(missing_mask, X.flatten(), np.nan), np.where(missing_mask, y.flatten(), np.nan), color="tab:orange", linestyle=linestyle, label="Missing" if use_label else None)
    ax.plot(np.where(noisy_mask, X.flatten(), np.nan), np.where(noisy_mask, y.flatten(), np.nan), color="tab:green", linestyle=linestyle, label="Noisy" if use_label else None)
axs[0].legend(ncol=3, loc="lower left")
# samples
size = 1
axs[1].set_title("Training Data")
axs[1].scatter(X_train, y_train, color="tab:blue", marker=".", s=size, label="Samples")
axs[1].legend(ncol=3, loc="lower left")
# mean and total var
axs[2].set_title("Mean and Total Uncertainty")
axs[2].plot(X, mean, color="tab:red", label="Mean")
axs[2].fill_between(X.flatten(), (mean - stdx).flatten(), (mean + stdx).flatten(), alpha=0.5, color="tab:grey", label=r"$1\sigma$")
axs[2].set_ylabel("Output")
axs[2].legend(ncol=3, loc="lower left")
# mean and model var
axs[3].set_title("Mean and Model Uncertainty")
axs[3].plot(X, mean, color="tab:red")
axs[3].fill_between(X.flatten(), (mean - model_stdx).flatten(), (mean + model_stdx).flatten(), alpha=0.5, color="tab:grey")
# mean and data var
axs[4].set_title("Mean and Data Uncertainty")
axs[4].plot(X, mean, color="tab:red")
axs[4].fill_between(X.flatten(), (mean - data_stdx).flatten(), (mean + data_stdx).flatten(), alpha=0.5, color="tab:grey")
# preds
axs[5].set_title("Predictions")
for pred in preds:
    axs[5].plot(X, pred, color="tab:red", alpha=np.maximum(0.2, 1/len(preds)))
axs[-1].set_xlabel("Input")
fig.savefig("mlp_model_data.png")
