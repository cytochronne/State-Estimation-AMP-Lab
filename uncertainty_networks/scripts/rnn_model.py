
from uncertainty_networks import UncertaintyNetwork
import matplotlib.pyplot as plt
import numpy as np
import torch

# reproducibility
torch.manual_seed(0)
np.random.seed(0)


def train(X_train, y_train, model, epochs, device, shuffle):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    X_train = torch.Tensor(X_train).to(device)
    y_train = torch.Tensor(y_train).to(device)
    batch_size = len(X_train)
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, y_train),
        shuffle=shuffle,
        batch_size=batch_size)
    model.train()
    model.to(device)

    for epoch in range(epochs):
        epoch_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()

            x, y = batch
            # format to shape (seq, batch, ...)
            x = np.swapaxes(x, 0, 1)
            y = np.swapaxes(y, 0, 1)
            hidden = model.init_hidden(batch_size)
            preds, _ = model(x, hidden)

            # loss on each individual prediction
            # preds have shape (num_models*num_passes, output_size)
            # y has shape (output_size)
            # pytorch automatically applies broadcasting in the subtraction
            loss = 0.5*torch.mean(torch.square(preds - y))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch % 1000 == 0 or epoch == epochs - 1:
            print("Epoch: {}, Loss: {:.4f}".format(epoch, epoch_loss))


def test(X_test, y_test, model, device):
    loss_fn = torch.nn.MSELoss()
    with torch.no_grad():
        X_test = torch.Tensor(X_test).to(device)
        y_test = torch.Tensor(y_test).to(device)
        # add batch dimension
        X_test = X_test[..., None, :]
        y_test = y_test[..., None, :]
        hidden = model.init_hidden(1)
        model.eval()
        model.to(device)
        preds, _ = model(X_test, hidden)
        # compute model uncertainty from predictions
        var, mean = torch.var_mean(preds, dim=0, unbiased=False)
        loss = (mean - y_test).square().mean()
        return (
            mean.to("cpu").numpy(),
            var.to("cpu").numpy(),
            preds.to("cpu").numpy(),
            loss.to("cpu").numpy())


# Training
# parameters
epochs = 10
seq_length = 100
# input sizes of input encoder, gru and latent encoder
input_size = [1, 3, 3]
# hidden sizes of input encoder, gru and latent encoder
hidden_size = [[4, 4], 3, [4, 4]]
# output sizes of input encoder, and latent encoder
output_size = [3, 1]
num_layers = 2
num_std_plot = 2
# TODO
device = "cpu"
shuffle = True
use_jit = True

# dynamics of pendulum with dampening
G = 9.81
MASS = 1.0
POLE_LENGTH = 1.0
MUE = 0.01
VEL_0 = 0
ANG_0 = 1
# dynamics equation:
# m*l**2*acc = -mu*vel + m*g*l*sin(angle)
C1 = -MUE/(MASS*POLE_LENGTH**2)
C2 = G/(POLE_LENGTH)
T, DT = np.linspace(0, 15, 500, retstep=True)
vel = np.ones_like(T)*VEL_0
ang = np.ones_like(T)*ANG_0
def step(velocity, angle):
    acceleration = (C1*velocity + C2*np.sin(angle))
    velocity = velocity + DT*acceleration
    angle = angle + DT*velocity + 0.5*DT**2*acceleration
    return velocity, angle
for i in range(1, T.shape[0]):
    vel[i], ang[i] = step(vel[i-1], ang[i-1])

# generate training dataset
train_indices = np.index_exp[:T.shape[0]//3*2]
test_indices = np.index_exp[T.shape[0]//3*2:]
X_train = np.concatenate([ang[i] for i in train_indices])
y_train = np.concatenate([vel[i] for i in train_indices])
# generate sequences
indexer = np.arange(seq_length)[None, ...] + np.arange(X_train.shape[0] + 1 - seq_length)[..., None]
# dimension (batch, seq_length, input_size)
X_train = X_train[indexer][..., None]
y_train = y_train[indexer][..., None]
# generate training dataset with sequence length of 1
# dimension (batch, 1, input_size)
# repeat dataset by sequence length to have a dataset of similar number of samples
X_train_seq = np.repeat(np.reshape(ang, (-1, 1, 1)), 10, axis=0)
y_train_seq = np.repeat(np.reshape(vel, (-1, 1, 1)), 10, axis=0)
# generate test dataset
# one big sequence (seq_length, input_size)
X_test = ang[..., None]
y_test = vel[..., None]

# train deterministic baseline with sequence length of 1
rnn_seq = UncertaintyNetwork(
    input_size=input_size,
    hidden_size=hidden_size,
    output_size=output_size,
    num_layers=num_layers,
    dropout_prob=0,
    num_passes=10,
    num_models=1,
    initialization="sl",
    device=device)
rnn_seq = torch.jit.script(rnn_seq) if use_jit else rnn_seq
train(X_train_seq, y_train_seq, rnn_seq, epochs, device, shuffle)
mean_seq, var_seq, preds_seq, loss_seq = test(X_test, y_test, rnn_seq, device)

# train deterministic baseline
rnn_0 = UncertaintyNetwork(
    input_size=input_size,
    hidden_size=hidden_size,
    output_size=output_size,
    num_layers=num_layers,
    dropout_prob=0,
    num_passes=10,
    num_models=1,
    initialization="sl",
    device=device)
rnn_0 = torch.jit.script(rnn_0) if use_jit else rnn_0
train(X_train, y_train, rnn_0, epochs, device, shuffle)
mean_0, var_0, preds_0, loss_0 = test(X_test, y_test, rnn_0, device)

# train dropout
rnn_1 = UncertaintyNetwork(
    input_size=input_size,
    hidden_size=hidden_size,
    output_size=output_size,
    num_layers=num_layers,
    dropout_prob=0.05,
    num_passes=10,
    num_models=1,
    initialization="sl",
    device=device)
rnn_1 = torch.jit.script(rnn_1) if use_jit else rnn_1
train(X_train, y_train, rnn_1, epochs, device, shuffle)
mean_1, var_1, preds_1, loss_1 = test(X_test, y_test, rnn_1, device)

# train ensemble
rnn_2 = UncertaintyNetwork(
    input_size=input_size,
    hidden_size=hidden_size,
    output_size=output_size,
    num_layers=num_layers,
    dropout_prob=0,
    num_passes=1,
    num_models=10,
    initialization="sl",
    device=device)
rnn_2 = torch.jit.script(rnn_2) if use_jit else rnn_2
train(X_train, y_train, rnn_2, epochs, device, shuffle)
mean_2, var_2, preds_2, loss_2 = test(X_test, y_test, rnn_2, device)

# train ensemble with dropout
rnn_3 = UncertaintyNetwork(
    input_size=input_size,
    hidden_size=hidden_size,
    output_size=output_size,
    num_layers=num_layers,
    dropout_prob=0.05,
    num_passes=2,
    num_models=5,
    initialization="sl",
    device=device)
rnn_3 = torch.jit.script(rnn_3) if use_jit else rnn_3
train(X_train, y_train, rnn_3, epochs, device, shuffle)
mean_3, var_3, preds_3, loss_3 = test(X_test, y_test, rnn_3, device)

# train ensemble with dropout
rnn_4 = UncertaintyNetwork(
    input_size=input_size,
    hidden_size=hidden_size,
    output_size=output_size,
    num_layers=num_layers,
    dropout_prob=0.05,
    num_passes=5,
    num_models=2,
    initialization="sl",
    device=device)
rnn_4 = torch.jit.script(rnn_4) if use_jit else rnn_4
train(X_train, y_train, rnn_4, epochs, device, shuffle)
mean_4, var_4, preds_4, loss_4 = test(X_test, y_test, rnn_4, device)

# Plotting

# dataset
fig = plt.figure(dpi=300, constrained_layout=True)
gs = fig.add_gridspec(2, 2)
axs = [fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]),fig.add_subplot(gs[1, 1])]
axs[0].set_title("Ground Truth Data", loc="left")
axs[0].grid()
axs[0].set_xlabel("Angle (Input)")
axs[0].set_ylabel("Velocity (Output)")
for i in train_indices:
    axs[0].plot(ang[i], vel[i], color="tab:blue")
for i in test_indices:
    axs[0].plot(ang[i], vel[i], color="tab:orange")
axs[1].grid()
axs[1].set_ylabel("Angle (Input)")
for i in train_indices:
    axs[1].plot(T[i], ang[i], color="tab:blue")
for i in test_indices:
    axs[1].plot(T[i], ang[i], color="tab:orange")
axs[2].grid()
axs[2].set_xlabel("Time (Sequence Length)")
axs[2].set_ylabel("Velocity (Output)")
for i in train_indices:
    axs[2].plot(T[i], vel[i], color="tab:blue")
for i in test_indices:
    axs[2].plot(T[i], vel[i], color="tab:orange")
fig.savefig("rnn_model_dataset.png")

# baselines
fig = plt.figure(dpi=300, constrained_layout=True)
axs = fig.subplots(2)
axs[0].set_title("Deterministic Baseline\nTest Loss = {:.4}".format(loss_0), loc="left")
# plot function
for ax in axs.flatten():
    ax.grid()
    for i in train_indices:
        ax.plot(T[i], vel[i], color="tab:blue", linestyle="--")
    for i in test_indices:
        ax.plot(T[i], vel[i], color="tab:orange", linestyle="--")
axs[0].plot(T, mean_0.flatten(), color="tab:red")
axs[1].set_title("Deterministic Baseline with seq_len = 1\nTest Loss = {:.4}".format(loss_seq), loc="left")
axs[1].plot(T, mean_seq.flatten(), color="tab:red")
fig.savefig("rnn_model_baseline.png")

# models
fig = plt.figure(dpi=300, figsize=(12, 9), constrained_layout=True)
axs = np.array(fig.subplots(4, 2))
mean_plot = np.array([mean_1.flatten(), mean_2.flatten(), mean_3.flatten(), mean_4.flatten()])
std_plot = num_std_plot*np.sqrt(np.array([var_1.flatten(), var_2.flatten(), var_3.flatten(), var_4.flatten()]))
pred_plot = [preds_1, preds_2, preds_3, preds_4]
# plot function
for ax in axs.flatten():
    ax.grid()
    for i in train_indices:
        ax.plot(T[i], vel[i], color="tab:blue", linestyle="--")
    for i in test_indices:
        ax.plot(T[i], vel[i], color="tab:orange", linestyle="--")
# titles
axs[0, 0].set_title("MC Dropout (10 passes)\nTest Loss = {:.4}".format(loss_1), loc="left")
axs[1, 0].set_title("Ensemble (10 models)\nTest Loss = {:.4}".format(loss_2), loc="left")
axs[2, 0].set_title("Ensemble (5 models) MC Dropout (2 passes)" "\nTest Loss = {:.4}".format(loss_3), loc="left")
axs[3, 0].set_title("Ensemble (2 models) MC Dropout (5 passes)" "\nTest Loss = {:.4}".format(loss_4), loc="left")
# axs[2, 0].set_title("Patricle Filter GRU (5 particles)" "\nTest Loss = {:.4}".format(loss_3), loc="left")
# plot mean and std
axs[0, 0].set_title(r"Final Predictions ({}$\sigma$)".format(num_std_plot))
for i, ax in enumerate(axs[:, 0]):
    ax.plot(T, mean_plot[i], color="tab:red")
    ax.fill_between(T, mean_plot[i] - std_plot[i], mean_plot[i] + std_plot[i], alpha=0.2, color="tab:grey")
# plot predictions
axs[0, 1].set_title("Individual Predictions")
for i, ax in enumerate(axs[:, 1]):
    for pred in pred_plot[i]:
        ax.plot(T, pred.flatten(), color="tab:red", alpha=np.maximum(0.2, 1/len(pred_plot[i])))
fig.savefig("rnn_model.png")
