# ------------------------
# Imports
# ------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim


# ------------------------
# Modelo simples (seno)
# ------------------------
class DataGenerator:
    def __init__(self, func, x_start=0, x_end=4*np.pi, num_points=1024):
        self.func = func
        self.x_start = x_start
        self.x_end = x_end
        self.num_points = num_points
    
    def training_set(self):
        # gera X uniformemente distribuído no intervalo
        x = np.linspace(self.x_start, self.x_end, self.num_points)
        # y real + ruído normal (gaussiano)
        y = self.func(x) + np.random.normal(0, 0.05, size=x.shape)
        return x, y
    
    def test_set(self):
        # gera X uniformemente distribuído no intervalo
        x = np.linspace(self.x_start, self.x_end, self.num_points)
        # y real, sem ruído
        y = self.func(x)
        return x, y



# ------------------------
# MLP Model
# ------------------------
class ValueModel(nn.Module):
    def __init__(self, n_input, hidden_units, hidden_layers, activation=nn.ReLU):
        super().__init__()

        layers = [nn.Linear(n_input, hidden_units), activation()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_units, hidden_units), activation()]
        layers += [nn.Linear(hidden_units, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)



# ------------------------
# Treinamento
# ------------------------
def train(model,
          x,
          y,
          dydx=None,
          differential=False,
          lam=0.1,
          lr_schedule=None,
          epochs=100,
          batch_size=256,
          verbose=True):
    # Data
    dataset = torch.utils.data.TensorDataset(x, y) if dydx is None else torch.utils.data.TensorDataset(x, y, dydx)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Architecture components
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()
    # Hyperparameters
    if differential and dydx is not None:
        lambda_j = 1.0 / torch.sqrt(torch.mean(dydx**2, dim=0, keepdim=True)).to(device)
        alpha = 1.0 / (1.0 + lam)
        beta = 1.0 - alpha
    else:
        lambda_j = None
        alpha = 1.0
        beta = 0.0
    # Histórico de losses
    history = {
        "total": [],
        "value_loss": [],
        "deriv_loss": []
    }
    # Loop de treino
    for epoch in range(epochs):
        epoch_total_loss = 0.0
        epoch_value_loss = 0.0
        epoch_deriv_loss = 0.0

        for batch in loader:
            optimizer.zero_grad()
            if differential:
                x_b, y_b, dydx_b = batch
                x_b = x_b.to(device)
                y_b = y_b.to(device)
                dydx_b = dydx_b.to(device)

                x_b.requires_grad_(True)
                y_pred = model(x_b)
                dydx_pred = torch.autograd.grad(
                    y_pred, x_b,
                    grad_outputs=torch.ones_like(y_pred),
                    create_graph=True
                )[0]
                                
                loss = loss_fn(y_pred, y_b)
                diff_loss = loss_fn(dydx_pred, dydx_b)
                # diff_loss = loss_fn(dydx_pred * lambda_j.to(device), dydx_b * lambda_j.to(device))
                loss = alpha * loss + beta * diff_loss

                epoch_value_loss += loss.item()
                epoch_deriv_loss += diff_loss.item()
            else:
                x_b, y_b = batch
                x_b = x_b.to(device)
                y_b = y_b.to(device)
                y_pred = model(x_b)
                loss = loss_fn(y_pred, y_b)
                epoch_value_loss += loss.item()
                epoch_deriv_loss = 0.0  # Não aplicável

            epoch_total_loss += loss.item()
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Armazenar histórico
        history["total"].append(epoch_total_loss)
        history["value_loss"].append(epoch_value_loss)
        history["deriv_loss"].append(epoch_deriv_loss)

        # Verbose opcional
        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch+1}/{epochs} | Total: {epoch_total_loss:.4f} | y_loss: {epoch_value_loss:.4f} | dydx_loss: {epoch_deriv_loss:.4f}")

    # Plot final se verbose
    if verbose:
        plt.plot(history["total"], label="Total Loss")
        plt.plot(history["value_loss"], label="Value Loss (y)")
        if differential:
            plt.plot(history["deriv_loss"], label="Derivative Loss (dydx)")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss over epochs")
        plt.grid(True)
        plt.show()

    return history


# ------------------------
# Aproximador PyTorch
# ------------------------
class NeuralApproximator:
    def __init__(self,
                 x_raw,
                 y_raw,
                 dydx_raw):
        self.x_raw = x_raw
        self.y_raw = y_raw
        self.dydx_raw = dydx_raw
        self.model = None

        if self.x_raw.ndim == 1:
            self.x_raw = self.x_raw.reshape(-1, 1)
        if self.y_raw.ndim == 1:
            self.y_raw = self.y_raw.reshape(-1, 1)
        if self.dydx_raw is not None and self.dydx_raw.ndim == 1:
            self.dydx_raw = self.dydx_raw.reshape(-1, 1)
        self.model = None

    # Prepare data
    def prepare(self,
                m,
                differential,
                lam=10.0,
                hidden_units=20,
                hidden_layers=4):
        # Just cut first m data (batching)
        if m is not None:
            x = self.x_raw[:m]
            y = self.y_raw[:m]
            dydx = self.dydx_raw[:m] if self.dydx_raw is not None else None
        else:
            x = self.x_raw
            y = self.y_raw
            dydx = self.dydx_raw
        # Convert to tensor float32
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.dydx = torch.tensor(dydx, dtype=torch.float32) if dydx is not None else None

        self.model = ValueModel(self.x.shape[1], hidden_units, hidden_layers, activation=nn.ELU) # ReLU, LeakyReLU, GELU
        self.differential = differential
        self.lam = lam

    # Train method
    def train(self,
              epochs=100):
        train(self.model, self.x, self.y, self.dydx if self.differential else None,
              differential=self.differential, lam=self.lam, epochs=epochs, verbose=True)

    # Test method
    def predict_values_and_derivs(self,
                                  x_in):
        x_tensor = torch.tensor(x_in, dtype=torch.float32, requires_grad=True)        
        y_pred = self.model(x_tensor)
        dydx_pred = torch.autograd.grad(y_pred,
                                        x_tensor,
                                        grad_outputs=torch.ones_like(y_pred),
                                        create_graph=False)[0]
        return y_pred.detach().numpy(), dydx_pred.detach().numpy()


# ------------------------
# Plot dos resultados
# ------------------------
def graph(title, 
          predictions, 
          xAxis, 
          targets, 
          sizes,
          computeRmse=False,
          prefix='bs_'):
    
    numRows = len(sizes)
    numCols = 2  # standard e differential

    fig, ax = plt.subplots(numRows, numCols, squeeze=False)
    fig.set_size_inches(3.5 * numCols + 1.5, 3.5 * numRows)

    for i, size in enumerate(sizes):
        ax[i, 0].annotate(f"size {size}", xy=(0, 0.5), 
                          xytext=(-ax[i, 0].yaxis.labelpad - 5, 0),
                          xycoords=ax[i, 0].yaxis.label, textcoords='offset points',
                          ha='right', va='center')

    ax[0, 0].set_title(f"standard")
    ax[0, 1].set_title(f"differential")

    for i, size in enumerate(sizes):        
        for j, regType in enumerate(["standard", "differential"]):
            key = (prefix + regType, size)
            x_vals = xAxis[key]
            y_preds = predictions[key]
            y_true = targets[key]

            if computeRmse:
                errors = y_preds - y_true
                rmse = np.sqrt((errors ** 2).mean(axis=0))
                xlabel_text = f"rmse: {rmse:.2f}"

            ax[i, j].set_xlabel(xlabel_text)            

            # predicted (em ciano com círculo branco) e targets (vermelho)
            ax[i, j].plot(x_vals, y_true * 100, 'r.', 
                          markersize=0.5, label='targets')
            ax[i, j].plot(x_vals, y_preds * 100, 'co', 
                          markersize=2, markerfacecolor='white', label="predicted")

            ax[i, j].legend(prop={'size': 8}, loc='upper left')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.suptitle(f"{title}", fontsize=16)
    plt.show()


# ------------------------------------------------
# ------------------------------------------------
# INSTANCIAÇÃO
# ------------------------------------------------
# ------------------------------------------------
if __name__ == "__main__":

    # 1. Parâmetros
    sizes = [512, 4096]  # tamanhos dos datasets
    differential_modes = [False, True]  # sem e com derivada
    noise_std = 0.1

    # 2. Função e sua derivada (para diferencial)
    f = lambda x: np.sin(x)
    df = lambda x: np.cos(x)
    predictions = {}
    targets = {}
    xAxis = {}

    # 3. Gerar dados e treinar a rede neural
    for size in sizes:
        # Instancia gerador de dados seno
        gen = DataGenerator(func=f, x_start=0, x_end=4*math.pi, num_points=size)
        
        for differential in differential_modes:
            x_train, y_train = gen.training_set()
            dydx_train = df(x_train) if differential else None

            # Preparar tensores para a rede
            x_train_t = torch.tensor(x_train, dtype=torch.float32).reshape(-1, 1)
            y_train_t = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
            dydx_train_t = torch.tensor(dydx_train, dtype=torch.float32).reshape(-1, 1) if differential else None

            # Criar e treinar a rede neural
            nn_model = NeuralApproximator(x_train_t.numpy(), y_train_t.numpy(), dydx_train_t.numpy() if differential else None)
            nn_model.prepare(m=size, differential=differential, hidden_units=50, hidden_layers=3)
            nn_model.train(epochs=300)

            # Test set para plot
            x_test = np.linspace(0, 4*math.pi, 1000).reshape(-1, 1)
            y_test = f(x_test)

            # Predição no conjunto de treinamento para comparar
            y_pred, dydx_pred = nn_model.predict_values_and_derivs(x_train_t.numpy())

            # Guardar para plotar
            key = ('nn_' + ('differential' if differential else 'standard'), size)
            predictions[key] = y_pred.squeeze()
            targets[key] = y_train  # usamos o conjunto com ruído como target
            xAxis[key] = x_train

    # 4. Plotar os resultados
    graph(
        title="Approximation of y=sin(x) with NN",
        predictions=predictions,
        xAxis=xAxis,
        targets=targets,
        sizes=sizes,
        computeRmse=True,
        prefix='nn_'
    )
