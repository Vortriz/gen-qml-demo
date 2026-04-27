import itertools
import numpy as np
import torch
import tensorcircuit as tc
import ot

import marimo as mo
from scipy.stats import unitary_group
from numpy import pi
from enum import Enum, auto
from typing import no_type_check
from torch import nn
import qutip
import matplotlib.pyplot as plt

tc.set_dtype("complex128")
K = tc.set_backend("pytorch")

class LossFunction(Enum):
    MMD = auto()
    WASSERSTEIN = auto()

def mmd_distance(ensemble1, ensemble2):
    r11 = 1.0 - torch.mean(torch.abs(torch.einsum("mi,ni->mn", ensemble1.conj(), ensemble1)).pow(2.0))
    r22 = 1.0 - torch.mean(torch.abs(torch.einsum("mi,ni->mn", ensemble2.conj(), ensemble2)).pow(2.0))
    r12 = 1.0 - torch.mean(torch.abs(torch.einsum("mi,ni->mn", ensemble1.conj(), ensemble2)).pow(2.0))
    return 2 * r12 - r11 - r22

def wasserstein_distance(ensemble1, ensemble2):
    D = 1.0 - torch.abs(ensemble1.conj() @ ensemble2.T).pow(2.0)
    emt = torch.empty(0)
    return ot.emd2(emt, emt, M=D)

class ForwardProcess:
    def __init__(self, n_qubits, ensemble, T, schedule):
        self.n_qubits = n_qubits
        self.initial_ensemble = ensemble
        self.T = T
        self.schedule = schedule
        self.ensembles = None

    def scramble(self, rng, progress_bar=None):
        size = self.initial_ensemble.shape[0]
        self.ensembles = torch.zeros((self.T + 1, size, 2**self.n_qubits), dtype=torch.complex128)
        self.ensembles[0] = self.initial_ensemble

        for t in range(self.T):
            if progress_bar:
                progress_bar.update(1)
            for s in range(size):
                c = tc.Circuit(self.n_qubits, inputs=self.ensembles[t, s])
                all_qubits = range(self.n_qubits)
                weight = self.schedule[t]

                c.rx(all_qubits, theta=weight * (rng.random(self.n_qubits) * pi / 4 - pi / 8))
                c.ry(all_qubits, theta=weight * (rng.random(self.n_qubits) * pi / 4 - pi / 8))
                c.rx(all_qubits, theta=weight * (rng.random(self.n_qubits) * pi / 4 - pi / 8))

                if self.n_qubits > 1:
                    combinations = list(itertools.combinations(all_qubits, 2))
                    angles = rng.random(len(combinations)) * 0.2 + 0.4
                    for p, (i, j) in enumerate(combinations):
                        c.rzz(i, j, theta=(weight * angles[p]) / (2 * np.sqrt(self.n_qubits)))

                self.ensembles[t + 1, s] = torch.as_tensor(c.state(), dtype=torch.complex128)
        return self.ensembles

class BackwardProcess(nn.Module):
    def __init__(self, n_qubits, n_ancilla, T, n_layers, ensemble_size, epochs, loss_func, target_mode="previous", lr=0.001):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_ancilla = n_ancilla
        self.n_total = n_qubits + n_ancilla
        self.T = T
        self.n_layers = n_layers
        self.batch_size = 200
        self.epochs = epochs
        self.loss_func = loss_func
        self.target_mode = target_mode
        self.lr = lr

        self._back_circuit_vmap = K.vmap(self._back_circuit, vectorized_argnums=0)
        self.trained_params = None
        self.loss_history = None

    @no_type_check
    def _back_circuit(self, input_state, params):
        all_qubits = range(self.n_total)
        c = tc.Circuit(self.n_total, inputs=input_state)
        for i in range(self.n_layers):
            layer_params = params[i]
            c.rx(all_qubits, theta=layer_params[:self.n_total])
            c.ry(all_qubits, theta=layer_params[self.n_total: 2*self.n_total])
            c.cz(list(all_qubits)[:-1:2], list(all_qubits)[1::2])
            if self.n_total > 2:
                c.cz(list(all_qubits)[1::2], list(all_qubits)[2::2])
        return c.state()

    def _measure(self, inputs):
        m_probs = (torch.abs(inputs.reshape(inputs.shape[0], 2**self.n_ancilla, 2**self.n_qubits))**2).sum(dim=2)
        m_res = torch.multinomial(m_probs, num_samples=1).squeeze()
        indices = 2**self.n_qubits * m_res.view(-1, 1) + torch.arange(2**self.n_qubits)
        post_state = torch.gather(inputs, 1, indices)
        norms = torch.sqrt(torch.sum(torch.abs(post_state)**2, dim=1)).unsqueeze(dim=1)
        return (1.0 / norms) * post_state

    def _denoise_step(self, input_states, params):
        input_with_ancilla = torch.concatenate((
            input_states,
            torch.zeros((input_states.shape[0], 2**self.n_total - 2**self.n_qubits), dtype=torch.complex128)
        ), dim=1)
        output = self._back_circuit_vmap(input_with_ancilla, params)
        return self._measure(output)

    def train(self, forward_ensembles, rng):
        self.loss_history = np.zeros((self.T, self.epochs))
        self.trained_params = torch.zeros((self.T, self.n_layers, 2*self.n_total), dtype=torch.float64)

        current_states = torch.from_numpy(unitary_group.rvs(dim=2**self.n_qubits, size=self.batch_size, random_state=rng)[:, :, 0])

        loss_call = mmd_distance if self.loss_func == LossFunction.MMD else wasserstein_distance

        with mo.status.progress_bar(
            total=self.T, title="Training...", completion_title="Training complete!",
        ) as training_bar:
            for t in range(self.T - 1, -1, -1):
                params = torch.tensor(rng.normal(size=(self.n_layers, 2*self.n_total)), requires_grad=True, dtype=torch.float64)
                optimizer = torch.optim.Adam([params], lr=self.lr)

                target_idx = 0 if self.target_mode == "initial" else t

                for epoch in mo.status.progress_bar(range(self.epochs), title=f"Progress for time step {t+1}", remove_on_exit=True):
                    indices = rng.choice(forward_ensembles.shape[1], size=self.batch_size, replace=False)
                    true_data = forward_ensembles[target_idx, indices]

                    output = self._denoise_step(current_states, params)
                    loss = loss_call(output, true_data)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    self.loss_history[t, epoch] = loss.item()

                with torch.no_grad():
                    self.trained_params[t] = params.detach()
                    current_states = self._denoise_step(current_states, self.trained_params[t])

                if training_bar:
                    training_bar.update(1)

    def run(self, rng, size):
        states_history = torch.zeros((self.T + 1, size, 2**self.n_qubits), dtype=torch.complex128)
        states_history[self.T] = torch.from_numpy(unitary_group.rvs(dim=2**self.n_qubits, size=size, random_state=rng)[:, :, 0])

        for t in range(self.T - 1, -1, -1):
            with torch.no_grad():
                states_history[t] = self._denoise_step(states_history[t+1], self.trained_params[t])
        return states_history

def generate_initial_ensemble(n_qubits, type, size, scale, rng):
    base_state = rng.random(2**n_qubits)
    states = np.zeros((size, 2**n_qubits), dtype=np.complex128)

    if type == "clustered":
        for i in range(size):
            perturbation = rng.standard_normal(2**n_qubits) + 1j * rng.standard_normal(2**n_qubits)
            states[i] = base_state + scale * perturbation
    elif type == "circle":
        phis = rng.uniform(0, 2 * np.pi, size)
        states = np.vstack((np.cos(phis), np.sin(phis))).T

    normalized_states = states / np.linalg.norm(states, axis=1, keepdims=True)
    return torch.as_tensor(normalized_states, dtype=torch.complex128)

def plot_bloch_sphere(ensemble):
    if ensemble.shape[1] != 2:
        return mo.md("Bloch sphere visualization is only available for single-qubit states.")

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(azim=-40, elev=30, projection="3d")
    sphere = qutip.Bloch(axes=ax)

    points = []
    for i in range(len(ensemble)):
        state = ensemble[i][0] * qutip.basis(2, 0) + ensemble[i][1] * qutip.basis(2, 1)
        points.append([
            qutip.expect(qutip.sigmax(), state),
            qutip.expect(qutip.sigmay(), state),
            qutip.expect(qutip.sigmaz(), state),
        ])

    points_transposed = np.array(points).T.tolist()
    sphere.add_points(points_transposed)
    sphere.point_size = [2]
    sphere.render()
    plt.close(fig)
    return fig
