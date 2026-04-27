import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import lib
    import warnings
    from wigglystuff import PlaySlider

    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

    rng = np.random.default_rng(1234)


@app.cell(hide_code=True)
def _():
    mo.Html("""
    <h1>Generative Quantum Machine Learning</h1>
    """).style(
        {"font-size": "2.4rem", "font-weight": "700", "color": "#111827"}
    ).center()
    return


@app.cell(hide_code=True)
def _():
    mo.md(f"""
    ## Introduction

    The preparation and characterization of complex quantum states are fundamental challenges in quantum information science. While quantum computers promise to simulate many-body systems and solve optimization problems that are classically intractable, the practical utility of these devices often depends on the ability to prepare specific, high-fidelity quantum states, such as ground states of Hamiltonians or thermal states. Generative Quantum Machine Learning models have shown promise in learning and sampling from complex quantum state distributions.

    This notebook explores the generation of quantum state ensembles using Quantum DDPM, as proposed in the paper [**"Generative Quantum Machine Learning via Denoising Diffusion Probabilistic Models"**](https://www.alphaxiv.org/abs/2310.05866).

    ## The Core Idea of Quantum DDPM
    An ensemble of states is scrambled over multiple steps in the forward process and then a series of PQCs are trained to reverse this scrambling process step by step in the reverse process. This divides the task of learning a map from full noise to the target distribution into subtasks with low-depth circuits which can avoid barren plateaus.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. Defining the Ensembles

    We focus on two types of 1-qubit ensembles, as they can be nicely visualized on a Bloch sphere:
    - **Clustered Ensemble**: States localized around an arbitrary state.
    - **Circle Ensemble**: States distributed along a great circle on the Bloch sphere. This ensemble has a non-trivial topological structure that is challenging for standard generative models.
    """)
    return


@app.cell(hide_code=True)
def _():
    ensemble_type_ui = mo.ui.dropdown(
        options=["clustered", "circle"],
        value="clustered",
        label="**Ensemble Type**",
    )
    n_samples_ui = mo.ui.slider(
        start=100,
        stop=1500,
        step=100,
        value=1000,
        debounce=True,
        include_input=True,
        label="**Number of Samples**",
    )
    scale_ui = mo.ui.slider(
        start=0.01,
        stop=0.1,
        step=0.01,
        value=0.04,
        debounce=True,
        include_input=True,
        label="**Clustering Scale** (for clustered ensemble only)",
    )

    mo.vstack([ensemble_type_ui, n_samples_ui, scale_ui], justify="start")
    return ensemble_type_ui, n_samples_ui, scale_ui


@app.cell(hide_code=True)
def _(ensemble_type_ui, n_samples_ui, scale_ui):
    # Parameters for the ensemble
    n_qubits = 1
    n_samples = n_samples_ui.value
    scale = scale_ui.value
    ensemble_type = ensemble_type_ui.value

    initial_ensemble = lib.generate_initial_ensemble(
        n_qubits, ensemble_type, n_samples, scale, rng
    )
    return initial_ensemble, n_qubits


@app.cell(hide_code=True)
def _(initial_ensemble):
    mo.vstack([
        mo.md("Drag and rotate this around to get a better visual feel of the distribution"),
        mo.mpl.interactive(lib.plot_bloch_sphere(initial_ensemble)).center(),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.vstack(
        [
            mo.md(r"""
            ## 2. Forward process (Diffusion)

            This consists of $T$ steps, where at each step $t$ a random unitary $U_i^{(t)}$ is applied to each state $i$ in the ensemble, which scrambles them. After $T$ steps, the states are completely scrambled and form a Haar random ensemble.
            """),
            mo.image(
                src="assets/noising-cluster-demo.png",
                caption="Scrambling m=1000 states sampled from the clustered distribution over T=20 steps, as shown on the Bloch sphere.",
            ),
            mo.md(r"""
            In practice, $U_t$ is implemented using the fast scrambling model and the Hardware Efficient Ansatz (HEA).
            """),
        ],
    )
    return


@app.cell(hide_code=True)
def _():
    _forward_circuit = mo.image(src="assets/forward-circuit.svg")
    _qsc_block = mo.image(src="assets/qsc-block.svg")

    _fast_scrambling_details = mo.md(f"""
    This is what the forward scrambling circuit looks like:

    {_forward_circuit.center()}

    where each Quantum Scrambling Circuit (QSC) block looks like this:

    {_qsc_block.center()}

    There are layers of tunable single-qubit rotation gates followed by homogeneous entangling layers of all-to-all ZZ rotations in the QSC block. The parameters of the single-qubit gates are randomly sampled at each step to achieve fast scrambling.
    """).style(
        {
            "margin": "0px 20px",
        }
    )

    mo.md(f"""
    <details>
        <summary>
        If you are curious about the details of the Fast Scrambling method, click here.
        </summary>

        {_fast_scrambling_details}
    </details>
    """)
    return


@app.cell
def _(initial_ensemble, n_qubits):
    T_param = 4
    schedule = 0.3 * (np.arange(1, T_param + 1) ** 2)

    forward_process = lib.ForwardProcess(
        n_qubits, initial_ensemble, T_param, schedule
    )

    with mo.status.progress_bar(
        total=T_param, title="Scrambling Ensemble...", remove_on_exit=True
    ) as bar:
        forward_process.scramble(rng, progress_bar=bar)
    return T_param, forward_process


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Scrambling vizualized

    Hit the play button to see the diffusion process!
    """)
    return


@app.cell(hide_code=True)
def _(forward_process):
    forward_slider = mo.ui.anywidget(
        PlaySlider(
            min_value=0,
            max_value=forward_process.T,
            step=1,
            interval_ms=1000,
            loop=True,
        )
    )
    return (forward_slider,)


@app.cell(hide_code=True)
def _(forward_slider):
    mo.vstack(
        [
            mo.md("Timestep $t$"),
            forward_slider,
        ],
        align="center",
    )
    return


@app.cell(hide_code=True)
def _(forward_process, forward_slider):
    mo.as_html(
        lib.plot_bloch_sphere(
            forward_process.ensembles[int(forward_slider.value.get("value", 0))]
        )
    ).center()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. Backward Process (Denoising)

    The reverse process consists of $T$ steps, where at each step $t$ a PQC is trained to learn the denoising map from the scrambled distribution at step $t$ to the less scrambled distribution at step $t-1$.
    """)
    return


@app.cell(hide_code=True)
def _():
    backward_circuit = mo.image(src="assets/backward-circuit.svg")
    pqc_block = mo.image(src="assets/pqc-block.svg")

    denoising_details = mo.md(f"""
    This is what the backward denoising circuit looks like:

    {backward_circuit.center()}

    where each Parametrized Quantum Circuit (PQC) block looks like this:

    {pqc_block.center()}

    The $R_X$ and $R_Y$ gates have learnable parameters.
    """).style(
        {
            "margin": "0px 20px",
        }
    )

    mo.md(f"""
    <details>
        <summary>
        If you are curious about the details of the Denoising method, click here.
        </summary>

        {denoising_details}
    </details>
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### The Loss Function

    We use **Wasserstein Distance** to measure how different the ensembles are from each other. It captures the geometry of the underlying space and is a faithful measure of distributional similarity. So this will make for a perfect loss function!
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ### Training Configuration
    """)
    return


@app.cell(hide_code=True)
def _():
    epochs_ui = mo.ui.number(
        start=0, stop=1000, step=50, value=600, label="Epochs per Step"
    )
    layers_ui = mo.ui.slider(
        start=2, stop=10, step=1, value=4, show_value=True, label="PQC Layers"
    )
    ancilla_ui = mo.ui.slider(
        start=1, stop=4, step=1, value=1, show_value=True, label="Ancilla Qubits"
    )
    ensemble_size = 200

    mo.hstack([epochs_ui, layers_ui, ancilla_ui])
    return ancilla_ui, ensemble_size, epochs_ui, layers_ui


@app.cell
def _():
    train_button = mo.ui.run_button(label="Start Training")
    train_button
    return (train_button,)


@app.cell
def _(
    ancilla_ui,
    ensemble_size,
    epochs_ui,
    forward_process,
    layers_ui,
    n_qubits,
    train_button,
):
    mo.stop(
        not train_button.value,
        mo.md("Click 'Start Training' to see the results."),
    )

    model = lib.BackwardProcess(
        n_qubits=n_qubits,
        n_ancilla=ancilla_ui.value,
        T=forward_process.T,
        n_layers=layers_ui.value,
        ensemble_size=ensemble_size,
        epochs=epochs_ui.value,
        loss_func=lib.LossFunction.WASSERSTEIN,
        target_mode="previous",
    )

    model.train(forward_process.ensembles, rng)
    return (model,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Trained model
    """)
    return


@app.cell
def _(model):
    eval = model.run(rng, 500)
    return (eval,)


@app.cell(hide_code=True)
def _(model):
    backward_slider = mo.ui.anywidget(
        PlaySlider(
            min_value=0,
            max_value=model.T,
            step=1,
            interval_ms=1000,
            loop=True,
        )
    )
    return (backward_slider,)


@app.cell(hide_code=True)
def _(backward_slider):
    mo.vstack(
        [
            mo.md("The model was trained. Hit the play button to see how the trained circuits perform in the reverse process!"),
            mo.md("Timestep $t$").center(),
            backward_slider.center(),
        ],
    )
    return


@app.cell(hide_code=True)
def _(backward_slider, eval, model):
    mo.as_html(
        lib.plot_bloch_sphere(
            eval[
                model.T - int(backward_slider.value.get("value", 0))
            ]
        )
    ).center()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## **Can we make it better?**

    Instead of training each PQC to map from step $t$ to step $t-1$, we can train them to map directly to the initial distribution at step $t=0$. This way, we can bypass the diffusion process entirely and potentially achieve better performance with the same number of parameters. This is a simple alteration to the original Quantum DDPM framework, but it can have a significant impact on the training dynamics and the quality of the generated states.

    Let's train with this modified approach using the same hyperparameters and compare it to the standard method!
    """)
    return


@app.cell
def _():
    compare_params = {
        "n_qubits": 1,
        "n_samples": 1000,
        "ensemble_type": "circle",
        "ensemble_size": 600,
        "epochs": 600,
        "layers": 8,
        "ancilla": 2,
        "T": 6,
    }
    return (compare_params,)


@app.cell
def _():
    compare_button = mo.ui.run_button(
        label="Compare", kind="neutral"
    )
    compare_button
    return (compare_button,)


@app.cell(hide_code=True)
def _(T_param, compare_button, compare_params):
    mo.stop(
        not compare_button.value,
        mo.md("Click 'Compare' to see the results."),
    )

    _n_qubits = compare_params.get("n_qubits")
    _n_samples = compare_params.get("n_samples")
    _ensemble_type = compare_params.get("ensemble_type")
    _ensemble_size = compare_params.get("ensemble_size")
    _epochs = compare_params.get("epochs")
    _layers = compare_params.get("layers")
    _ancilla = compare_params.get("ancilla")
    _T = compare_params.get("T")

    _initial_ensemble = lib.generate_initial_ensemble(
        _n_qubits,
        _ensemble_type,
        _n_samples,
        0,
        rng,
    )

    standard_model = lib.BackwardProcess(
        n_qubits=_n_qubits,
        n_ancilla=_ancilla,
        T=_T,
        n_layers=_layers,
        ensemble_size=_ensemble_size,
        epochs=_epochs,
        loss_func=lib.LossFunction.WASSERSTEIN,
        target_mode="previous",
    )

    improved_model = lib.BackwardProcess(
        n_qubits=_n_qubits,
        n_ancilla=_ancilla,
        T=_T,
        n_layers=_layers,
        ensemble_size=_ensemble_size,
        epochs=_epochs,
        loss_func=lib.LossFunction.WASSERSTEIN,
        target_mode="initial",
    )

    _schedule = 0.3 * (np.arange(1, _T + 1) ** 2)

    _forward_process = lib.ForwardProcess(
        _n_qubits, _initial_ensemble, _T, _schedule
    )

    with mo.status.progress_bar(total=T_param, title="Scrambling Ensemble...", remove_on_exit=True) as _bar:
        _forward_process.scramble(rng, progress_bar=_bar)

    standard_model.train(_forward_process.ensembles, rng)

    mo.output.append(
        mo.md(
            "Standard QuDDPM trained. training the improved version..."
        )
    )

    improved_model.train(_forward_process.ensembles, rng)
    return improved_model, standard_model


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Results
    """)
    return


@app.cell(hide_code=True)
def _(improved_model, standard_model):
    standard_eval = standard_model.run(rng, 500)
    improved_eval = improved_model.run(rng, 500)
    return improved_eval, standard_eval


@app.cell(hide_code=True)
def _(improved_eval, standard_eval):
    mo.hstack(
        [
            mo.vstack(
                [
                    mo.md("Standard Quantum DDPM"),
                    lib.plot_bloch_sphere(
                        standard_eval[0]
                    )
                ],
                align="center",
            ),
            mo.vstack(
                [
                    mo.md("Improved version"),
                    lib.plot_bloch_sphere(
                        improved_eval[0]
                    )
                ],
                align="center",
            ),
        ],
        align="center",
    )
    return


@app.cell(hide_code=True)
def _(compare_button):
    mo.stop(
        not compare_button.value,
        mo.md("""
        You can see that the improved model converged much faster and produces a much better ensemble with the same number of parameters, compared to the standard model which is struggling with the constrained set of parameters.
        """)
    )
    return


@app.cell(hide_code=True)
def _(compare_params):
    def ema(data, span):
        """
        Computes the Exponential Moving Average (EMA) of data.
        """
        if len(data) == 0:
            return np.array([])
        alpha = 2 / (span + 1.0)
        ema_data = np.zeros_like(data, dtype=float)
        ema_data[0] = data[0]
        for i in range(1, len(data)):
            ema_data[i] = alpha * data[i] + (1 - alpha) * ema_data[i - 1]
        return ema_data

    def plot_loss_comparison(model1, model2):
        fig, ax = plt.subplots(figsize=(10, 5))
        epochs = compare_params.get("epochs")
        T = model1.T
        for t in range(1, T+1):
            ax.plot(
                range(t * epochs, (t + 1) * epochs),
                ema(model1.loss_history[T-t], span=10),
                color="blue",
                lw=0.75,
                label="Standard (t -> t-1)" if t == 1 else ""
            )
            ax.plot(
                range(t * epochs, (t + 1) * epochs),
                ema(model2.loss_history[T-t], span=10),
                color="green",
                lw=0.75,
                label="Improved (t -> 0)" if t == 1 else ""
            )
        ax.set_title("Loss History")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Wasserstein Loss")
        ax.set_yscale("log")
        ax.legend()
        return fig

    return (plot_loss_comparison,)


@app.cell
def _(improved_model, plot_loss_comparison, standard_model):
    plot_loss_comparison(standard_model, improved_model)
    return


if __name__ == "__main__":
    app.run()
