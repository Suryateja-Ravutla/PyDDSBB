
import time
from dataclasses import dataclass
from typing import Tuple, Dict, Callable, Optional
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# Try import PyDDSBB
try:
    import PyDDSBB
except Exception:
    PyDDSBB = None

# -------------------- Functions to optimize --------------------

def f_sin2pix(x):        # sin(2πx)
    return x*np.sin(2*np.pi*x)

def f_sin_combo(x):      # sin(3x) + 0.5 sin(7x)
    return np.sin(3*x) + 0.5*np.sin(7*x)

def f_poly(x):           # x^3 - 0.5x^2 + 0.2x
    return x**3 - 0.5*x**2 + 0.2*x

def f_gauss_bump(x):
    return np.exp(-((x - 0.2)**2)/0.01) - 0.7*np.exp(-((x - 0.75)**2)/0.02)

def f_piecewise(x):
    return np.abs(x - 0.5)

def f_rational(x):
    return (x**2 + 10*x) / (1 + x**2)

def f_linear(x):
    return 2*x - 1

FUNCTIONS = {
    "x * Sin(sin(2πx))": f_sin2pix,
    "Sine (sin(3x) + 0.5 sin(7x))": f_sin_combo,
    "Polynomial (x^3 - 0.5x^2 + 0.2x)": f_poly,
    "Gaussian bump": f_gauss_bump,
    "Piecewise (abs(x - 0.5))": f_piecewise,
    "Rational (x^2 + 10x) / (1 + x^2)": f_rational,
    "Linear-ish (2x - 1)": f_linear,
}

# -------------------- Helpers --------------------

def noisy_eval(f: Callable[[np.ndarray], np.ndarray], x: float, noise_std: float, rng: np.random.Generator) -> float:
    v = float(f(np.array([x]))[0] if isinstance(f(np.array([x])), np.ndarray) else f(x))
    if noise_std > 0:
        v += rng.normal(0.0, noise_std)
    return v

def curve(f: Callable[[np.ndarray], np.ndarray], xmin: float, xmax: float, noise_std: float = 0.0, seed: int = 42):
    xs = np.linspace(xmin, xmax, 600)
    ys = f(xs)
    if noise_std > 0:
        rng = np.random.default_rng(seed)
        ys = ys + rng.normal(0.0, noise_std, size=ys.shape)
    return xs, ys

def make_plot(xs, ys, xopt: Optional[float], yopt: Optional[float], f_true_xs, f_true_ys, title="Objective landscape"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=f_true_xs, y=f_true_ys, mode="lines", name="True f(x)"))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Observed f(x)"))
    if xopt is not None and yopt is not None:
        fig.add_trace(go.Scatter(x=[xopt], y=[yopt], mode="markers", name="PyDDSBB optimum", marker=dict(size=10)))
    fig.update_layout(title=title, xaxis_title="x", yaxis_title="f(x)", hovermode="x", height=520, margin=dict(l=10,r=10,t=40,b=10))
    return fig

# -------------------- Streamlit UI --------------------

st.set_page_config(page_title="PyDDSBB 1D Global Optimization", layout="wide")
st.title("PyDDSBB Demo : 1D Global Optimization")

if PyDDSBB is None:
    st.error("PyDDSBB not available. On Streamlit Cloud, include `glpk` via packages.txt and install PyDDSBB from GitHub in requirements.txt.")
    st.stop()

with st.sidebar:
    st.header("Objective")
    kind = st.selectbox("Choose f(x)", list(FUNCTIONS.keys()), index=0)
    f = FUNCTIONS[kind]
    x_min, x_max = st.slider("Domain [x_min, x_max]", -3.0, 3.0, (-1.0, 1.0), step=0.1)
    seed = st.number_input("Random seed", 0, 999999, 42)
    noise_std = st.slider("Observation noise (optional)", 0.0, 0.5, 0.0, step=0.01, help="If >0, the objective is noisy.")

    st.header("Solver (PyDDSBB)")
    n_init = st.number_input("Initial samples (n_init)", min_value=3, max_value=200, value=12, step=1)
    split_method = st.selectbox("Split method", ["equal_bisection", "golden_section"])
    variable_selection = st.selectbox("Variable selection", ["longest_side", "svr_var_select"])
    multifidelity = st.checkbox("Multifidelity", value=False)
    sense = st.selectbox("Sense", ["minimize", "maximize"], index=0)

    st.subheader("Stopping criteria")
    abs_tol = st.number_input("absolute_tolerance", value=1e-3, format="%.6f")
    rel_tol = st.number_input("relative_tolerance", value=1e-3, format="%.6f")
    min_bound = st.number_input("minimum_bound", value=0.01, format="%.4f")
    sampling_limit = st.number_input("sampling_limit", min_value=10, max_value=20000, value=800, step=10)
    time_limit = st.number_input("time_limit (s)", min_value=1.0, max_value=36000.0, value=20.0, step=1.0, format="%.1f")

    st.divider()
    auto_run = st.checkbox("Auto-run on change", value=False)
    run_btn = st.button("Run optimization")
    resume_btn = st.button("↻ Resume with more budget")

# Build objective and model
rng = np.random.default_rng(int(seed))

def objective(x_arr):
    # x_arr is an array-like; 1D problem → first element
    xval = float(x_arr[0])
    return noisy_eval(f, xval, noise_std, rng)

# Create DDSBB model
model = PyDDSBB.DDSBBModel.Problem()
model.add_objective(objective, sense=sense)
model.add_variable(float(x_min), float(x_max))

stop_option = {
    "absolute_tolerance": float(abs_tol),
    "relative_tolerance": float(rel_tol),
    "minimum_bound": float(min_bound),
    "sampling_limit": int(sampling_limit),
    "time_limit": float(time_limit),
}

# Keep solver in session to support "resume"
if "solver" not in st.session_state:
    st.session_state.solver = None
if "last_cfg" not in st.session_state:
    st.session_state.last_cfg = None
if "result" not in st.session_state:
    st.session_state.result = None

cfg = dict(kind=kind, x_min=x_min, x_max=x_max, seed=int(seed), noise_std=float(noise_std),
           n_init=int(n_init), split_method=split_method, variable_selection=variable_selection,
           multifidelity=multifidelity, stop_option=stop_option, sense=sense)

def new_solver():
    return PyDDSBB.DDSBB(
        int(n_init),
        split_method=split_method,
        variable_selection=variable_selection,
        multifidelity=multifidelity,
        stop_option=stop_option,
        sense=sense
    )

should_run = False
if run_btn:
    st.session_state.solver = new_solver()
    should_run = True
elif auto_run and cfg != st.session_state.last_cfg:
    st.session_state.solver = new_solver()
    should_run = True
elif resume_btn and st.session_state.solver is not None:
    # Increase budgets for resume
    extra_sampling = int(max(100, 0.5 * sampling_limit))
    new_stop = dict(stop_option)
    new_stop["sampling_limit"] = int(stop_option["sampling_limit"]) + extra_sampling
    st.session_state.solver.resume(new_stop)

if should_run and st.session_state.solver is not None:
    st.session_state.last_cfg = cfg
    with st.spinner("Optimizing with PyDDSBB..."):
        start = time.time()
        st.session_state.solver.optimize(model)
        elapsed = time.time() - start
        yopt = float(st.session_state.solver.get_optimum())
        xopt = float(st.session_state.solver.get_optimizer()[0])
        st.session_state.result = {"xopt": xopt, "yopt": yopt, "elapsed": elapsed}

# ---- Visualization
xs_true, ys_true = curve(FUNCTIONS[kind], x_min, x_max, noise_std=0.0)
xs_obs, ys_obs = curve(FUNCTIONS[kind], x_min, x_max, noise_std=noise_std, seed=int(seed))

xopt = st.session_state.result["xopt"] if st.session_state.result else None
yopt = st.session_state.result["yopt"] if st.session_state.result else None
fig = make_plot(xs_obs, ys_obs, xopt, yopt, xs_true, ys_true, title=f"{kind}")

col_main, col_side = st.columns([4, 1.4])
with col_main:
    st.plotly_chart(fig, use_container_width=True)

with col_side:
    st.subheader("Result")
    if st.session_state.result is None:
        st.info("Set options and click **Run optimization**.")
    else:
        st.metric("x*", f"{xopt:.6g}")
        st.metric("f(x*)", f"{yopt:.6g}")
        st.caption(f"Elapsed: {st.session_state.result['elapsed']:.2f} s")
        st.caption(f"n_init: {n_init} · split: {split_method} · var sel: {variable_selection}")
        st.caption(f"noise σ: {noise_std}")

st.caption("PyDDSBB solves a black-box global optimization by building data-driven convex relaxations and adaptively sampling the domain. This demo uses a 1D objective f(x) with optional noise.")
