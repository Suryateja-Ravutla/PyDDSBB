
import time
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
import streamlit as st
import plotly.graph_objects as go

try:
    import PyDDSBB
except Exception:
    PyDDSBB = None

def set_seeds(seed: int = 42):
    np.random.seed(seed)

@dataclass
class DataBundle:
    x: np.ndarray
    y: np.ndarray
    y_clean: np.ndarray
    x_grid: np.ndarray
    y_clean_grid: np.ndarray
    noise_std: float

def true_function(x: np.ndarray, kind: str) -> np.ndarray:
    if kind == "Sine (sin(2πx))":
        return np.sin(2 * np.pi * x)
    if kind == "Sine (sin(3x) + 0.5 sin(7x))":
        return np.sin(3 * x) + 0.5 * np.sin(7 * x)
    if kind == "Polynomial (x^3 - 0.5x^2 + 0.2x)":
        return x**3 - 0.5 * x**2 + 0.2 * x
    if kind == "Gaussian bump":
        return np.exp(-((x - 0.2) ** 2) / 0.01) - 0.7 * np.exp(-((x - 0.75) ** 2) / 0.02)
    if kind == "Piecewise (abs(x - 0.5))":
        return np.abs(x - 0.5)
    if kind == "Rational (x^2 + 10x) / (1 + x^2)":
        return (x**2 + 10 * x) / (1 + x**2)
    return 2 * x - 1

def generate_data(n_points: int, x_range: Tuple[float, float], noise_std: float, kind: str, seed: int) -> DataBundle:
    set_seeds(seed)
    x = np.random.uniform(x_range[0], x_range[1], size=n_points)
    x = np.sort(x)
    y_clean = true_function(x, kind)
    y = y_clean + np.random.normal(0.0, noise_std, size=n_points)
    x_grid = np.linspace(x_range[0], x_range[1], 400)
    y_clean_grid = true_function(x_grid, kind)
    return DataBundle(x=x, y=y, y_clean=y_clean, x_grid=x_grid, y_clean_grid=y_clean_grid, noise_std=noise_std)

def model_poly(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    a0, a1, a2, a3 = theta
    return a0 + a1*x + a2*(x**2) + a3*(x**3)

def model_rational(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    a, b, c = theta
    return (a*(x**2) + b*x) / (1.0 + np.abs(c)*(x**2) + 1e-8)

def model_gauss2(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    A1, mu1, s1, A2, mu2, s2 = theta
    s1 = np.abs(s1) + 1e-6
    s2 = np.abs(s2) + 1e-6
    return A1*np.exp(-0.5*((x-mu1)/s1)**2) + A2*np.exp(-0.5*((x-mu2)/s2)**2)

MODEL_ZOO = {
    "Cubic polynomial (a0 + a1 x + a2 x^2 + a3 x^3)": (model_poly, ["a0","a1","a2","a3"]),
    "Rational ( (a x^2 + b x) / (1 + c x^2) )": (model_rational, ["a","b","c"]),
    "Two Gaussians (A1, mu1, s1, A2, mu2, s2)": (model_gauss2, ["A1","mu1","s1","A2","mu2","s2"]),
}

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred)**2))

def run_pyddsbb(X, y, model_fn, var_bounds, n_init, split_method, variable_selection, multifidelity, stop_option):
    if PyDDSBB is None:
        raise ImportError("PyDDSBB is not installed.")
    def objective(theta_vec):
        theta = np.array(theta_vec, dtype=float)
        return mse(y, model_fn(X, theta))
    model = PyDDSBB.DDSBBModel.Problem()
    model.add_objective(objective, sense='minimize')
    for (lb, ub) in var_bounds:
        model.add_variable(lb, ub)
    solver = PyDDSBB.DDSBB(n_init, split_method=split_method, variable_selection=variable_selection,
                           multifidelity=multifidelity, stop_option=stop_option)
    start = time.time()
    solver.optimize(model)
    return {
        "theta_opt": np.array(solver.get_optimizer(), dtype=float),
        "obj_opt": float(solver.get_optimum()),
        "train_time": time.time() - start,
    }

def make_fit_plot(data: DataBundle, model_fn, theta_opt: np.ndarray, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.x, y=data.y, mode="markers", name="Data"))
    if theta_opt is not None:
        fig.add_trace(go.Scatter(x=data.x_grid, y=model_fn(data.x_grid, theta_opt),
                                 mode="lines", name="PyDDSBB best fit"))
    fig.add_trace(go.Scatter(x=data.x_grid, y=data.y_clean_grid, mode="lines",
                             name="True function", line=dict(dash="dash")))
    fig.update_layout(title=title, xaxis_title="x", yaxis_title="y")
    return fig

st.set_page_config(page_title="PyDDSBB 1D Fit", layout="wide")
st.title("PyDDSBB: 1D Parametric Fitting")

if PyDDSBB is None:
    st.error("PyDDSBB not available. Make sure to include glpk in packages.txt.")
    st.stop()

with st.sidebar:
    st.header("Data")
    func_kind = st.selectbox("Function", [
        "Sine (sin(2πx))", "Sine (sin(3x) + 0.5 sin(7x))",
        "Polynomial (x^3 - 0.5x^2 + 0.2x)", "Gaussian bump",
        "Piecewise (abs(x - 0.5))", "Rational (x^2 + 10x) / (1 + x^2)",
        "Linear-ish (2x - 1)"
    ])
    n_points = st.slider("Points", 30, 2000, 200, step=10)
    x_min, x_max = st.slider("x range", -2.0, 2.0, (-1.0, 1.0), step=0.1)
    noise_std = st.slider("Noise std", 0.0, 1.0, 0.05, step=0.01)
    seed = st.number_input("Seed", 0, 999999, 42)

    st.header("Model")
    model_name = st.selectbox("Family", list(MODEL_ZOO.keys()))
    model_fn, param_names = MODEL_ZOO[model_name]

    bounds = []
    for p in param_names:
        lo = st.number_input(f"{p} lower", value=-2.0, key=f"lb_{p}")
        hi = st.number_input(f"{p} upper", value=2.0, key=f"ub_{p}")
        bounds.append((float(lo), float(hi)))

    st.header("Solver")
    n_init = st.number_input("n_init", 3, 200, max(5, len(param_names)*3))
    split_method = st.selectbox("Split", ["equal_bisection", "golden_section"])
    variable_selection = st.selectbox("Var selection", ["longest_side", "svr_var_select"])
    multifidelity = st.checkbox("Multifidelity", False)
    stop_option = {
        "absolute_tolerance": st.number_input("abs tol", value=1e-3, format="%.6f"),
        "relative_tolerance": st.number_input("rel tol", value=1e-3, format="%.6f"),
        "minimum_bound": st.number_input("min bound", value=0.01, format="%.4f"),
        "sampling_limit": st.number_input("samples", 10, 100000, 800),
        "time_limit": st.number_input("time (s)", 1.0, 36000.0, 30.0, format="%.1f")
    }

    run_btn = st.button("Run")

data = generate_data(n_points, (x_min, x_max), noise_std, func_kind, seed)
if run_btn:
    res = run_pyddsbb(data.x, data.y, model_fn, bounds, int(n_init),
                      split_method, variable_selection, multifidelity, stop_option)
    st.plotly_chart(make_fit_plot(data, model_fn, res["theta_opt"], "Fit"), use_container_width=True)
    st.json(res)
