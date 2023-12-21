from pathlib import Path
import os
from itertools import product
import numpy as np

ECHO_LOAD_ORDER = ["αx", "αy", "αz", "ξ", "p", "Γ1", "Γ2", "Γ3", "sten", "sw", "pw", "dw", "θ90", "θ180", "p90", "p180"]
SIM_LOAD_ORDER = ["nx", "ny", "dt", "τ", "line_width"]


def load_data(dir_name):

    # load the echoes from output
    rpath = Path(os.path.join(dir_name, "real_output1.txt"))
    ipath = Path(os.path.join(dir_name, "imag_output1.txt"))
    z_rpath = Path(os.path.join(dir_name, "z_real_output1.txt"))
    z_ipath = Path(os.path.join(dir_name, "z_imag_output1.txt"))

    # load the real data
    with rpath.open() as f:
        data = f.read()
        lines = data.split("\n")
        r = []
        for line in lines:
            line = [float(x) for x in line.split(" ") if x != ""]
            if line:
                r.append(line)

    # load the real data
    with ipath.open() as f:
        data = f.read()
        lines = data.split("\n")
        i = []
        for line in lines:
            line = [float(x) for x in line.split(" ") if x != ""]
            if line:
                i.append(line)

    # load the real data
    with z_rpath.open() as f:
        data = f.read()
        lines = data.split("\n")
        zr = []
        for line in lines:
            line = [float(x) for x in line.split(" ") if x != ""]
            if line:
                zr.append(line)

    # load the real data
    with z_ipath.open() as f:
        data = f.read()
        lines = data.split("\n")
        zi = []
        for line in lines:
            line = [float(x) for x in line.split(" ") if x != ""]
            if line:
                zi.append(line)

    return r, i, zr, zi


def load_params():

    epath = Path(os.path.join("params", "echo_params1.txt"))
    spath = Path(os.path.join("params", "sim_params1.txt"))

    with epath.open() as f:
        data = f.read()
        lines = data.split("\n")
        echo_params = []
        for line in lines:
            line = [float(x) for x in line.split(" ") if x != ""]
            if line:
                echo_params.append(line)

    with spath.open() as f:
        data = f.read()
        lines = data.split("\n")
        sim_params = []
        for line in lines:
            line = [float(x) for x in line.split(" ") if x != ""]
            if line:
                sim_params.append(line)

    params = list(product(sim_params, echo_params))
    for idx, param in enumerate(params):
        t_param = {}
        for jdx in range(len(SIM_LOAD_ORDER)):
            t_param[SIM_LOAD_ORDER[jdx]] = param[0][jdx]
        for jdx in range(len(ECHO_LOAD_ORDER)):
            t_param[ECHO_LOAD_ORDER[jdx]] = param[1][jdx]
        params[idx] = t_param

    # loop through and make a new omega p, omega z parameter
    for param in params:

        # get the stencil weight
        stencil_lookup = {
            0: gauss,
            1: pow_law,
            2: rkky
        }
        w = calc_stencil(stencil_lookup[int(param["sten"])], int(param["nx"]), int(param["ny"]), ξ=param["ξ"])

        # get the planar weight
        ωp = 2 * param["αx"] * w

        # save this
        Γ = 50000
        π = 3.14
        γ = 2 * π * 1e6
        param["ωp"] = γ * ωp / (2 * π * Γ)

    return params


# a function to calculate the stencil weight given ξ, p
# the "sweet spot" for the interaction is best set relative to stencil weight
def calc_stencil(func, nx, ny, **kwargs):

    # just go through each particle and add it up
    output = 0
    for idx in range(nx):
        for jdx in range(ny):

            # no self coupling
            if idx == jdx:
                continue

            # periodic boundary conditions
            x = (idx + nx / 2) % nx - nx / 2
            y = (jdx + ny / 2) % ny - ny / 2

            # distnace from origin to point
            r = np.sqrt(x ** 2 + y ** 2)

            # run through the stencil func and give stencil args as kwargs
            output += func(r, **kwargs)

    return output


# a few pre-defined stencils
def gauss(r, ξ=10000.0):  # default to "global"
    return np.exp(-(r / ξ) ** 2)


# default to inverse square
def pow_law(r, p=2.0):
    return np.power(r, -p)


# rkky type
def rkky(r, ξ):
    x = 2 * r / ξ
    return np.pow(x, -4) * (x * np.cos(x) - np.sin(x))
