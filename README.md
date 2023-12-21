# Spin Echo Sim
A `CUDA` simulation for the dynamics of coupled spin ensembles in the context of an nuclear magnetic resonance (NMR) spin echo experiment.

### The Simulation
The simulation uses `CUDA` simulate large ($n\approx 10,000 - 100,000$) spin $1/2$ ensembles whose interactions can be represented at the mean field level.  Since the individual calculations for each particle involve small matrices, each particle's density matrix can be assigned its own thread on the GPU, allowing the simulation realize significant improvements in computation time over CPU or hybrid CPU/GPU simulations.

The code as written (`spin_echo_sim_new.cu`) is hard-coded with the current simulation in mind (i.e. the current ensemble type and Hamiltonian), but with moderate effort can be updated to model other mean-field type Hamiltonians.

Included in the repository is the `CUDA` simulation (`spin_echo_sim_new.cu`), two `Julia` versions (`SpinEchoSim_cpu.jl` and `SpinEchoSim_gpu.jl`) which represent CPU and hybrid CPU/GPU versions of the simulation, and several utilities for the generation of parameter files for all three simulation variants.  The main focus of these notes is the `CUDA` variant, as the other two variants are primarily used for benchmarking the performance of the `CUDA` simulation.

### Interaction

The simulation models the following interaction on a 2D lattice with periodic boundary conditions:

$H_i = -\nu_i I_z^{(i)} + \sum_{j} f(r_{ij}) \vec{\alpha} \cdot \vec{I}^{(j)}$

Where $I_{d}^{(i)}$ is the nuclear spin operator along the $d$ axis for particle $i$, $\nu_i$ is the Larmor frequency for particle $i$, $\vec{\alpha}$ is the vector strength of the interaction between the particles $\vec{\alpha} = (\alpha_x, \alpha_y, \alpha_z)$, $f(r_{ij})$ is a localization function which depends on the distance between particles $i$ and $j$, and the sum over $j$ represents the sum over the entire lattice.  This interaction represents a local mean field generated by a correlated electron mode mediated by the local nuclear magnetization (i.e. the alignment of the nuclear spins enforces alignment of the electron spins via hyperfine interactions, which feeds-back to act on the nuclear spins).

This interaction is calculated in the simulation in several steps.
* A "stencil" is created which pre-calculates the values of $f(r_{ij})$ for the whole lattice (`calc_stencil_2D`).  Because the simulation uses periodic boundary conditions, this stencil is origin-agnostic and can be periodically shifted to apply to any point on the lattice.  It is thus calculated once with $i = (0, 0)$ and re-used throughout the simulation.

Then, at each timepoint, for each particle (in parallel):
* The local magnetization at each point in the lattice is calculated (`calc_local_M_2D`) by taking the sum of the element-wise product of the stencil (a 2D matrix $A_{ij} \equiv f(r_{ij})$ and the evaluated magnetization of each particle in the lattice $B_{ij} \equiv$ magnetization i.e. $\langle \vec{I} \rangle$ of the particle at coordinate $(i,j)$.
* This local magnetization is given to the Hamiltonian calculation function (`calc_H`) along with the interaction strengths $\vec{\alpha}$ to calculate the Hamiltonian.

The simulation can updated to model other Hamiltonians by updating and/or removing these three functions within the propagation loop.  The Hamiltonian calculation should return a flattened array in row-major order such that the $i,j$th element of the Hamiltonian for particle $k$ corresponds to index $4k + 2i + j$ (with all indices starting at zero).

### Propagation

Time propagation is done in Liouville space using a discrete time-stepping method where the Hamiltonian is assumed constant on some small time step $dt$:

$\rho(t+dt) = exp(-iL(t)dt/\hbar)\rho(t)$

The Liouvillian operator $L$ is calculated from the Hamiltonian via $L = H \otimes I - I \otimes \bar{H}$.  Matrix exponentiation of the Liouvillian is performed using the scaling and squaring method.  A built-in matrix exponential function is not present in `CUDA` nor its linear algebra libraries i.e. `cuBLAS`  or `cuSolver`, so a custom algorithm was written for this simulation.  The key step of solving for the Pade Approximant $R \approx exp(A)$ for $A = -iLdt/\hbar$ in $QR = P$ was done using a simple Gaussian Elimination algorithm.  Because the simulation deals with matrices which are at most $4 \times 4$, the extreme scaling of the Gaussian Elimination algorithm is not a significant factor in performance.

## Using the CUDA Simulation

The `CUDA` simulation requires two parameter files: `echo_params{n}.txt` and `sim_params{n}.txt` for $n$ some integer (used to divide a large simulation set up into multiple jobs).  `echo_params` must contain the following parameters for the simulation in this order with a space between each one:

$\alpha_x\quad\alpha_y\quad\alpha_z\quad\xi\quad p\quad\Gamma_1\quad\Gamma_2\quad\Gamma_3\quad stencil\quad s_w\quad p_w\quad d_w\quad \theta_{90}\quad \theta_{180}\quad p_{90}\quad p_{180}$

The parameters $\xi$ and $p$ are used in the generation of the stencil ($\xi$ corresponds roughly to the correlation length of the electrons, and $p$ to the power scaling of the localization).  "stencil" is an integer between 0 and 3 which indicates which of the built-in forms of the stencil to be used (0 = Gaussian, 1 = Power Law, 2 = RKKY, 3 = constant).  The parameters $\Gamma$ represent the strengths of the dissipation.  The values $s_w, p_w, d_w$ specify the angular weight of the stencil.  $\theta_{90/180}$ are the pulse angles in radians and $p_{90/180}$ (an integer from 0 to 3) are the pulse phases.

The `spin_params` file must contain the following values in order:

$n_x\quad n_y\quad dt \quad \tau\quad LW$

With $n_x, n_y$ the dimensions of the lattice in particles, $dt$ the time step, $\tau$ the echo time, and $LW$ the linewidth of the frequency distribution.

With these parameter files specified, the `CUDA` simulation will loop over each combination of parameters:
```
for each line A of sim_params
   for each line B of echo_params
      simulate with params = A + B
```

The simulation code can be compiled with the `CUDA` compiler, e.g. `nvcc spin_echo_sim_new.cu -o spinsim` and takes as command line arguments several integer values which direct it to the right parameter files:

`./spinsim <trial number> <time execution> <check errors> <resample frequencies> <error steps>`

"Trial number" is the $n$ value of the parameter files.  "Time execution" is a boolean (i.e. 0 or 1) used for benchmarking the simulation.  "Check Errors" and "Error Steps" are for debugging, and save intermediate parameters for a number of time steps = "Error Steps" at a significant cost to execution time.  "Resample Frequencies" indicates whether a fixed distribution should be used or a fresh frequency distribution should be generated for each trial/set of parameters.  In typical use, all but potentially "resample frequencies" will be set to zero, i.e. "off":

`./spinsim <n> 0 0 0 0`

The outputs of the `CUDA` simulation are `.txt` files which, on each line, contain the signal (i.e. net magnetization) of the ensemble vs time for a given parameter set, e.g. the 1st row of the output contains the net magnetization vs time (space delimited) for the 1st set of `spin_params` and first set of `echo_params`.  The 2nd row corresponds to the 1st set of `spin_params` and second set of `echo_params`, and so on.

The output is split into real and imaginary, in-plane and out-of-plane, for four outputs: `real_output{n}.txt`, `imag_output{n}.txt`, `z_real_output{n}.txt`, and `z_imag_output{n}.txt`.

## The Julia Simulation

The `Julia` simulation is used in much the same way as the `CUDA` simulation, but it only requires a single parameter file (given the memory does not need to be allocated by hand, we do not need to split up those parameters which inform the allocation from those which do not).  Its parameter file is just called `params.txt` and contains the variables in the following order:

$\alpha_x\quad\alpha_y\quad\alpha_z\quad\xi\quad p\quad\Gamma_1\quad\Gamma_2\quad\Gamma_3\quad stencil\quad n_x \quad n_y \quad dt \quad \tau \quad s_w \quad p_w \quad d_w \quad \theta$

The `Julia` simulation does not currently support phase cycling, and its pulse angles are fixed at $\theta, 2\theta$ (i.e. 90 and 180, and cannot do other pulsing schemes such as 90, 90).  The simulation can be run from the command line with `julia generate_data_set_<gpu/cpu>.jl`, and the echo data is stored to `.txt` files `echoes_r, echoes_i, zechoes_r, zechoes_i`.

The `Julia` simulation also includes a Jupyter Notebook version of `generate_data_set` for interactive simulation.
