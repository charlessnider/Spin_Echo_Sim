## `CUDA` Simulation Example Data and Workspace

To use the example...
* Copy the parameter files in `params` to the directory from which you will run the simulation (e.g. `Spin_Echo_Sim/cuda`).
* Run the simulation _with resample frequencies set to true_
```
> nvcc spin_echo_sim_new.cu -o spinsim
> ./spinsim 1 0 0 1 0
```
You may need to adjust the compiler path according to your own install.
* Copy the resulting `.txt` files (`real_output`, `imag_output`, `z_real_output`, `z_imag_output`) to `example/output`.  Do not overwrite `std_output`!  This is the reference data.
* Run the `Analyze Echoes` notebook to examine your output versus the expected output.

## Reference Notes
The simulation was last tested on `CUDA` release 12.3 on December 21st, 2023 on a GTX 1080 Ti.  The reference data was generated using this configuration.
