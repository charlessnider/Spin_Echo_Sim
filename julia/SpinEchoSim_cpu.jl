module SpinEchoSim

  include("cpu/SpinSimParams.jl")
  export SpinSimParams
  include("cpu/spin_sims.jl")
  export spin_echo_sim, spin_echo_sim_liouville

end

using .SpinEchoSim
using .SpinSimParams
