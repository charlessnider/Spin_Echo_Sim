module SpinEchoSim

  include("gpu/SpinSimParams.jl")
  export SpinSimParams
  include("gpu/spin_sims.jl")
  export spin_echo_sim, spin_echo_sim_liouville

end

using .SpinEchoSim
using .SpinSimParams
