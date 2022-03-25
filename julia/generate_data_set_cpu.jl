using StaticArrays, DelimitedFiles, StatsBase, Random, Distributions, LinearAlgebra
include("SpinEchoSim_cpu.jl")

# make the basic parameter dict
echo_params = make_params()

# load the pre-set frequency distribution
ν_pre = [];
strname = "freqs.txt";
open(strname) do file
    for l in eachline(file)
        push!(ν_pre, parse(Float32, l));
    end
end

# input file formatting: | αx | αy | αz | ξ | p | Γ1 | Γ2 | Γ3 | stencil form | nx | ny | dt | τ | sw | pw | dw | θ |
fname = "params.txt";
input_params = readdlm(fname, Float32)
num_samps = size(input_params)[1]

# first echo's lattice
echo_params["nx"] = convert(Int64, input_params[1,10]);
echo_params["ny"] = convert(Int64, input_params[1,11]);
echo_params["n"] = (echo_params["nx"], echo_params["ny"]);

# format the frequency distribution
ν_res = zeros(Float32, echo_params["nx"], echo_params["ny"]);
for i = 1:echo_params["nx"]
    for j = 1:echo_params["ny"]
        ν_res[i,j] = ν_pre[echo_params["ny"]*(i-1) + j];
    end
end

fM_list = Array{Any}(undef, num_samps)
fMz_list = Array{Any}(undef, num_samps)
for idx in range(1, length = num_samps)

    # load this echo's parameters
    αx = convert(Float32, input_params[idx, 1]); # correlation sterngth
    αy = convert(Float32, input_params[idx, 2]); # correlation sterngth
    αz = convert(Float32, input_params[idx, 3]); # correlation sterngth
    ξ = convert(Float32, input_params[idx, 4]); # correlation length
    p = convert(Float32, input_params[idx, 5]); # correlation power
    Γ1 = convert(Float64, input_params[idx, 6]); # dissipation power
    Γ2 = convert(Float64, input_params[idx, 7]); # dissipation power
    Γ3 = convert(Float64, input_params[idx, 8]); # dissipation power
    sten_form = input_params[idx, 9]; # stencil form
    nx = convert(Int64, input_params[idx, 10]); # num spins in x direction
    ny = convert(Int64, input_params[idx, 11]); # num spins in y direction
    dt = convert(Float32, input_params[idx, 12]); # time step size
    τ = convert(Float32, input_params[idx, 13]); # echo time
    sw = convert(Float32, input_params[idx, 14]);
    pw = convert(Float32, input_params[idx, 15]);
    dw = convert(Float32, input_params[idx, 16]);
    θ = convert(Float32, input_params[idx, 17]);

    # interaction strengths
    echo_params["αx"] = αx;
    echo_params["αy"] = αy;
    echo_params["αz"] = αz;
    
    # sample size
    echo_params["nx"] = nx
    echo_params["ny"] = ny
    echo_params["n"] = (nx, ny)
    
    # time steps
    echo_params["dt"] = dt;
    echo_params["τ"] = τ;
    
    # dissipation
    Γ = [Γ1, Γ2, Γ3];
    echo_params["Γ"] = Γ;
    
    # stencil variables & creation
    echo_params["spd"] = [sw, pw, dw]
    echo_params["ξ"] = ξ;
    echo_params["p"] = p;
    echo_params["M_stencil"] = make_stencil(echo_params, sten_form);
    
    # flip angle
    echo_params["flip_angle"] = θ

    # generate temporary parameters
    tparams = make_temp_params(echo_params)

    # assign fixed freq distribution if you want it
    tparams["ν"] = ν_res;
    
    # simulate
    print("starting echo ", string(idx),"/", string(num_samps),"\n")
    @time fM_list[idx], fMz_list[idx], tparams = spin_echo_sim_liouville(tparams)
    
    # save profiling data
    echo_params["profiling"] = deepcopy(tparams["profiling"])
    
end

fname_r = "zechoes_r.txt"
fname_i = "zechoes_i.txt"
open(fname_r, "w") do io
    writedlm(io, real(fMz_list))
end
open(fname_i, "w") do io
    writedlm(io, imag(fMz_list))
end

fname_r = "echoes_r.txt"
fname_i = "echoes_i.txt"
open(fname_r, "w") do io
    writedlm(io, real(fM_list))
end
open(fname_i, "w") do io
    writedlm(io, imag(fM_list))
end

fname_t = "profiling_data.txt"
open(fname_t, "w") do io
    
    for x in keys(echo_params["profiling"])
        
        str = x*"\n"
        write(io, str)
        
        str = ""
        for idx = 1:length(echo_params["profiling"][x])-1
            str *= string(echo_params["profiling"][x][idx])*" "
        end
        str *= string(echo_params["profiling"][x][end])*"\n"
        write(io, str)
        
    end
    
end    