module SpinSimParams

    using StaticArrays
    using StatsBase
    using Random, Distributions
    using LinearAlgebra
    using CUDA

    # fixed parameters
    function make_params()

        f = Dict();

        # magnetization
        f["Ix"] = convert.(Complex{Float32}, @SArray [0 1/2; 1/2 0])
        f["Iy"] = convert.(Complex{Float32}, @SArray [0 -1im/2; 1im/2 0])
        f["Iz"] = convert.(Complex{Float32}, @SArray [1/2 0; 0 -1/2])
        f["Ip"] = f["Ix"] + 1im*f["Iy"]
        f["Im"] = f["Ix"] - 1im*f["Iy"]
        f["M_op"] = f["Ip"]

        # pulse operators
        f["flip_angle"] = π/2

        # time variables
        f["γ"] = convert.(Float32, 2*pi*1e6)
        f["τ"] = convert.(Float32, 50e-6)
        f["dt"] = convert.(Float32, 2)

        # interaction variables
        f["αx"] = convert(Float32, 0.0005)
        f["αy"] = convert(Float32, 0.0005)
        f["αz"] = convert(Float32, 0.0)
        f["ξ"] = convert(Float32, 10.0)
        f["p"] = convert(Float32, 2.0)

        # spin ensemble variables
        f["ν0"] = convert(Float32, 10)
        f["bw"] = convert(Float32, 0.5)
        f["n"] = (100, 100)
        f["line_width"] = convert(Float32, 0.05)

        # default jump operator values
        f["Γ"] = convert.(Float32, [0, 0, 0.001])

        # initial condition
        f["ψ0"] = convert.(Complex{Float32}, @SArray[1; 0])
    
        # profiling
        f["profiling"] = Dict()
        f["profiling"]["M_eval"] = []
        f["profiling"]["Mz_eval"] = []
        f["profiling"]["M"] = []
        f["profiling"]["Mz"] = []
        f["profiling"]["M_eval_to_CuArray"] = []
        f["profiling"]["Mz_eval_to_CuArray"] = []
        f["profiling"]["M_stencil_vec"] = []
        f["profiling"]["M_local"] = []
        f["profiling"]["Mz_local"] = []
        f["profiling"]["M_local_to_Array"] = []
        f["profiling"]["Mz_local_to_Array"] = []
        f["profiling"]["calc_H"] = []
        f["profiling"]["H_to_Liouville"] = []
        f["profiling"]["calc_U"] = []
        f["profiling"]["time_evolve"] = []

        return f

    end

    # dependent parameters
    function make_temp_params(params)

        # copy of params
        f = deepcopy(params)

        # hard code pulse operators (x/x pulsing)
        θ = f["flip_angle"]
        f["U90"]= convert.(Complex{Float32}, [cos(θ/2) -1im*sin(θ/2); -1im*sin(θ/2) cos(θ/2)]) # along x = phase "0"
        f["U180"] = convert.(Complex{Float32}, [cos(θ) -1im*sin(θ); -1im*sin(θ) cos(θ)]) # along x = phase "0"
        # f["U180"] = convert.(Complex{Float32}, [cos(θ) -sin(θ); sin(θ) cos(θ)]) # along y = phase "1"

        # number of points
        f["nτ"] = convert(Int64, round(f["τ"]*f["γ"]/f["dt"]));

        # number of frequencies
        f["nfreq"] = prod(f["n"]);

        # frequency sampling
        x = convert.(Float32, collect(LinRange(f["ν0"] - f["bw"]/2, f["ν0"] + f["bw"]/2, f["nfreq"])))
        f["ν"] = convert.(Complex{Float32}, sample(x, Weights(lorentzian.(x, f["ν0"], f["line_width"])), f["n"]))
        f["P"] = convert.(Complex{Float32}, fill(1/prod(f["n"]), f["n"]))
        f["ψ_init"] = fill(f["ψ0"], f["n"])

        # hard code the dissipation (using float64: exp more accurate in float64)
        Γ1 = f["Γ"][1]
        Γ2 = f["Γ"][2]
        Γ3 = f["Γ"][3]
        J1 = convert.(Complex{Float64}, [-Γ1 0 0 Γ1; 0 -Γ1/2 0 0; 0 0 -Γ1/2 0; 0 0 0 0])
        J2 = convert.(Complex{Float64}, [0 0 0 0; 0 -Γ2/2 0 0; 0 0 -Γ2/2 0; Γ2 0 0 -Γ2])
        J3 = convert.(Complex{Float64}, [0 0 0 0; 0 -Γ3/2 0 0; 0 0 -Γ3/2 0; 0 0 0 0])
        f["J"] = J1 + J2 + J3;

        return f

    end

    # make the stencil
    function make_stencil(params, func = 0)

        # size of lattice
        nx = params["n"][1]
        ny = params["n"][2]

        # parameters
        ξ = params["ξ"]
        p = params["p"]
    
        # spd waves
        sw = params["spd"][1]
        pw = params["spd"][2]
        dw = params["spd"][3]

        # calculate the stencil
        stencil = zeros(Complex{Float32}, nx, ny)

        for i = 1:nx
            for j = 1:ny

                # shift the indices
                mod_i = (((i - 1) + nx/2) % nx) - nx/2;
                mod_j = (((j - 1) + ny/2) % ny) - ny/2;

                # x and y position
                x = convert(Float32, mod_i)
                y = convert(Float32, mod_j)
            
                # angular factor
                θ = atan(y,x)
                ang_fac = convert(Float32, sw + pw*cos(θ) + dw*cos(2.0*θ))

                # radial distance
                r = convert(Float32, sqrt(x^2 + y^2));

                # gaussian stencil
                if func == 0

                    p = convert(Float32, 2.0)
                    stencil[i,j] = convert(Complex{Float32}, ang_fac*exp(-(r/ξ)^p))

                # power law stencil
                elseif func == 1

                    stencil[i,j] = convert(Complex{Float32}, ang_fac*r^(-p))

                # RKKY stencil
                elseif func == 2

                    xh = convert(Float32, 2*(r/ξ))
                    stencil[i,j] = convert(Complex{Float32}, ang_fac*(xh^(-4)*(xh*cos(xh) - sin(xh))))

                # otherwise, uniform stencil
                else 
                    stencil[i,j] = convert(Complex{Float32}, 1.0)
                end

            end
        end

        # set self coupling to zero
        stencil[1,1] = convert(Complex{Float32}, 0);

        return stencil

    end

    # lorentzian distribution
    function lorentzian(x, μ, Γ)
        L = (1/π)*(Γ/2)/((x-μ)^2+(Γ/2)^2)
        return L
    end

    export lorentzian
    export make_params, make_temp_params, make_stencil

end