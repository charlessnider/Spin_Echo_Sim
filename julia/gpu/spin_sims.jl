include("../lib/liouville_tools.jl")
using .LiouvilleTools
using LinearAlgebra
using CUDA
using StatsBase

function spin_echo_sim_liouville(params)   
    
    # initialize M_list
    M_list = [];
    Mz_list = [];
    
    # get left and right π/2 pulse operators
    UL90 = params["U90"]
    UR90 = UL90'
    
    # get left and right π pulse operators
    UL180 = params["U180"]
    UR180 = UL180'
    
    # load initial parameters
    ψ_list = [ψ for ψ in params["ψ_init"]]
    ρ_list = [ψ*ψ' for ψ in ψ_list]
    
    # pulse in hilbert space
    ρ_list = [UL90*ρ*UR90 for ρ in ρ_list]
    
    # bump to liouville space
    ρ_list_L = [dm_H2L(ρ) for ρ in ρ_list]
    
    # first tau
    t0 = convert(Float32, 0.0);
    ρ_list_L, M_list, Mz_list, t1, params = time_propagate_liouville(ρ_list_L, M_list, Mz_list, t0, params["dt"], params["nτ"], params)
    
    # 180 pulse
    ρ_list_L = [dm_H2L(UL180 * dm_L2H(ρ_L) * UR180) for ρ_L in ρ_list_L];
        
    # second tau
    ρ_list_L, M_list, Mz_list, t2, params = time_propagate_liouville(ρ_list_L, M_list, Mz_list, t1, params["dt"], 2*params["nτ"], params)
    
  return M_list, Mz_list, params

end

function time_propagate_liouville(ρ_list_L, M_list, Mz_list, t0, dt, nsteps, params) 
            
    # spectrum info
    ν0 = params["ν0"]
    ν = params["ν"]
    nS = params["nfreq"]
    
    # operators
    M_op = params["M_op"]
    Iz = params["Iz"]
    
    # additional values
    n = params["n"]
    nx = n[1]
    ny = n[2]
    P = convert(Complex{Float32}, 1/nS)
    
    # interaction parameters
    αx = params["αx"]
    αy = params["αy"]
    αz = params["αz"]
    
    # initial time
    t = t0;

    # jump operators
    J_L = params["J"]
  
    # initial magnetization
    t_t = @elapsed begin
        M_eval = [convert(Complex{Float32}, ρ[3]) for ρ in ρ_list_L]
    end
    push!(params["profiling"]["M_eval"], t_t)
    
    t_t = @elapsed begin
        Mz_eval = [(convert(Complex{Float32}, 0.5))*(ρ[1] - ρ[4]) for ρ in ρ_list_L]
    end
    push!(params["profiling"]["Mz_eval"], t_t)
    
    t_t = @elapsed begin
        M = sum(P.*M_eval)
    end
    push!(params["profiling"]["M"], t_t)
    
    t_t = @elapsed begin
        Mz = sum(P.*Mz_eval)
    end
    push!(params["profiling"]["Mz"], t_t)
    
    push!(M_list, M)
    push!(Mz_list, Mz)
    
    # move to GPU
    t_t = @elapsed begin
        M_eval = CuArray(M_eval)
    end
    push!(params["profiling"]["M_eval_to_CuArray"], t_t)
    
    t_t = @elapsed begin
        Mz_eval = CuArray(Mz_eval)
    end
    push!(params["profiling"]["Mz_eval_to_CuArray"], t_t)
    
    # prepare the stencils
    M_stencil = params["M_stencil"]
    t_t = @elapsed begin
        M_stencil_vec = shift_stencil(params)
    end
    push!(params["profiling"]["M_stencil_vec"], t_t)

    # calculate the local magnetization (planar)
    t_t = @elapsed begin
        M_eval = repeat(M_eval, 1, 1, nx, ny)
        M_eval .*= M_stencil_vec
        M_local = mapreduce(identity, +, M_eval; dims = [1,2])
        M_local = dropdims(M_local, dims = (1,2))
    end
    push!(params["profiling"]["M_local"], t_t)
    
    # calculate local magnetization (z)
    t_t = @elapsed begin
        Mz_eval = repeat(Mz_eval, 1, 1, nx, ny)
        Mz_eval .*= M_stencil_vec
        Mz_local = mapreduce(identity, +, Mz_eval; dims = [1,2])
        Mz_local = dropdims(Mz_local, dims = (1,2))
    end
    push!(params["profiling"]["Mz_local"], t_t)
    
    # time evolve
    for idx = 1:nsteps
        
        # convert to array type
        t_t = @elapsed begin
            M_local = Array(M_local)
        end
        push!(params["profiling"]["M_local_to_Array"], t_t)
        
        t_t = @elapsed begin
            Mz_local = Array(Mz_local)
        end
        push!(params["profiling"]["Mz_local_to_Array"], t_t)

        # calculate hamiltonian (straight up loop is faster than more aesthetic structures)
        t_t = @elapsed begin
            H_H = Array{Any}(undef, nx, ny)
            for i = 1:nx
                for j = 1:ny

                    # load local mags
                    t_Mp = M_local[i,j]
                    t_Mz = Mz_local[i,j]

                    # temp hamiltonian
                    tH = zeros(Complex{Float32}, 2, 2)

                    # hard-code hamiltonian
                    tH[1,1] = convert(Complex{Float32}, (1/2)*(ν0 - ν[i,j])) - convert(Complex{Float32}, (αz/2)*t_Mz)
                    tH[1,2] = -convert(Complex{Float32}, (αy/4)*(conj(t_Mp) - t_Mp*exp(-2im*ν0*t))) - convert(Complex{Float32}, (αx/4)*(conj(t_Mp) + t_Mp*exp(-2im*ν0*t)))
                    tH[2,1] = -convert(Complex{Float32}, (αy/4)*(t_Mp - conj(t_Mp)*exp(2im*ν0*t))) - convert(Complex{Float32}, (αx/4)*(t_Mp + conj(t_Mp)*exp(2im*ν0*t)))
                    tH[2,2] = convert(Complex{Float32}, (1/2)*(ν[i,j] - ν0)) + convert(Complex{Float32}, (αz/2)*t_Mz)

                    # save to big array
                    H_H[i,j] = tH

                end
            end
        end
        push!(params["profiling"]["calc_H"], t_t)        
        
        # calculate hamiltonian (convert to float64: exp more accurate in float64)
        t_t = @elapsed begin
            H_L = [convert.(Complex{Float64}, h_H2L(H)) for H in H_H]
        end
        push!(params["profiling"]["H_to_Liouville"], t_t)
        
        # calculate propagators, adding in dissipation & then convert back to float32
        t_t = @elapsed begin
            # U_L = [convert.(Complex{Float32}, exp(( -1im*H + J_L )*dt)) for H in H_L]
            U_L = Array{Any}(undef, nx, ny)
            for i = 1:nx
                for j = 1:ny
                    
                    U_L[i,j] = convert.(Complex{Float32}, exp(( -1im*H_L[i,j] + J_L )*dt))
                    
                end
            end
        end
        push!(params["profiling"]["calc_U"], t_t)
        
        # time evolve
        t_t = @elapsed begin
            ρ_list_L = U_L.*ρ_list_L
        end
        push!(params["profiling"]["time_evolve"], t_t)

        # magnetization    
        t_t = @elapsed begin
            M_eval = [convert(Complex{Float32}, ρ[3]) for ρ in ρ_list_L]
        end
        push!(params["profiling"]["M_eval"], t_t)
        
        t_t = @elapsed begin
            Mz_eval = [(convert(Complex{Float32}, 0.5))*(ρ[1] - ρ[4]) for ρ in ρ_list_L]
        end
        push!(params["profiling"]["Mz_eval"], t_t)
        
        t_t = @elapsed begin
            M = sum(P.*M_eval)
        end
        push!(params["profiling"]["M"], t_t)
        
        t_t = @elapsed begin
            Mz = sum(P.*Mz_eval)
        end
        push!(params["profiling"]["Mz"], t_t)
        
        push!(M_list, M)
        push!(Mz_list, Mz)

        # move to GPU
        t_t = @elapsed begin
            M_eval = CuArray(M_eval)
        end
        push!(params["profiling"]["M_eval_to_CuArray"], t_t)
        
        t_t = @elapsed begin
            Mz_eval = CuArray(Mz_eval)
        end
        push!(params["profiling"]["Mz_eval_to_CuArray"], t_t)
        
        # calculate the local magnetization (planar)
        t_t = @elapsed begin
            M_eval = repeat(M_eval, 1, 1, nx, ny)
            M_eval .*= M_stencil_vec
            M_local = mapreduce(identity, +, M_eval; dims = [1,2])
            M_local = dropdims(M_local, dims = (1,2))
        end
        push!(params["profiling"]["M_local"], t_t)

        # calculate local magnetization (z)
        t_t = @elapsed begin
            Mz_eval = repeat(Mz_eval, 1, 1, nx, ny)
            Mz_eval .*= M_stencil_vec
            Mz_local = mapreduce(identity, +, Mz_eval; dims = [1,2])
            Mz_local = dropdims(Mz_local, dims = (1,2))
        end
        push!(params["profiling"]["Mz_local"], t_t)
        
        # advance time
        t += dt;

    end
        
    return ρ_list_L, M_list, Mz_list, t, params
    
end

function shift_stencil(params)

    # dimension of mesh
    nx = params["n"][1]
    ny = params["n"][2]
        
    # real quick make the spin indices
    spin_idx = Array{Any}(undef, nx, ny)
    for i = 1:nx
        for j = 1:nx
            spin_idx[i,j] = (i,j)
        end
    end
    
    # load the stencil and prepare big stencil
    t_stencil = params["M_stencil"]
    stencil = CuArray(t_stencil)
    bigS = CUDA.zeros(Complex{Float32}, nx, ny, nx, ny)

    # do the shift
    for v in spin_idx
        shift = (spin_idx[v...][1] - 1, spin_idx[v...][2] - 1)
        temp = circshift(stencil, shift)
        bigS[:,:,v[1],v[2]] = temp
    end

    return bigS

end   