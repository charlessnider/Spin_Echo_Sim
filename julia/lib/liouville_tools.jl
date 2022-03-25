module LiouvilleTools

    using StaticArrays
    using LinearAlgebra


    # column major order for the density matrix
    function dm_H2L(ρ)

        tρ = [ρ[1,1]; ρ[1,2]; ρ[2,1]; ρ[2,2]]

        return tρ

    end

    # column major order for the density matrix
    function dm_L2H(ρ)

        tρ = [ρ[1] ρ[2]; ρ[3] ρ[4]]

        return tρ

    end

    # hand code it the long way to match cuda
    function h_H2L(H)

        # scope
        temp_R = convert(Complex{Float32}, 0.0)
        temp_L = convert(Complex{Float32}, 0.0)

        # output
        H_L = zeros(Complex{Float32}, 4, 4)

        for i = 1:2
            for j = 1:2
                for k = 1:2
                    for ℓ = 1:2

                        # flattened indexing that i no longer understand, but it works
                        b_idx = 4*2*(i-1) + 2*(j-1)
                        s_idx = b_idx + 4*(k-1) + (ℓ-1)

                        # calculate h otimes i
                        if k == ℓ
                            temp_R = H[i,j]
                        else
                            temp_R = convert(Complex{Float32}, 0.0)
                        end

                        # calculate i otimes conj(h)
                        if i == j
                            temp_L = conj(H[k,ℓ])
                        else
                            temp_L = convert(Complex{Float32}, 0.0)
                        end

                        # unflatten the index
                        col_idx = s_idx % 4
                        row_idx = convert(Int64, (s_idx - col_idx)/4)

                        # save
                        H_L[row_idx + 1, col_idx + 1] = temp_R - temp_L

                    end
                end
            end
        end

        return H_L

    end

    # an explicit but nicer way of doing it, not the same as cuda tho
    function ham_H2L(H)

        # output
        H_L = zeros(Complex{Float32}, 4, 4)

        # identity
        I = convert.(Float32, [1 0; 0 1])

        # top left
        H11 = conj(H) - H[1,1]*I

        # top right
        H12 = -H[1,2]*I

        # bottom left
        H21 = -H[2,1]*I

        # bottom right
        H22 = conj(H) - H[2,2]*I

        # fill it out
        H_L[1:2,1:2] = H11
        H_L[1:2,3:4] = H12
        H_L[3:4,1:2] = H21
        H_L[3:4,3:4] = H22

        return H_L

    end

    export dm_H2L, dm_L2H, h_H2L, ham_H2L

end