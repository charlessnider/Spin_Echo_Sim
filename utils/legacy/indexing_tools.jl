# for creating parameter files
using Distributions
using Formatting
using DelimitedFiles

function make_parameter_file_julia(f, vars, fname_idx, include_idx)
    
    # specify formatting
    fspec = FormatSpec(".8f")
    
    # load order (MUST MATCH ORDER OF PARAMETER FILE)
    # | αx | αy | αz | ξ | p | Γ1 | Γ2 | Γ3 | stencil form | nx | ny | dt | τ | sw | pw | dw | θ | line_width | #
    ld_ord = ["αx", "αy", "αz", "ξ", "p", "Γ1", "Γ2", "Γ3", "sten", "nx", "ny", "dt", "τ", "sw", "pw", "dw", "θ", "line_width"]
    
    if !isempty(vars)
    
        # get dimensions of each variable
        d = ()
        for v in vars
            # if a tuple, just get the first element for the dimension (idk why but putting "String" wasn't working)
            if typeof(v) != typeof("a")
                v = v[1]
            end
            d = (d..., length(f[v]))
        end

        # create the indexing
        λ = sample(1:prod(d), d, replace = false)
        I = reshape([findall(x -> x == temp, λ)[1] for temp in λ], prod(d))
        
        # get the name
        fname = "params.txt"
        if include_idx
            fname = "params"*string(fname_idx)*".txt"
        end
        
        # open the file
        open(fname, "w") do io
        
            # loop until have lines = number of trials = prod(d)
            t_idx = 1
            while t_idx <= prod(d)

                # this trial's indexing
                tI = I[t_idx]

                # string to write to file
                str = ""

                # loop over load order
                for x in ld_ord
                    
                    # bool cuz we have to be round-about here
                    is_var = false
                    var_idx = 1
                    
                    # loop over variables, look for matches
                    for v in vars
                        
                        # if v is a tuple, loop over elements of tuple
                        if typeof(v) != typeof("a")
                            for t_v in v
                                if x == t_v
                                    is_var = true
                                    var_idx = findall(y -> y == v, vars)[1]
                                end
                            end
                        elseif x == v
                            is_var = true
                            var_idx = findall(y -> y == v, vars)[1]
                        end
                        
                    end

                    # if one of the variables, assign the appropriate index from tI
                    if is_var
                        t_val = f[x][tI[var_idx]]
                        if x == "line_width"
                            t_val *= 2
                        end
                        str *= string(fmt(fspec, t_val))
                    else
                        t_val = f[x][1]
                        if x == "line_width"
                            t_val *= 2
                        end
                        str *= string(fmt(fspec, t_val))
                    end
                    
                    # if the last in load order, add a line break-- else, a space
                    if x == ld_ord[end]
                        str *= "\n"
                    else
                        str *= " "
                    end
                    
                end
                
                # write to the file
                write(io, str)
            
                # increment t_idx
                t_idx += 1
            
            end
            
        end
        
    else
        
        # dimension
        d = (1)
        
        # get the name
        fname = "params.txt"
        if include_idx
            fname = "params"*string(fname_idx)*".txt"
        end
        
        # open the file
        open(fname, "w") do io

            # string to write to file
            str = ""

            # loop over load order
            for x in ld_ord

                # write the variable
                str *= string(fmt(fspec, f[x][1]))

                # if the last in load order, add a line break-- else, a space
                if x == ld_ord[end]
                    str *= "\n"
                else
                    str *= " "
                end

                # write to the file
                write(io, str)

            end
            
        end
        
    end
    
    return d
    
end                    

function make_parameter_file_cuda(f, echo_vars, sim_vars, fname_idx, include_idx, dir_name)
    
    # specify formatting
    fspec = FormatSpec(".8f")
    
    ##### CREATE ECHO PARAMS #####
    
    # load order (MUST MATCH ORDER OF PARAMETER FILE)
    # | αx | αy | αz | ξ | p | Γ1 | Γ2 | Γ3 | stencil form | sw | pw | dw | θ90 | θ180 | phase90 | phase180 | #
    ld_ord = ["αx", "αy", "αz", "ξ", "p", "Γ1", "Γ2", "Γ3", "sten", "sw", "pw", "dw", "θ90", "θ180", "phase90", "phase180"]
    
    if !isempty(echo_vars)
    
        # get dimensions of each variable
        d = ()
        for v in echo_vars
            # if a tuple, just get the first element for the dimension (idk why but putting "String" wasn't working)
            if typeof(v) != typeof("a")
                v = v[1]
            end
            d = (d..., length(f[v]))
        end

        # create the indexing
        λ = sample(1:prod(d), d, replace = false)
        I = reshape([findall(x -> x == temp, λ)[1] for temp in λ], prod(d))
        
        # get the name
        fname = dir_name*"echo_params.txt"
        if include_idx
            fname = dir_name*"echo_params"*string(fname_idx)*".txt"
        end
        
        # open the file
        open(fname, "w") do io
        
            # loop until have lines = number of trials = prod(d)
            t_idx = 1
            while t_idx <= prod(d)

                # this trial's indexing
                tI = I[t_idx]

                # string to write to file
                str = ""

                # loop over load order
                for x in ld_ord
                    
                    # bool cuz we have to be round-about here
                    is_var = false
                    var_idx = 1
                    
                    # loop over variables, look for matches
                    for v in echo_vars
                        
                        # if v is a tuple, loop over elements of tuple
                        if typeof(v) != typeof("a")
                            for t_v in v
                                if x == t_v
                                    is_var = true
                                    var_idx = findall(y -> y == v, echo_vars)[1]
                                end
                            end
                        elseif x == v
                            is_var = true
                            var_idx = findall(y -> y == v, echo_vars)[1]
                        end
                        
                    end

                    # if one of the variables, assign the appropriate index from tI
                    if x in ["sten", "phase90", "phase180"] # have to do sten and phases separately since integers
                    
                        if is_var
                            str *= string(f[x][tI[var_idx]])
                        else
                            str *= string(f[x][1])
                        end
                    
                    else
                        
                        if is_var
                            str *= string(fmt(fspec, f[x][tI[var_idx]]))
                        else
                            str *= string(fmt(fspec, f[x][1]))
                        end
                        
                    end
                        
                    # if the last in load order, add a line break-- else, a space
                    if x == ld_ord[end]
                        str *= "\n"
                    else
                        str *= " "
                    end
                    
                end
                
                # write to the file
                write(io, str)
            
                # increment t_idx
                t_idx += 1
            
            end
            
        end
        
    else
        
        # get the name
        fname = dir_name*"echo_params.txt"
        if include_idx
            fname = dir_name*"echo_params"*string(fname_idx)*".txt"
        end
        
        # open the file
        open(fname, "w") do io

            # string to write to file
            str = ""

            # loop over load order
            for x in ld_ord

                if x in ["sten", "phase90", "phase180"] # have to do sten and phases separately since integers
                    
                    # write the variable
                    str *= string(f[x][1])
                    
                else
                    
                    # write the variable
                    str *= string(fmt(fspec, f[x][1]))                    
                    
                end
                


                # if the last in load order, add a line break-- else, a space
                if x == ld_ord[end]
                    str *= "\n"
                else
                    str *= " "
                end

            end
            
            # write to the file
            write(io, str)
            
        end
        
    end
    
    ##### CREATE SIM PARAMS #####
    
    # load order
    # | nx | ny | dt | τ | line_width | #
    ld_ord = ["nx", "ny", "dt", "τ", "line_width"]
    
    if !isempty(sim_vars)
    
        # get dimensions of each variable
        d = ()
        for v in sim_vars
            # if a tuple, just get the first element for the dimension (idk why but putting "String" wasn't working)
            if typeof(v) != typeof("a")
                v = v[1]
            end
            d = (d..., length(f[v]))
        end

        # create the indexing
        λ = sample(1:prod(d), d, replace = false)
        I = reshape([findall(x -> x == temp, λ)[1] for temp in λ], prod(d))
        
        # get the name
        fname = dir_name*"sim_params.txt"
        if include_idx
            fname = dir_name*"sim_params"*string(fname_idx)*".txt"
        end
        
        # open the file
        open(fname, "w") do io
        
            # loop until have lines = number of trials = prod(d)
            t_idx = 1
            while t_idx <= prod(d)

                # this trial's indexing
                tI = I[t_idx]

                # string to write to file
                str = ""

                # loop over load order
                for x in ld_ord
                    
                    # bool cuz we have to be round-about here
                    is_var = false
                    var_idx = 1
                    
                    # loop over variables, look for matches
                    for v in sim_vars
                        
                        # if v is a tuple, loop over elements of tuple
                        if typeof(v) != typeof("a")
                            for t_v in v
                                if x == t_v
                                    is_var = true
                                    var_idx = findall(y -> y == v, sim_vars)[1]
                                end
                            end
                        elseif x == v
                            is_var = true
                            var_idx = findall(y -> y == v, sim_vars)[1]
                        end
                        
                    end

                    # if one of the variables, assign the appropriate index from tI
                    if x in ["nx", "ny"]
                        
                        if is_var
                            str *= string(f[x][tI[var_idx]])
                        else
                            str *= string(f[x][1])
                        end

                        # if the last in load order, add a line break-- else, a space
                        if x == ld_ord[end]
                            str *= "\n"
                        else
                            str *= " "
                        end
                        
                    else
                        
                        if is_var
                            str *= string(fmt(fspec, f[x][tI[var_idx]]))
                        else
                            str *= string(fmt(fspec, f[x][1]))
                        end

                        # if the last in load order, add a line break-- else, a space
                        if x == ld_ord[end]
                            str *= "\n"
                        else
                            str *= " "
                        end
                        
                    end
                    
                end
                
                # write to the file
                write(io, str)
            
                # increment t_idx
                t_idx += 1
            
            end
            
        end
        
    else
        
        # get the name
        fname = dir_name*"sim_params.txt"
        if include_idx
            fname = dir_name*"sim_params"*string(fname_idx)*".txt"
        end
        
        # open the file
        open(fname, "w") do io

            # string to write to file
            str = ""

            # loop over load order
            for x in ld_ord

                # write the variable
                if x in ["nx", "ny"]
                    str *= string(f[x][1])
                else
                    str *= string(fmt(fspec, f[x][1]))
                end

                # if the last in load order, add a line break-- else, a space
                if x == ld_ord[end]
                    str *= "\n"
                else
                    str *= " "
                end

            end

            # write to the file
            write(io, str)
            
        end
        
    end
    
end                    