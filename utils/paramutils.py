from itertools import product
from pathlib import Path
import os


# the load order of the variables
ECHO_LOAD_ORDER = ["αx", "αy", "αz", "ξ", "p", "Γ1", "Γ2", "Γ3", "sten", "sw", "pw", "dw", "θ90", "θ180", "p90", "p180"]
SIM_LOAD_ORDER = ["nx", "ny", "dt", "τ", "line_width"]


def make_cuda_parameter_files(params, trial_idx=1, paired=None):

    make_param_file(params, ECHO_LOAD_ORDER, "echo_params", trial_idx=trial_idx, paired=paired)
    make_param_file(params, SIM_LOAD_ORDER, "sim_params", trial_idx=trial_idx, paired=paired)
    return


def make_param_file(params, load_order, filename, trial_idx=1, paired=None):

    # no mutables
    if paired is None:
        paired = []

    # listify everything for iteration
    for key in params:
        if type(params[key]) is not list:
            params[key] = [params[key]]

    # re-structure params to be just those in the load order
    cuda_params = {key: params[key] for key in load_order}

    # pre-prepare the paired variables
    paired_cuda_params = []
    for pairs in paired:
        if all(var in load_order for var in pairs):
            paired_variables = [cuda_params[key] for key in pairs]
            paired_cuda_params.append(list(zip(*paired_variables)))  # they should all be the same length
        elif any(var in load_order for var in paired):
            raise ValueError(f"Paired variables must all be echo params or sim params, not mixed: {pairs}.")

    # product out the remaining variables that aren't paired
    unpaired = [key for key in cuda_params if not any(key in pair for pair in paired)]
    unpaired_cuda_params = list(product(*[cuda_params[key] for key in unpaired]))

    # product the paired and un-paired params
    iterator = list(product(*paired_cuda_params, unpaired_cuda_params))

    # go through the iterator and grab the right variables in order
    # element of itr looks like ((paired 1 values), (paired 2 values), ..., (unpaired values))
    result = []
    for itr in iterator:
        output = []
        for var in load_order:

            # initialize a none for checking
            value = None

            # look if its in a pair
            for pair in paired:
                if var in pair:
                    itr_values = itr[paired.index(pair)]
                    value = itr_values[pair.index(var)]
                    break

            # if we didn't find it, it's in the unpaired variables
            if value is None:
                itr_values = itr[-1]
                value = itr_values[unpaired.index(var)]

            # do some conversions
            if var in ["sten", "p90", "p180", "nx", "ny"]:
                value = int(value)

            # otherewise, round to 8 digits for consistency
            else:
                value = round(value, 8)

            # save
            output.append(str(value))
        result.append(output)

    # print to the output file
    ofile = Path(os.path.join(Path(__file__).parent.parent, "cuda", f"{filename}{trial_idx}.txt"))
    with open(ofile, "w+") as f:
        for line in result:
            f.write(" ".join(line))
            f.write("\n")

    return
