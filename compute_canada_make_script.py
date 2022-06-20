import os
import itertools
import argparse


dir_jobs = "jobs/todo"
params = []

def make_jobs_script(params, dir_jobs):
    if not os.path.exists(dir_jobs):
        os.makedirs(dir_jobs)

    header = "#!/bin/bash" + "\n"
    header += "module load cuda scipy-stack python/3" + "\n"
    header += "source /home/$USER/juho-usra/bin/activate" + "\n"
    header += "cd /home/$USER/scratch/$USER/juho-usra" + "\n"
    num_jobs = 0
    for param in params:
        values = list(param.values())
        keys = list(param.keys())
        for com in itertools.product(*values):
            hp = []
            for key, valuei in zip(keys, com):
                hp += [f"--{key}={valuei}"]
            hp += [f"--case_num={num_jobs}"]
            command = "python3 main.py " + " ".join(hp)
            fn = f"jobs{num_jobs}.sh"
            fn = os.path.join(dir_jobs, fn)
            with open(fn, "w") as f:
                f.write(header)
                f.write(command)
            
            num_jobs += 1


if __name__ == "__main__":

    ###### how to run this program ########

    # python3 compute_canada_make_script.py --function=<d or sr> --w0_lowerbound=<w0 parameter lowerbound in float> --w0_upperbound=<w0 parameter upperbound in float>
    # --w0_scale_factor=<w0 scale factor in float> --learning_rate_lowerbound=<learning rate lowerbound in float> --learning_rate_upperbound=<learning rate upperbound in float>
    # --learning_rate_scale_factor=<learning rate scale factor infloat>

    #######################################

    parser = argparse.ArgumentParser()
    parser.add_argument('--function', type=str,
                       default='b',
                       help="d for denoising and sr for super_resolution and b for both")
    parser.add_argument('--w0_lowerbound', type=float,
                       default=30.,
                       help="lowerbound of the w0 parameter for Siren.")
    parser.add_argument('--w0_upperbound', type=float,
                       default=30.,
                       help="upperbound of the w0 parameter for Siren.")
    parser.add_argument('--w0_scale_factor', type=float,
                       default=30.,
                       help="scale factor of the w0 parameter for Siren.")
    parser.add_argument('--learning_rate_lowerbound', type=float,
                       default=0.01,
                       help="lowerbound of the learning rate for Siren") 
    parser.add_argument('--learning_rate_upperbound', type=float,
                       default=0.01,
                       help="upperbound of the learning rate for Siren")   
    parser.add_argument('--learning_rate_scale_factor', type=float,
                       default=0.01,
                       help="scale factor of the learning rate for Siren") 
    args = parser.parse_args()
    function_name = args.function
    w0_lowerbound = args.w0_lowerbound
    w0_upperbound = args.w0_upperbound
    w0_scale_factor = args.w0_scale_factor
    learning_rate_lowerbound = args.learning_rate_lowerbound
    learning_rate_upperbound = args.learning_rate_upperbound
    learning_rate_scale_factor = args.learning_rate_scale_factor

    param = {}

    function = []
    if function_name == 'b':
        function.append("d")
        function.append("sr")
    else:
        function.append(function_name)
    param.update({"function": function})

    w0 = []
    while w0_lowerbound < w0_upperbound:
        w0.append(w0_lowerbound)
        w0_lowerbound *= w0_scale_factor
    param.update({"w0": w0})

    learning_rate = []
    while learning_rate_lowerbound < learning_rate_upperbound:
        learning_rate.append(learning_rate_lowerbound)
        learning_rate_lowerbound *= learning_rate_scale_factor
    param.update({"learning_rate": learning_rate})

    params.append(param)

    make_jobs_script(params, dir_jobs)