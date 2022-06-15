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
    # header += "eval \"$(conda shell.bash hook)\"" + "\n"
    # header += "conda activate pcd" + "\n"

    header += "cd /home/$USER/scratch/juho-usra/train" + "\n"
    num_jobs = 0
    for param in params:
        values = list(param.values())
        keys = list(param.keys())
        for com in itertools.product(*values):
            hp = []
            for key, valuei in zip(keys, com):
                hp += [f"--{key}={valuei}"]
            
            command = "python3 main.py " + " ".join(hp)
            fn = f"jobs{num_jobs}.sh"
            fn = os.path.join(dir_jobs, fn)
            with open(fn, "w") as f:
                f.write(header)
                f.write(command)
            
            num_jobs += 1


if __name__ == "__main__":

    ###### how to run this program ########

    # python main.py --function 'd or sr' --w0_lowerbound 'w0 parameter lowerbound in float' --w0_upperbound 'w0 parameter upperbound in float'
    # --w0_step 'w0 step' --learning_rate_lowerbound 'learning rate lowerbound in float' --learning_rate_upperbound 'learning rate upperbound in float'
    # --learning_rate_step 'learning rate step'

    #######################################

    parser = argparse.ArgumentParser()
    parser.add_argument('--function', type=str,
                       default='d',
                       help="d for denoising and sr for super_resolution")
    parser.add_argument('--w0_lowerbound', type=float,
                       default=30.,
                       help="lowerbound of the w0 parameter for Siren.")
    parser.add_argument('--w0_upperbound', type=float,
                       default=30.,
                       help="upperbound of the w0 parameter for Siren.")
    parser.add_argument('--w0_step', type=float,
                       default=30.,
                       help="step of the w0 parameter for Siren.")
    parser.add_argument('--learning_rate_lowerbound', type=float,
                       default=0.01,
                       help="lowerbound of the learning rate for Siren") 
    parser.add_argument('--learning_rate_upperbound', type=float,
                       default=0.01,
                       help="upperbound of the learning rate for Siren")   
    parser.add_argument('--learning_rate_step', type=float,
                       default=0.01,
                       help="step of the learning rate for Siren") 
    args = parser.parse_args()
    function_name = args.function
    w0_lowerbound = args.w0_lowerbound
    w0_upperbound = args.w0_upperbound
    w0_step = args.w0_step
    learning_rate_lowerbound = args.learning_rate_lowerbound
    learning_rate_upperbound = args.learning_rate_upperbound
    learning_rate_step = args.learning_rate_step

    param = {
        "function": [function_name],
        "w0": [1],
        "learning_rate": ["uni-180-0.2"],
    }

    w0 = []
    while w0_lowerbound < w0_upperbound:
        w0.append(w0_lowerbound)
        w0_lowerbound = w0_lowerbound + w0_step
    param.update({"w0": w0})

    learning_rate = []
    while learning_rate_lowerbound < learning_rate_upperbound:
        learning_rate.append(learning_rate_lowerbound)
        learning_rate_lowerbound = learning_rate_lowerbound + learning_rate_step
    param.update({"learning_rate": learning_rate})

    params.append(param)

    make_jobs_script(params, dir_jobs)