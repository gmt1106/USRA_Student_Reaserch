import argparse
from denoising import sirenTrain as denoising_sirenTrain
from super_resolution import sirenTrain as super_resolution_sirenTrain


def main(function_name, w0, leanrning_rate, case_num):
    """The main function."""

    if function_name == 'd':
        denoising_sirenTrain(w0, leanrning_rate, case_num)
    elif function_name == 'sr':
        super_resolution_sirenTrain(w0, leanrning_rate, case_num)

if __name__ == "__main__":

    ###### how to run this program ########

    # python main.py --function=<d or sr> --w0=<w0 parameters in float> --learning_rate=<learning rate in float> 
    # --case_num=<compute canada script case number>

    #######################################

    parser = argparse.ArgumentParser()
    parser.add_argument('--function', type=str,
                       default='d',
                       help="d for denoising and sr for super_resolution")
    parser.add_argument('--w0', type=float,
                       default=30.,
                       help="w0 parameter for Siren")
    parser.add_argument('--learning_rate', type=float,
                       default=0.01,
                       help="learning rate for Siren")   
    parser.add_argument('--case_num', type=str,
                       default=0.01,
                       help="compute canada script case number")                    
    args = parser.parse_args()
    function_name = args.function
    w0 = args.w0
    leanrning_rate = args.learning_rate
    case_num = args.case_num

    main(function_name, w0, leanrning_rate, case_num)
