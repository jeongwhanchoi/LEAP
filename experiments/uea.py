import common
import datasets
import os
import numpy as np
import random
import torch
from parse import parse_args
from random import SystemRandom
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

args = parse_args()

# from tensorboardX import SummaryWriter
FUNC_3OUTPUT_MODELS = ['learnable_func_kg','learnable_func_kf', 'learnable_func_fg']

def main(
    dataset_name=args.dataset_name,
    manual_seed=args.seed,
    missing_rate=args.missing_rate,  # dataset parameters
    device="cuda",
    max_epochs=args.epoch,
    *,  # training parameters
    model_name=args.model,
    hidden_channels=args.h_channels,
    ode_hidden_hidden_channels=args.ode_hidden_hidden_channels,
    hidden_hidden_channels=args.hh_channels,
    num_hidden_layers=args.layers,  # model parameters
    lr=args.lr,
    weight_decay=args.weight_decay,
    c1 = args.c1,
    c2 = args.c2,
    dry_run=False,
    step_mode = args.step_mode,
    **kwargs
):
    # kwargs passed on to cdeint
    np.random.seed(manual_seed)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    # if you are using GPU
    torch.cuda.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    torch.random.manual_seed(manual_seed)

    batch_size = 32
    lr = lr * (batch_size / 32)
    PATH=os.path.dirname(os.path.abspath(__file__))
    intensity_data = True if model_name in ("odernn", "dt", "decay") else False

    (
        times,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        num_classes,
        input_channels,
    ) = datasets.uea.get_data(
        dataset_name,
        missing_rate,
        device,
        intensity=intensity_data,
        batch_size=batch_size,
    )

    if num_classes == 2:
        output_channels = 1
    else:
        output_channels = num_classes
    experiment_id = int(SystemRandom().random()*100000)
    file = PATH+'/h_0/'+f'uea_{experiment_id}.npy'
    folder_name = 'UEA'
    test_name = "step_" + "_".join([ str(j) for i,j in dict(vars(args)).items()]) + "_" + str(experiment_id)
    result_folder = PATH+'/tensorboard_test'
    make_model = common.make_model(
        model_name,
        input_channels,
        output_channels,
        hidden_channels,
        hidden_hidden_channels,
        ode_hidden_hidden_channels, 
        num_hidden_layers,
        file,
        use_intensity=False,
        initial=True,
    )

    def new_make_model():
        
        model, regularise = make_model()
        model.linear.weight.register_hook(lambda grad: 100 * grad)
        model.linear.bias.register_hook(lambda grad: 100 * grad)
        return model, regularise

        
        

    
    if dry_run:
        name = None
    else:
        name = dataset_name + str(int(missing_rate * 100))
    
    return common.main(
        name,
        model_name,
        times,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        device,
        new_make_model,
        num_classes,
        max_epochs,
        lr,
        weight_decay,
        file,
        kwargs,
        step_mode=step_mode,
        c1=c1,
        c2=c2
    )


if __name__ == "__main__":
    main(method = args.method)