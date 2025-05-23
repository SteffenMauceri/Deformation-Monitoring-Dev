import glob
import os
import time
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from argparse import ArgumentParser
import random
import matplotlib.pyplot as plt

# user
from builders.model_builder import build_model
from builders.dataset_builder import build_dataset_test
from dataset.phaseUnwrapping import denormalize_label, denormalize_input # Import denormalization functions


def parse_args():
    parser = ArgumentParser(description='PUNet Test Script with Visualization')
    parser.add_argument('--model', type=str, default="PUNet", help="model name: (default PUNet)")
    parser.add_argument('--dataRootDir', type=str,
                        default=r"dataset",
                        help="dataset dir")
    parser.add_argument('--dataset', default="phaseUnwrapping", help="dataset")
    parser.add_argument('--num_workers', type=int, default=4, help="the number of parallel threads") # Adjusted default from 8 to 4 like in train
    parser.add_argument('--batch_size', type=int, default=1,
                        help=" the batch_size is set to 1 when evaluating or testing")
    parser.add_argument('--checkpoint', type=str,default="",
                        help="use the file to load the checkpoint for evaluating or testing ")
    parser.add_argument('--cuda', default=True, type=lambda x: (str(x).lower() == 'true'), help="run on CPU or GPU")
    parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
    parser.add_argument('--save_viz_dir', type=str, default="test_visualizations", help="directory to save visualization plots")
    args = parser.parse_args()
    return args


def save_comparison_plots(samples_data, title_prefix, base_save_dir, num_samples=6):
    if not samples_data:
        print(f"No data provided for {title_prefix} plots.")
        return

    os.makedirs(base_save_dir, exist_ok=True)
    plot_filename = os.path.join(base_save_dir, f"{title_prefix.lower().replace(' ', '_')}_comparison.png")

    num_actual_samples = min(len(samples_data), num_samples)
    if num_actual_samples == 0:
        print(f"Not enough samples for {title_prefix} plots (found 0).")
        return
        
    fig, axes = plt.subplots(num_actual_samples, 4, figsize=(20, 5 * num_actual_samples))
    if num_actual_samples == 1: # Ensure axes is always 2D
        axes = axes.reshape(1, -1)

    fig.suptitle(f"{title_prefix} (Top {num_actual_samples} Samples)", fontsize=16)

    for i in range(num_actual_samples):
        sample = samples_data[i]
        name = sample['name']
        input_denorm = sample['input_denorm']
        output_denorm = sample['output_denorm']
        truth_denorm = sample['truth_denorm']
        diff_map = sample['diff_map']
        sample_rmse = sample['sample_rmse']

        # Squeeze out channel dimension if present (e.g., (1, H, W) -> (H, W))
        if input_denorm.ndim == 3 and input_denorm.shape[0] == 1:
            input_denorm = input_denorm.squeeze(0)
        if output_denorm.ndim == 3 and output_denorm.shape[0] == 1:
            output_denorm = output_denorm.squeeze(0)
        if truth_denorm.ndim == 3 and truth_denorm.shape[0] == 1:
            truth_denorm = truth_denorm.squeeze(0)
        if diff_map.ndim == 3 and diff_map.shape[0] == 1:
            diff_map = diff_map.squeeze(0)

        # Plot Input
        ax = axes[i, 0]
        im = ax.imshow(input_denorm, cmap='gray')
        ax.set_title(f"Input: {name}")
        ax.axis('off')
        # fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


        # Plot Ground Truth
        ax = axes[i, 1]
        im = ax.imshow(truth_denorm, cmap='viridis') # viridis is good for phase
        ax.set_title(f"Truth (RMSE: {sample_rmse:.4f})")
        ax.axis('off')
        # fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Plot Model Output
        ax = axes[i, 2]
        im = ax.imshow(output_denorm, cmap='viridis')
        ax.set_title("Model Output")
        ax.axis('off')
        # fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Plot Difference
        ax = axes[i, 3]
        im = ax.imshow(diff_map, cmap='coolwarm', vmin=-np.abs(diff_map).max(), vmax=np.abs(diff_map).max()) # Center colorbar
        ax.set_title("Difference (Output - Truth)")
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to make space for suptitle
    plt.savefig(plot_filename)
    plt.close(fig)
    print(f"Saved plot: {plot_filename}")


def test(args, test_loader, model, device):
    """
    args:
      test_loader: loaded for test dataset
      model: model
      device: torch device
    return: list of result dictionaries
    """
    output_save_dir = os.path.join(args.dataRootDir, args.dataset, args.model + "_predictions")
    os.makedirs(output_save_dir, exist_ok=True)

    model.eval()
    total_batches = len(test_loader)
    results_data = []

    for i, batch_data in enumerate(test_loader):
        # Unpack based on whether ground truth is present or not
        if len(batch_data) == 4: # Assuming (input, label, size, name)
            input_tensor, label_tensor, size, name = batch_data
            has_truth = True
        elif len(batch_data) == 3: # Assuming (input, size, name) from PhaseUnwrappingTestDataSet
            input_tensor, size, name = batch_data
            label_tensor = None # No ground truth
            has_truth = False
        else:
            raise ValueError(f"Unexpected number of items in batch_data: {len(batch_data)}")

        with torch.no_grad():
            input_var = input_tensor.to(device)
        
        prediction_save_file = os.path.join(output_save_dir, name[0]) # name is a tuple/list
        
        start_time = time.time()
        output = model(input_var)
        if device.type != 'cpu':
            torch.cuda.synchronize() if device.type == 'cuda' else torch.mps.synchronize()
        time_taken = time.time() - start_time
        
        print(f'Testing: [{i + 1}/{total_batches}]  time: {time_taken:.2f}s')
        
        # Process output
        output_numpy = output.cpu().data[0].numpy() # Assuming batch_size is 1 for testing
        output_numpy.tofile(prediction_save_file) # Save raw prediction

        # Denormalize for metrics and visualization
        # Input is normalized by dataloader, denormalize it for viz
        input_denorm_np = denormalize_input(input_tensor.cpu().data[0].numpy())
        output_denorm_np = denormalize_label(output_numpy)

        current_result = {
            "name": name[0],
            "input_denorm": input_denorm_np,
            "output_denorm": output_denorm_np,
            "sample_rmse": float('inf'), # Default if no truth
            "truth_denorm": None,
            "diff_map": None
        }

        if has_truth and label_tensor is not None:
            # label_tensor is already normalized by dataloader
            truth_denorm_np = denormalize_label(label_tensor.cpu().data[0].numpy())
            diff_map_np = output_denorm_np - truth_denorm_np
            sample_rmse = np.sqrt(np.mean(diff_map_np ** 2))
            
            current_result["truth_denorm"] = truth_denorm_np
            current_result["diff_map"] = diff_map_np
            current_result["sample_rmse"] = sample_rmse
        
        results_data.append(current_result)

    return results_data


def test_model(args):
    """
     main function for testing
     param args: global arguments
     return: None
    """
    print(args)

    # Setup device
    device_str = "cpu"
    if args.cuda:
        if torch.cuda.is_available():
            print("=====> use gpu id: '{}'".format(args.gpus))
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
            device_str = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("=====> using Apple Metal Performance Shaders (MPS)")
            device_str = "mps"
        else:
            print("=====> CUDA/MPS not available, falling back to CPU")
    else:
        print("=====> Running on CPU")
    device = torch.device(device_str)


    # build the model
    model = build_model(args.model, num_channels=1)
    model = model.to(device)

    if device.type != 'cpu':
        cudnn.benchmark = True # Only if using CUDA/MPS

    # load the test set
    # build_dataset_test expects input_size as a tuple, not string.
    # For now, we assume the default (256,256) from dataset/phaseUnwrapping.py is used
    # or that Punet.py's test block correctly sets a compatible size if it's different.
    # If using --input_size in test.py, it should be parsed to tuple.
    # The dataset loader (PhaseUnwrappingDataSet used by build_dataset_test) normalizes inputs and labels.
    datas, testLoader = build_dataset_test(args.dataRootDir, args.dataset, args.num_workers, none_gt=False) # none_gt=False means we expect ground truth

    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            print("=====> loading checkpoint '{}'".format(args.checkpoint))
            checkpoint = torch.load(args.checkpoint, map_location=device) # Ensure loading to correct device
            model.load_state_dict(checkpoint['model'])
        else:
            print("=====> no checkpoint found at '{}'".format(args.checkpoint))
            raise FileNotFoundError("no checkpoint found at '{}'".format(args.checkpoint))
    else: # Try to load latest model from default checkpoint dir
        try:
            # Adjust path to match how train.py saves checkpoints (includes device type in path)
            # This part is tricky as savedir in train.py depends on device.
            # We'll try a common pattern first, may need adjustment if savedir is very custom.
            # Example: ./checkpoint/phaseUnwrapping/PUNetbs6mps1/
            # For simplicity, let's assume a generic search or require specific checkpoint for testing with viz
            
            # Simplified: User should provide checkpoint for robust testing with visualizations
            # If not, we could try to guess, but it's error-prone due to varying save paths.
            # For now, let's make providing checkpoint more critical for viz.
            print("=====> No specific checkpoint provided. Please use --checkpoint for reliable testing with visualizations.")
            # Try loading latest, might fail if path is not found
            model_paths = glob.glob(os.path.join('checkpoint', args.dataset, args.model + 'bs*', '*.pth'))
            if not model_paths:
                raise FileNotFoundError("No checkpoint files found in default locations.")
            p = sorted(model_paths)[-1]
            print(f"=====> Found and loading latest checkpoint '{p}'")
            checkpoint = torch.load(p, map_location=device)
            model.load_state_dict(checkpoint['model'])
            if 'epoch' in checkpoint:
                 print(f"=====> loaded checkpoint from epoch {checkpoint['epoch']}")
            else:
                print("=====> loaded checkpoint (epoch number not found in checkpoint file)")

        except Exception as e:
            print(f"=====> Error loading latest checkpoint: {e}")
            print("=====> Please specify a checkpoint using --checkpoint for testing.")
            return # Exit if no model loaded

    print("=====> beginning testing and visualization")
    print("Test set length: ", len(testLoader))
    
    all_results = test(args, testLoader, model, device)

    if not all_results:
        print("No results to process or visualize.")
        return

    # Calculate overall average RMSE
    valid_rmses = [r['sample_rmse'] for r in all_results if r['sample_rmse'] != float('inf')]
    if valid_rmses:
        overall_avg_rmse = sum(valid_rmses) / len(valid_rmses)
        print(f'Overall Average RMSE: {overall_avg_rmse:.4f}')
    else:
        print("No samples with valid RMSE found.")

    # Filter results that have ground truth for sorting and visualization
    results_with_truth = [r for r in all_results if r['truth_denorm'] is not None]
    if not results_with_truth:
        print("No results with ground truth available for detailed visualization.")
        return

    # --- Generate Plots ---
    # 1. Random 6 samples
    if len(results_with_truth) >= 1: # Check if there's at least one sample
        random_samples = random.sample(results_with_truth, min(len(results_with_truth), 6))
        save_comparison_plots(random_samples, "Random Samples", args.save_viz_dir)
    else:
        print("Not enough samples with ground truth for random visualization.")

    # 2. Worst 6 samples (highest RMSE)
    worst_samples = sorted(results_with_truth, key=lambda x: x['sample_rmse'], reverse=True)[:6]
    save_comparison_plots(worst_samples, "Worst Samples (Highest RMSE)", args.save_viz_dir)

    # 3. Best 6 samples (lowest RMSE)
    best_samples = sorted(results_with_truth, key=lambda x: x['sample_rmse'])[:6]
    save_comparison_plots(best_samples, "Best Samples (Lowest RMSE)", args.save_viz_dir)


if __name__ == '__main__':
    args = parse_args()
    test_model(args)