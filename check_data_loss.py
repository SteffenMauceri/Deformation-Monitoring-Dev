import glob
import os
import time
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import torch.nn as nn
import tempfile # Added for temporary file

# user
from builders.model_builder import build_model
from builders.dataset_builder import build_dataset_test


def parse_args():
    parser = ArgumentParser(description='Check data for high loss samples using PUNet')
    parser.add_argument('--model', type=str, default="PUNet", help="model name: (default PUNet)")
    parser.add_argument('--dataRootDir', type=str,
                        default=r"dataset",
                        help="Base directory for dataset structure (e.g., 'dataset')")
    parser.add_argument('--dataset', default="phaseUnwrapping", 
                        help="Dataset subdirectory name (e.g., 'phaseUnwrapping', used with dataRootDir to define dataset base path)")
    parser.add_argument('--target_directory', type=str, required=True,
                        help="Directory containing the .wzp files to check (e.g., dataset/phaseUnwrapping/interf)")
    parser.add_argument('--file_pattern', type=str, default="*.wzp",
                        help="Pattern to match files in target_directory (default: *.wzp)")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help="Path to the trained model checkpoint (.pth file)")
    parser.add_argument('--cuda', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help="run on CPU or GPU (True/False)")
    parser.add_argument('--loss_threshold_percentile', type=float, default=95.0,
                        help="Percentile to determine high loss threshold (e.g., 99 means top 1%% losses are outliers)")
    parser.add_argument('--input_size', type=str, default="256,256", help="input size of model (h,w)")

    args = parser.parse_args()
    return args


def calculate_losses_and_identify_outliers(args, data_loader, model, device, criterion):
    model.eval()
    all_losses = []
    outlier_files = []

    print(f"Calculating losses for {len(data_loader)} samples from {args.target_directory}...")
    for i, (input_tensor, label_tensor, size, name_from_dataset) in enumerate(data_loader):
        with torch.no_grad():
            input_var = input_tensor.to(device)
            label_var = label_tensor.to(device)

            output = model(input_var)
            loss = criterion(output, label_var)
            
            # name_from_dataset is from the dataloader, might be a tuple/list even with batch_size=1
            current_name_str = name_from_dataset[0] if isinstance(name_from_dataset, (list, tuple)) else name_from_dataset
            
            # Construct full-like path for reporting, as current_name_str is now just the basename
            reported_name = os.path.join(os.path.basename(args.target_directory), current_name_str)
            all_losses.append((reported_name, loss.item())) 

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(data_loader)} samples...")

    if not all_losses:
        print("No losses calculated. Check dataset or processing.")
        return [], "No plot generated."

    # Sort by loss to find threshold and outliers
    all_losses.sort(key=lambda x: x[1], reverse=True)

    # Plot histogram
    loss_values = [item[1] for item in all_losses]
    plt.figure(figsize=(10, 6))
    plt.hist(loss_values, bins=150, edgecolor='black')
    plt.title(f'Histogram of Sample Losses for {os.path.basename(args.target_directory)}')
    plt.xlabel('Loss Value')
    plt.ylabel('Frequency')
    plt.yscale('log') 
    plt.xlim(0, 10000)
    
    # Make plot filename more specific
    target_dir_basename = os.path.basename(args.target_directory.rstrip('/\\'))
    plot_filename = f"loss_histogram_{args.dataset}_{target_dir_basename}.png"
    plt.savefig(plot_filename)
    print(f"Loss histogram saved to {plot_filename}")
    # plt.show() # Comment out to prevent blocking in automated runs if any

    # Identify outliers based on percentile
    # Ensure loss_values is not empty before percentile calculation
    if not loss_values:
        print("No loss values to calculate percentile from.")
        return [], plot_filename

    threshold_index = int(len(loss_values) * (1 - args.loss_threshold_percentile / 100.0))
    # Ensure index is within bounds
    threshold_index = max(0, min(threshold_index, len(loss_values) - 1))
    
    loss_threshold_val = loss_values[threshold_index]

    print(f"Loss threshold (approx. top {100 - args.loss_threshold_percentile:.2f}% percentile, value > {loss_threshold_val:.4f}):")

    print("\\nFiles with loss greater than or equal to threshold (potential outliers):")
    count = 0
    # Iterate through all_losses, which is sorted by loss descending
    for file_name, loss_val in all_losses:
        if loss_val >= loss_threshold_val: # Using >= to include the threshold value itself
            # Check if we have already printed enough outliers according to percentile, to avoid printing too many if many have same high loss
            if count < (len(all_losses) - threshold_index): # Print up to the number of elements above or at threshold_index
                 print(f"File: {file_name}, Loss: {loss_val:.4f}")
                 outlier_files.append(file_name)
                 count += 1
            elif loss_val == loss_threshold_val: # If there are ties at the threshold, print them all
                 print(f"File: {file_name}, Loss: {loss_val:.4f} (tie at threshold)")
                 outlier_files.append(file_name)
                 # count increment is not strictly necessary here as we are just printing ties
            else: # if we are past the Nth percentile items and not a tie.
                break 
        else: # Since sorted, we can break early
            break
            
    if not outlier_files: # Check if the list is empty
        print("No files identified as outliers above the calculated loss threshold.")
    
    return outlier_files, plot_filename


def main_check_data(args):
    print("Script arguments:", args)

    device = "cpu"
    if args.cuda:
        if torch.cuda.is_available():
            print("=====> Using CUDA (default device)")
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("=====> using Apple Metal Performance Shaders (MPS)")
            device = "mps"
        else:
            print("=====> No GPU (CUDA or MPS) available/selected, falling back to CPU")
    else:
        print("=====> Running on CPU")

    # Build the model
    model = build_model(args.model, num_channels=1)
    model = model.to(device)

    if args.cuda and device == "cuda":
        cudnn.benchmark = True

    # Load checkpoint
    if not args.checkpoint or not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found at '{args.checkpoint}'")

    print(f"=====> loading checkpoint '{args.checkpoint}'")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    is_data_parallel = any(key.startswith('module.') for key in state_dict.keys())

    if is_data_parallel and not (device == "cuda" and torch.cuda.device_count() > 1 and isinstance(model, nn.DataParallel)):
        for k, v in state_dict.items():
            name_key = k[7:] if k.startswith('module.') else k
            new_state_dict[name_key] = v
        model.load_state_dict(new_state_dict)
        print("=====> Loaded DataParallel checkpoint onto a non-DataParallel model.")
    elif not is_data_parallel and (device == "cuda" and torch.cuda.device_count() > 1 and isinstance(model, nn.DataParallel)):
        print("Warning: Loading a non-DataParallel checkpoint onto a DataParallel model.")
        for k, v in state_dict.items():
            name_key = 'module.' + k if not k.startswith('module.') else k
            new_state_dict[name_key] = v
        model.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(state_dict)
    print("=====> Checkpoint loaded successfully.")

    # --- Generate file list internally ---
    dataset_base_path = os.path.join(args.dataRootDir, args.dataset) # e.g. dataset/phaseUnwrapping
    
    search_path = os.path.join(args.target_directory, args.file_pattern)
    print(f"=====> Scanning for files in: {search_path}")
    found_files_full_path = glob.glob(search_path)

    if not found_files_full_path:
        print(f"No files found in '{args.target_directory}' matching pattern '{args.file_pattern}'. Exiting.")
        return

    print(f"Found {len(found_files_full_path)} files.")
    
    # Create relative paths for the list file
    # PhaseUnwrappingDataSet expects paths relative to its 'root', which will be 'dataset_base_path'
    # It also expects these to be just basenames, as it will prepend 'interf/' or 'origin/' itself.
    file_basenames = [os.path.basename(f) for f in found_files_full_path]

    temp_list_file_path = "" # Initialize to prevent reference before assignment in finally if NamedTemporaryFile fails
    try:
        # Use a more robust way to get temp file path
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', dir='.') as tmp_f:
            for basename in file_basenames:
                tmp_f.write(f"{basename}\n")
            temp_list_file_path = tmp_f.name
        print(f"=====> Temporary list file created at: {temp_list_file_path} with {len(file_basenames)} entries.")

        h, w = map(int, args.input_size.split(','))
        input_size_tuple = (h, w)

        # Call build_dataset_test with dataRootDir and dataset, it will form the correct 'root' for PhaseUnwrappingDataSet
        _, data_loader = build_dataset_test(
            args.dataRootDir,
            args.dataset,
            num_workers=0, 
            list_path_override=temp_list_file_path,
            input_size=input_size_tuple
        )

        if len(data_loader) == 0:
            print(f"No data loaded from the generated list file {temp_list_file_path}. Please check paths and file content logic.")
            return

        criterion = nn.MSELoss(reduction='mean').to(device)

        print("=====> Starting loss calculation and outlier identification")
        outliers, plot_path = calculate_losses_and_identify_outliers(args, data_loader, model, device, criterion)

        if outliers:
            print(f"\\nIdentified {len(outliers)} outlier files (see details above). Histogram: {plot_path}")
        else:
            print(f"\\nNo outlier files identified based on the threshold. Histogram: {plot_path}")

    finally:
        if temp_list_file_path and os.path.exists(temp_list_file_path):
            os.remove(temp_list_file_path)
            print(f"=====> Temporary list file {temp_list_file_path} removed.")


if __name__ == '__main__':
    args = parse_args()
    main_check_data(args) 