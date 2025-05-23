import argparse
import os
import time
import torch
import numpy as np
import subprocess
import psutil # For memory usage

# user
from builders.model_builder import build_model
from builders.dataset_builder import build_dataset_test
from dataset.phaseUnwrapping import denormalize_input, denormalize_label # For PUNet output processing

def parse_args():
    parser = argparse.ArgumentParser(description='Benchmarking PUNet and SNAPHU')
    parser.add_argument('--model_name', type=str, default="PUNet", help="Model name for PUNet")
    parser.add_argument('--checkpoint', type=str, required=True, help="PUNet checkpoint file")
    parser.add_argument('--dataRootDir', type=str, default="dataset", help="Dataset root directory")
    parser.add_argument('--dataset', default="phaseUnwrapping", help="Dataset name")
    parser.add_argument('--num_workers', type=int, default=0, help="Number of workers for data loading")
    parser.add_argument('--cuda', default=True, type=lambda x: (str(x).lower() == 'true'), help="Run PUNet on GPU")
    parser.add_argument("--gpus", default="0", type=str, help="GPU IDs for PUNet (default: 0)")
    parser.add_argument('--snaphu_path', type=str, default="snaphu", help="Path to SNAPHU executable or command")
    parser.add_argument('--output_dir', type=str, default="benchmark_results", help="Directory to save benchmark results and outputs")
    # Add input_size, as it's crucial for both model and SNAPHU config
    parser.add_argument('--input_size', type=str, default="256,256", help="Input size of model (height,width)")

    args = parser.parse_args()
    return args

def benchmark_punet(args, test_loader, model, device):
    """Benchmarks the PUNet model."""
    print("\n--- Benchmarking PUNet ---")
    model.eval()
    total_time = 0
    # More detailed memory profiling might require inspecting resident memory changes
    # This is a basic start, might need refinement.
    process = psutil.Process(os.getpid())
    initial_mem = process.memory_info().rss / (1024 * 1024) # MB

    punet_outputs_dir = os.path.join(args.output_dir, "punet_outputs")
    os.makedirs(punet_outputs_dir, exist_ok=True)

    all_sample_times = []

    for i, batch_data in enumerate(test_loader):
        if len(batch_data) == 4: # (input, label, size, name)
            input_tensor, _, _, name = batch_data
        elif len(batch_data) == 3: # (input, size, name)
            input_tensor, _, name = batch_data
        else:
            raise ValueError(f"Unexpected number of items in batch_data: {len(batch_data)}")

        input_var = input_tensor.to(device)
        
        start_time = time.time()
        with torch.no_grad():
            output = model(input_var)
        
        if device.type != 'cpu':
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elif device.type == 'mps':
                if hasattr(torch, 'mps') and hasattr(torch.mps, 'synchronize'):
                    torch.mps.synchronize()
                else:
                    print("Warning: torch.mps.synchronize not found. MPS timing might be less accurate.")
        
        end_time = time.time()
        
        sample_time = end_time - start_time
        total_time += sample_time
        all_sample_times.append(sample_time)
        print(f"PUNet - Sample: {name[0]}, Time: {sample_time:.4f}s")

        # Save output (optional, but good for verification)
        output_numpy = output.cpu().data[0].numpy()
        # output_denorm_np = denormalize_label(output_numpy) # If you need to save denormalized
        output_filename = os.path.join(punet_outputs_dir, f"{name[0]}_punet_unwrapped.raw")
        output_numpy.tofile(output_filename)


    final_mem = process.memory_info().rss / (1024 * 1024) # MB
    avg_time_per_sample = total_time / len(test_loader) if test_loader else 0
    
    print(f"PUNet - Total processing time: {total_time:.4f}s")
    print(f"PUNet - Average time per sample: {avg_time_per_sample:.4f}s")
    # print(f"PUNet - Initial memory: {initial_mem:.2f} MB") # This might not be super accurate for GPU memory
    # print(f"PUNet - Final memory: {final_mem:.2f} MB") # Same as above
    # For GPU memory, torch.cuda.memory_allocated() and max_memory_allocated() are better.
    if device.type == 'cuda':
        print(f"PUNet - Max GPU memory allocated: {torch.cuda.max_memory_allocated(device) / (1024*1024):.2f} MB")
        torch.cuda.reset_max_memory_allocated(device) # Reset for next benchmark if any

    return {"total_time": total_time, "avg_time_per_sample": avg_time_per_sample, "samples_processed": len(test_loader)}


def benchmark_snaphu(args, test_loader):
    """Benchmarks SNAPHU."""
    print("\n--- Benchmarking SNAPHU ---")
    total_time = 0
    # SNAPHU runs as a separate process, so we measure its peak memory if possible,
    # or rely on system tools for more detailed profiling externally.
    # psutil can also monitor child processes, but it gets complex.
    # For simplicity, we'll focus on execution time here.

    snaphu_outputs_dir = os.path.join(args.output_dir, "snaphu_outputs")
    os.makedirs(snaphu_outputs_dir, exist_ok=True)
    snaphu_config_dir = os.path.join(args.output_dir, "snaphu_configs")
    os.makedirs(snaphu_config_dir, exist_ok=True)

    h, w = map(int, args.input_size.split(','))

    all_sample_times = []
    processed_samples = 0

    for i, batch_data in enumerate(test_loader):
        # The test_loader for PUNet gives normalized data.
        # SNAPHU needs raw wrapped phase. The 'img' field in dataset.files is the path to this.
        # We need to get the original file path.
        # The `test_loader` yields (input_tensor, (label_tensor), size, name).
        # `name` is the original filename (e.g., '00899.wzp').
        # We need to reconstruct the path to the raw input file.
        
        if len(batch_data) == 4: # (input, label, size, name) - has ground truth
             _, _, _, name_tuple = batch_data
        elif len(batch_data) == 3: # (input, size, name) - no ground truth
             _, _, name_tuple = batch_data
        else:
            raise ValueError(f"Unexpected number of items in batch_data: {len(batch_data)}")

        sample_name = name_tuple[0] # e.g., "00899.wzp"
        # Construct path to the original wrapped phase file
        # Assuming the structure dataset/<dataset_name>/interf/<sample_name>
        wrapped_phase_path = os.path.join(args.dataRootDir, args.dataset, "interf", sample_name)

        if not os.path.exists(wrapped_phase_path):
            print(f"SNAPHU - WARNING: Input file not found {wrapped_phase_path}, skipping.")
            continue
        
        unwrapped_output_path = os.path.join(snaphu_outputs_dir, f"{sample_name}_snaphu_unwrapped.raw")
        snaphu_config_path = os.path.join(snaphu_config_dir, f"{sample_name}.snaphu.conf")

        # Create a basic SNAPHU config file
        # Refer to SNAPHU documentation for more detailed config options.
        # This is a minimal config.
        config_content = f"""\
STATCOSTMODE DEFO
INFILE {wrapped_phase_path}
OUTFILE {unwrapped_output_path}
LINELENGTH {w}
# Optional: Add a connected components file if you generate one
# CONNCOMPFILE {os.path.join(snaphu_outputs_dir, f"{sample_name}_snaphu_conncomp.raw")}
"""
        with open(snaphu_config_path, 'w') as f:
            f.write(config_content)

        # Command to run SNAPHU
        # The `-f` flag tells SNAPHU to read parameters from a config file.
        # We redirect stdout and stderr to avoid cluttering the console.
        cmd = [args.snaphu_path, "-f", snaphu_config_path]

        start_time = time.time()
        try:
            # Using psutil to monitor the child process for memory
            # This is more involved; for now, let's focus on time and successful execution
            process = psutil.Process() # Current Python process
            initial_children_mem = sum([p.memory_info().rss for p in process.children(recursive=True)])

            snaphu_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = snaphu_process.communicate(timeout=300) # Added timeout

            final_children_mem = sum([p.memory_info().rss for p in process.children(recursive=True)])
            # This children memory tracking is very basic and might not be accurate for peak usage.

            if snaphu_process.returncode != 0:
                print(f"SNAPHU - ERROR processing {sample_name}:")
                print(f"  STDOUT: {stdout.decode()}")
                print(f"  STDERR: {stderr.decode()}")
                continue # Skip this sample if SNAPHU failed
        except subprocess.TimeoutExpired:
            print(f"SNAPHU - TIMEOUT processing {sample_name}. Killing process.")
            snaphu_process.kill()
            stdout, stderr = snaphu_process.communicate()
            print(f"  STDOUT: {stdout.decode()}")
            print(f"  STDERR: {stderr.decode()}")
            continue
        except Exception as e:
            print(f"SNAPHU - EXCEPTION processing {sample_name}: {e}")
            continue


        end_time = time.time()
        sample_time = end_time - start_time
        total_time += sample_time
        all_sample_times.append(sample_time)
        print(f"SNAPHU - Sample: {sample_name}, Time: {sample_time:.4f}s, Output: {unwrapped_output_path}")
        processed_samples +=1


    if processed_samples > 0:
        avg_time_per_sample = total_time / processed_samples
    else:
        avg_time_per_sample = 0
        print("SNAPHU - No samples were processed successfully.")

    print(f"SNAPHU - Total processing time for {processed_samples} samples: {total_time:.4f}s")
    print(f"SNAPHU - Average time per sample: {avg_time_per_sample:.4f}s")
    # Memory for SNAPHU is harder to track precisely from Python for a separate C executable.
    # One might use /usr/bin/time -v snaphu ... outside of Python for more detailed memory.

    return {"total_time": total_time, "avg_time_per_sample": avg_time_per_sample, "samples_processed": processed_samples}


def main():
    args = parse_args()
    print(args)

    # Add this line to potentially stabilize CPU execution
    # torch.set_num_threads(1) 
    #torch.set_num_threads(4) # Or your desired number
    print(f"PyTorch using {torch.get_num_threads()} threads.")
    print("PyTorch parallel_info:")
    print(torch.__config__.parallel_info())

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Setup device for PUNet ---
    device_str = "cpu"
    if args.cuda:
        if torch.cuda.is_available():
            print("=====> PUNet: use gpu id: '{}'".format(args.gpus))
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
            device_str = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): # MPS check
            print("=====> PUNet: using Apple Metal Performance Shaders (MPS)")
            device_str = "mps"
        else:
            print("=====> PUNet: CUDA/MPS not available, falling back to CPU")
    else:
        print("=====> PUNet: Running on CPU")
    device = torch.device(device_str)

    # --- Load Test Data ---
    # The test loader needs to provide data suitable for PUNet (normalized)
    # and also give us info (like filenames) to run SNAPHU on original data.
    # build_dataset_test expects input_size as a tuple.
    h, w = map(int, args.input_size.split(','))
    input_size_tuple = (h,w)

    # We need the original file paths for SNAPHU, and normalized tensors for PUNet.
    # `build_dataset_test` with `none_gt=True` (or rather, one that provides original file access)
    # `PhaseUnwrappingTestDataSet` provides (normalized_image, size, name)
    # The `name` is the key to get the original file.
    # `build_dataset_test` has `list_path_override` for `test.txt`. Default is fine.
    # Use `none_gt=True` to use `PhaseUnwrappingTestDataSet` which loads only input.
    # However, the default `build_dataset_test` with `none_gt=False` actually uses `PhaseUnwrappingDataSet`,
    # which also provides `name`. So, the standard loader should be fine.
    # Let's use `none_gt=False` as it's the default in `test.py` and provides label for potential comparison.
    # The batch size for testing/benchmarking is typically 1.
    _, test_loader = build_dataset_test(args.dataRootDir, args.dataset, args.num_workers, none_gt=False, input_size=input_size_tuple)
    print(f"Test set length: {len(test_loader)}")
    if len(test_loader) == 0:
        print("Test loader is empty. Check dataset path and test.txt file.")
        return

    # --- Benchmark PUNet ---
    punet_model_to_benchmark = None # Initialize
    if args.checkpoint and os.path.isfile(args.checkpoint):
        punet_model = build_model(args.model_name, num_channels=1) # Assuming 1 channel
        checkpoint_data = torch.load(args.checkpoint, map_location=device)
        punet_model.load_state_dict(checkpoint_data['model'])
        punet_model = punet_model.to(device)
        punet_model.eval() # Ensure model is in eval mode before compilation

        punet_model_to_benchmark = punet_model # Default to original model

        # --- Attempt torch.compile() --- 
        if hasattr(torch, 'compile'):
            print("Attempting to compile the model with torch.compile()...")
            try:
                # For faster compilation during testing, default mode is often fine.
                # For max performance, you might explore modes like "max-autotune" or "reduce-overhead",
                # but they can significantly increase compilation time for the first run.
                compiled_model = torch.compile(punet_model)
                punet_model_to_benchmark = compiled_model
                print("torch.compile() successful. Using compiled model for PUNet.")
            except Exception as e:
                print(f"torch.compile() failed: {e}. Using original model for PUNet.")
        else:
            print("torch.compile() not available in this PyTorch version. Using original model for PUNet.")
        # --- End torch.compile() attempt ---


        if device.type == 'cuda': # Note: cudnn.benchmark is typically for CUDA, not MPS
            torch.backends.cudnn.benchmark = True
        
        # punet_results = benchmark_punet(args, test_loader, punet_model, device)
        punet_results = benchmark_punet(args, test_loader, punet_model_to_benchmark, device)

    else:
        print("PUNet checkpoint not found or not specified. Skipping PUNet benchmark.")
        punet_results = None

    # --- Benchmark SNAPHU ---
    # SNAPHU uses the same test set. The loader already provides filenames.
    snaphu_results = benchmark_snaphu(args, test_loader)

    # --- Report Results ---
    print("\n--- Benchmark Summary ---")
    if punet_results:
        print(f"PUNet Model ({args.model_name}):")
        print(f"  Samples processed: {punet_results['samples_processed']}")
        print(f"  Total time: {punet_results['total_time']:.4f}s")
        print(f"  Average time per sample: {punet_results['avg_time_per_sample']:.4f}s")
        if device.type == 'cuda': # Re-fetch for summary if needed
             print(f"  Max GPU memory allocated: {torch.cuda.max_memory_allocated(device) / (1024*1024):.2f} MB")


    if snaphu_results:
        print(f"SNAPHU ({args.snaphu_path}):")
        print(f"  Samples processed: {snaphu_results['samples_processed']}")
        print(f"  Total time: {snaphu_results['total_time']:.4f}s")
        print(f"  Average time per sample: {snaphu_results['avg_time_per_sample']:.4f}s")

    # Further: Save results to a file (e.g., CSV or JSON)
    results_summary_file = os.path.join(args.output_dir, "benchmark_summary.txt")
    with open(results_summary_file, 'w') as f:
        f.write("Benchmark Summary:\n")
        if punet_results:
            f.write(f"PUNet Model ({args.model_name}):\n")
            f.write(f"  Samples processed: {punet_results['samples_processed']}\n")
            f.write(f"  Total time: {punet_results['total_time']:.4f}s\n")
            f.write(f"  Average time per sample: {punet_results['avg_time_per_sample']:.4f}s\n")
            if device.type == 'cuda':
                 f.write(f"  Max GPU memory allocated: {torch.cuda.max_memory_allocated(device) / (1024*1024):.2f} MB\n")

        if snaphu_results:
            f.write(f"SNAPHU ({args.snaphu_path}):\n")
            f.write(f"  Samples processed: {snaphu_results['samples_processed']}\n")
            f.write(f"  Total time: {snaphu_results['total_time']:.4f}s\n")
            f.write(f"  Average time per sample: {snaphu_results['avg_time_per_sample']:.4f}s\n")
    print(f"Benchmark summary saved to: {results_summary_file}")

if __name__ == '__main__':
    main() 