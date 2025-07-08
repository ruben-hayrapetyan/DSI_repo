import subprocess
import time
import os

# Set this to your GSD code directory
GSD_CODE_DIR = "/Users/rubenhayrapetyan/Downloads/Code/DSI/DSI_repo/Similarity Metric Implementations/All_GSD_Code"

experiments = [
    {"private": "usps"},
    {"private": "fashionmnist"},
    {"private": "svhn"},
    {"private": "cifar10"},
    {"private": "stl10"},
]

for i, exp in enumerate(experiments):
    print(f"\n{'='*50}")
    print(f"Starting experiment {i+1}/{len(experiments)}")
    print(f"MNIST (public) vs {exp['private'].upper()} (private)")
    print(f"{'='*50}")
    
    # Construct the command with all required parameters
    command = [
        "python3",
        os.path.join(GSD_CODE_DIR, "compute_distance.py"),
        "--device", "cpu",  # or "cuda" if you do have GPU
        "--model_type", "probe",
        "--private_dataset", exp["private"],
        "--public_dataset", "mnist",
        "--train_batch_size", "256",
        "--eval_batch_size", "256",
        "--num_eigenthings", "50",
        "--num_examples", "5000"
    ]
    
    try:
        # Start timer
        start_time = time.time()
        
        # Run the command with real-time output
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        # Print output in real-time
        for line in process.stdout:
            print(line, end='')
        
        # Wait for completion
        return_code = process.wait()
        
        # Calculate duration
        duration = time.time() - start_time
        
        if return_code == 0:
            print(f"✅ Experiment completed successfully in {duration:.2f} seconds")
        else:
            print(f"❌ Experiment failed with exit code {return_code} after {duration:.2f} seconds")
    
    except Exception as e:
        print(f"🚨 Unexpected error: {str(e)}")
    
    print("\n" + "="*50)
    print("Pausing before next experiment...")
    time.sleep(2)  # Short pause between experiments