import torch

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    try:
        x = torch.tensor([1.0, 2.0], device='cuda')
        print(f"GPU Execution Test: Success! {x}")
    except Exception as e:
        print(f"GPU Test Failed: {e}")
else:
    print("CUDA not available.")