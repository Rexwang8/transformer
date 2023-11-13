import os
import torch


def checkCUDA():
    iscuda = torch.cuda.is_available()
    print("CUDA is available: ", iscuda)
    print(f"Torch version: {torch.__version__}")

def checkTorchVars():
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"ENV TORCH_USE_CUDA_DSA: {os.environ.get('TORCH_USE_CUDA_DSA')}")
    print(f"ENV CUDA_LAUNCH_BLOCKING: {os.environ.get('CUDA_LAUNCH_BLOCKING')}")

if __name__ == "__main__":
    checkCUDA()
    checkTorchVars()