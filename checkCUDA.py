import torch


def checkCUDA():
    iscuda = torch.cuda.is_available()
    print("CUDA is available: ", iscuda)


if __name__ == "__main__":
    checkCUDA()