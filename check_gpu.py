import torch
info = []
info.append(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    info.append(f"device_name={torch.cuda.get_device_name(0)}")
    info.append(f"total_mem_gb={torch.cuda.get_device_properties(0).total_mem//(1024**3)}")
else:
    info.append("device_name=none")
    info.append("total_mem_gb=0")
with open("gpu_info.txt", "w") as f:
    f.write("\n".join(info))
print("\n".join(info))
