"""
Convert GPU model to CPU + reduce size via FP16
Usage: python convert_to_cpu.py input.pt [output.pt] [fp16]
"""

import torch
import sys

def convert(input_path, output_path=None, fp16=True):
    output_path = output_path or input_path.replace('.pt', '_cpu.pt')
    
    state = torch.load(input_path, map_location='cpu')
    
    def to_fp16(obj):
        if isinstance(obj, torch.Tensor) and obj.is_floating_point():
            return obj.half()
        if isinstance(obj, dict):
            return {k: to_fp16(v) for k, v in obj.items()}
        return obj
    
    if fp16:
        state = to_fp16(state)
    
    torch.save(state, output_path)
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)