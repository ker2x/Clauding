
import torch

def test_pin_memory():
    if not torch.backends.mps.is_available():
        print("MPS not available")
        return

    print("Testing pin_memory on MPS...")
    try:
        # Create CPU tensor
        x = torch.zeros(100, device='cpu')
        print(f"Created tensor on {x.device}")
        
        # Try to pin memory
        print("Calling pin_memory()...")
        x_pinned = x.pin_memory()
        print(f"Pinned tensor device: {x_pinned.device}")
        print(f"Pinned: {x_pinned.is_pinned()}")
        
        # Try to move to MPS
        print("Moving to MPS...")
        x_mps = x_pinned.to('mps')
        print(f"MPS tensor device: {x_mps.device}")
        print("Success!")
        
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pin_memory()
