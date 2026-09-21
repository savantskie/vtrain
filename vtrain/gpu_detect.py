"""
GPU device detection utility.
Lists all available Vulkan devices with name and helps identify
which GPU to use (discrete) vs skip (integrated / software).
"""

import kp


def detect_gpus() -> list:
    """
    Return a list of dicts, one per available Vulkan device:
      {
        "index": int,
        "name": str,
        "type": str,   # "discrete", "integrated", "software", "other"
      }
    """
    mgr = kp.Manager()
    devices_raw = mgr.list_devices()
    mgr.destroy()

    devices = []
    for i, raw in enumerate(devices_raw):
        name = raw.get("device_name", f"Unknown GPU {i}")
        name_lower = name.lower()

        if "llvmpipe" in name_lower or "lavapipe" in name_lower or "software" in name_lower:
            dtype = "software"
        elif "renoir" in name_lower or "rajaen" in name_lower:
            dtype = "integrated"
        elif "radeon" in name_lower and "graphics" in name_lower:
            dtype = "integrated"
        else:
            dtype = "discrete"

        devices.append({
            "index": i,
            "name":  name,
            "type":  dtype,
        })

    return devices


def print_gpu_info():
    """Print detected GPUs in a human-readable format."""
    gpus = detect_gpus()
    if not gpus:
        print("No Vulkan-capable GPUs detected.")
        return

    print(f"Detected {len(gpus)} GPU(s):")
    print()
    for g in gpus:
        t = g["type"]
        print(f"  [{g['index']}] {g['name']}")
        print(f"       type: {t}")
    print()

    discrete = [g for g in gpus if g["type"] == "discrete"]
    integrated = [g for g in gpus if g["type"] == "integrated"]

    if discrete:
        print(f"Discrete GPU(s) at indices: {[g['index'] for g in discrete]}")
    if integrated:
        print(f"Integrated GPU at index {integrated[0]['index']} — skip this one.")
    print(f"Recommend: --device {discrete[0]['index'] if discrete else '0'}")


if __name__ == "__main__":
    print_gpu_info()