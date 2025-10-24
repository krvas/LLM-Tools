


gpus_available = {
    "A10": {
        "memory_GB": 24,
        "compute_TFLOPS": {
            "FP32": 31.2,
            "TF32_Tensor": 62.5
        },
        "cloud_pricing": {
            "AWS_g5.xlarge": {
                "provider": "AWS",
                "instance_type": "g5.xlarge",
                "on_demand_usd_per_hour": 1.01
            }
        }
    },
    "A100_40GB": {
        "memory_GB": 40,
        "compute_TFLOPS": {
            "FP32": 19.5,
            "Tensor": 312
        },
        "cloud_pricing": {
            "GCP_a2-highgpu-1g": {
                "provider": "Google Cloud",
                "instance_type": "a2-highgpu-1g",
                "on_demand_usd_per_hour": 4.10
            }
        }
    },
    "H100": {
        "memory_GB": 80,
        "compute_TFLOPS": {
            "FP32": 67,
            "Tensor": 989
        },
        "cloud_pricing": {
            "GCP_a3-highgpu-1g": {
                "provider": "Google Cloud",
                "instance_type": "a3-highgpu-1g",
                "on_demand_usd_per_hour": 11.06
            }
        }
    }
}
