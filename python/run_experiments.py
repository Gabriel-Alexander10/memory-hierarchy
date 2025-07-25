#!/usr/bin/env python3
import os
import subprocess
import re
import time
import pandas as pd
import json
from typing import List, Dict

class SpikeTest:
    def __init__(self):
        self.MATRIX_SIZES = [128, 256, 512, 1024]
        self.BLOCK_SIZES = [32, 64, 128]
        
        self.L1_CONFIGS = [
            "dc=4:8:32",
            "dc=16:8:32",
            "dc=32:8:64",
            "dc=128:16:128",
        ]
        
        self.L2_CONFIGS = [
            "l2=64:8:32",
            "l2=128:8:64",
            "l2=256:8:64",
            "l2=512:16:128",
        ]
        
        self.LOOP_ORDERS = {
            0: "ijk", 
            1: "ikj",
            2: "jik",
            3: "jki",
            4: "kij",
            5: "kji"
        }

    def compile_and_run(
        self, 
        compile_flags: List[str], 
        output_name: str,
        dcache_config: str,
        l2_config: str,
    ) -> Dict[str, float]:
        os.makedirs("./compiled", exist_ok=True)
        output_name = f"{output_name}_{l2_config}_{dcache_config}"
        
        compile_cmd = [
            "riscv64-unknown-elf-gcc",
            "-O2",
            "-march=rv64gc",
            *compile_flags,
            "./gemm/main.c",
            "-o",
            f"./compiled/{output_name}.elf"
        ]

        print(" ".join(compile_cmd))

        subprocess.run(compile_cmd, check=True)
        
        spike_cmd = [
            "spike",
            f"--{dcache_config}",
            f"--{l2_config}",
            "pk",
            f"./compiled/{output_name}.elf"
        ]

        print(" ".join(spike_cmd))

        start_time = time.time()


        result = subprocess.run(spike_cmd, capture_output=True, text=True)
        exec_time = time.time() - start_time
        
        metrics = {
            "exec_time": exec_time,
            "L1_miss": float(re.search(r"D\$ Miss Rate:\s+(\d+\.\d+)%", result.stdout).group(1)),
            "L2_miss": float(re.search(r"L2\$ Miss Rate:\s+(\d+\.\d+)%", result.stdout).group(1)),
        }
        return metrics

    def run_all_experiments(self) -> List[Dict]:
        results = []

        for size in self.MATRIX_SIZES:
            print(f"Running size {size}")
            for dcache_config in self.L1_CONFIGS:
                for l2_config in self.L2_CONFIGS:
                    print("=== Testing Loop Orders ===")
                    for order, name in self.LOOP_ORDERS.items():
                        metrics = self.compile_and_run(
                            [f"-DN={size}", f"-DNAIVE", f"-DMMORDER={order}", "./gemm/naive_gemm.c"],
                            f"naive_{name}", dcache_config, l2_config
                        )

                        results.append({
                            "Test": f"Loop Order {name.upper()}",
                            "Matrix Size": size,
                            "Implementation": f"naive_{name}",
                            "dcache_config": dcache_config,
                            "l2_config": l2_config,
                            "Execution Time (s)": metrics["exec_time"],
                            "L1 Miss %": metrics["L1_miss"],
                            "L2 Miss %": metrics["L2_miss"],
                        })
                    
                    print("\n=== Testing GEMM Implementations ===")
                    for impl in ["vector", "transpose"]:
                        
                        metrics = self.compile_and_run(
                            [f"-DN={size}", f"-D{impl.upper()}", f"./gemm/{impl}_gemm.c"],
                            impl, dcache_config, l2_config
                        )
                        results.append({
                            "Test": f"{impl.capitalize()} GEMM",
                            "Matrix Size": size,
                            "Implementation": impl,
                            "dcache_config": dcache_config,
                            "l2_config": l2_config,
                            "Execution Time (s)": metrics["exec_time"],
                            "L1 Miss %": metrics["L1_miss"],
                            "L2 Miss %": metrics["L2_miss"],
                        })

                    for block_size in self.BLOCK_SIZES:
                        metrics = self.compile_and_run(
                            [f"-DN={size}", f"-D{impl.upper()}", f"-DBLOCK_SIZE={block_size}", f"./gemm/{impl}_gemm.c"],
                            f"blocked_{block_size}", dcache_config, l2_config
                        )

                        results.append({
                            "Test": f"Blocked N={size} B={block_size}",
                            "Matrix Size": size,
                            "Block Size": block_size,
                            "Implementation": f"blocked_{block_size}",
                            "dcache_config": dcache_config,
                            "l2_config": l2_config,
                            "Execution Time (s)": metrics["exec_time"],
                            "L1 Miss %": metrics["L1_miss"],
                            "L2 Miss %": metrics["L2_miss"],
                        })
                    
                    print("\n=== Testing GotoBLAS ===")
                    metrics = self.compile_and_run([f"-DN={size}", "-DGOTO", "./gemm/gotoblas.c"], "gotoblas", dcache_config, l2_config)

                    results.append({
                        "Test": "GotoBLAS",
                        "Matrix Size": size,
                        "Implementation": "gotoblas",
                        "dcache_config": dcache_config,
                        "l2_config": l2_config,
                        "Execution Time (s)": metrics["exec_time"],
                        "L1 Miss %": metrics["L1_miss"],
                        "L2 Miss %": metrics["L2_miss"],
                    })

            save_results(results, f"metrics_{size}", "json")        
        return results

def save_results(data: List[Dict], filename: str, format_type: str = "json"):
    os.makedirs("results", exist_ok=True)
    
    if format_type == "json":
        with open(f"results/{filename}.json", "w") as f:
            json.dump(data, f, indent=2)
    elif format_type == "csv":
        pd.DataFrame(data).to_csv(f"results/{filename}.csv", index=False)
    else:
        raise ValueError("Unsupported format type")

if __name__ == "__main__":
    spike_tests = SpikeTest()
    
    print("Running standard experiments...")
    data = spike_tests.run_all_experiments()
    save_results(data, "metrics", "json")
    save_results(data, "metrics", "csv")