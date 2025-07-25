#!/usr/bin/env python3
import os
import subprocess
import re
import time
import pandas as pd
import json
from typing import List, Dict

class CompareVectorAllocation:
    def __init__(self):
        self.MATRIX_SIZES = [128]
        
        self.L1_CONFIGS = [
            "dc=32:4:256"
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
    ) -> Dict[str, float]:
        os.makedirs("./compiled", exist_ok=True)
        output_name = f"{output_name}_{dcache_config}"
        
        # Compile command
        compile_cmd = [
            "riscv64-unknown-elf-gcc",
            "-O2",
            "-march=rv64gc",
            *compile_flags,
            "-o",
            f"./compiled/{output_name}.elf"
        ]

        print(" ".join(compile_cmd))

        subprocess.run(compile_cmd, check=True)
        
        spike_cmd = [
            "spike",
            f"--{dcache_config}",
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
        }
        return metrics

    def run_all_experiments(self) -> List[Dict]:
        results = []

        for size in self.MATRIX_SIZES:
            for dcache_config in self.L1_CONFIGS:
                    # 1. Compare loop orders
                    print("=== Testing Loop Orders ===")
                    for order, name in self.LOOP_ORDERS.items():
                        metrics = self.compile_and_run(
                            [f"-DN={size}", f"-DNAIVE", f"-DMMORDER={order}", "./gemm/naive_gemm.c", "./gemm/main.c"],
                            f"naive_gemm_{name}_{size}", dcache_config
                        )

                        results.append({
                            "Test": f"Loop Order {name.upper()}",
                            "Matrix Size": size,
                            "Implementation": f"naive_gemm_{name}",
                            "dcache_config": dcache_config,
                            "Execution Time (s)": metrics["exec_time"],
                            "L1 Miss %": metrics["L1_miss"],
                        })


                        metrics = self.compile_and_run(
                            [f"-DNMAX={size}", f"-DNAIVE", f"-DMMORDER={order}", "./gemm/mmult.c"],
                            f"mmult_{name}_{size}", dcache_config
                        )

                        results.append({
                            "Test": f"Loop Order {name.upper()}",
                            "Matrix Size": size,
                            "Implementation": f"mmult_{name}",
                            "dcache_config": dcache_config,
                            "Execution Time (s)": metrics["exec_time"],
                            "L1 Miss %": metrics["L1_miss"],
                        })


            save_results(results, f"compare_vector_allocation_{size}", "json")
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
    spike_tests = CompareVectorAllocation()
    print("Running standard experiments...")
    standard_data = spike_tests.run_all_experiments()
    save_results(standard_data, "compare_vector_allocation", "json")