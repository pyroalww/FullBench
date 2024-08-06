# FullBench
Ultimate PC Benchmark Tool

# FullBench: A Comprehensive System Benchmarking Tool

This Python script provides a set of advanced benchmarks designed to thoroughly evaluate various aspects of your computer's performance. It goes beyond basic benchmarks by incorporating diverse workloads, realistic scenarios, and detailed reporting. 

## Features

- **Comprehensive Hardware Analysis:**
    - **CPU:** Identifies CPU model, core count, clock speed, and other relevant information.
    - **GPU:** Detects GPU model, memory size, and compute capabilities.
    - **RAM:**  Reports total RAM capacity and usage.
    - **Storage:**  Provides details about your SSD/HDD, including capacity and read/write speeds.
    - **Network:**  Identifies your network interface and connection status.
- **Advanced Benchmarking Suite:**
    - **CPU Benchmark:** Measures CPU performance using:
        - Matrix operations (multiplication, inversion)
        - Prime number calculations
        - FFT (Fast Fourier Transform)
        - Compression algorithms (zlib)
    - **GPU Benchmark:**  Assesses GPU computational power through:
        - CUDA-accelerated matrix multiplication (if NVIDIA GPU is present)
        - OpenCL-based matrix multiplication for cross-platform compatibility 
    - **RAM Benchmark:** Evaluates RAM speed and efficiency through:
        - Sequential memory access tests
        - Random memory access tests
        - Varying data sizes to analyze performance at different scales 
    - **SSD Benchmark:**  Gauges SSD performance with:
        - Sequential read and write tests
        - Different file sizes to simulate real-world file operations
    - **Network Benchmark:** Measures network speed and latency via:
        - Downloading a file from a reliable server 
        - Calculating average latency to a well-known server (e.g., Google)
    - **Audio Benchmark:** Assesses audio processing capabilities by evaluating:
        - Audio playback latency 
        - Applying audio filters (low-pass filtering)
        - Performing FFT (Fast Fourier Transform) on audio data
    - **Image Benchmark:** Evaluates image processing performance with:
        - Gaussian blur
        - Edge detection (using the Canny algorithm) 
        - Image resizing
        - Utilizing OpenCV for image processing tasks 
    - **Encryption Benchmark:**  Benchmarks encryption speeds by measuring the time taken for:
        - SHA256 hashing algorithm
        - MD5 hashing algorithm 
        - Operations performed on a large dataset to simulate practical use cases
    - **Web Performance Benchmark:**  Simulates web-related tasks by timing:
        - API requests (using the GitHub API) 
        - Fetching a sample HTML page 
        - Measures response times and data transfer speeds
- **Detailed Reporting and Visualization:**
    - **Console Output:**  Presents comprehensive results for each benchmark, including timings, system information, and calculated scores.
    - **`benchmark_results.json`:** Saves all results in a structured JSON format for easy analysis and comparison. 
    - **`benchmark_results.png`:** Generates a bar chart visualizing the time taken for each benchmark test for quick visual insights.

## Requirements

- Python 3.6 or higher
- Libraries: 
    - `psutil`
    - `cpuinfo`
    - `GPUtil`
    - `pycuda` (for CUDA benchmarks - requires a compatible NVIDIA GPU and drivers)
    - `pyopencl` (for OpenCL benchmarks)
    - `numpy`
    - `matplotlib`
    - `scipy`
    - `zlib`
    - `sounddevice` 
    - `cv2` (OpenCV)
    - `hashlib`
    - `requests`

You can install the required libraries using pip:

```bash
pip install psutil cpuinfo GPUtil pycuda pyopencl numpy matplotlib scipy zlib sounddevice opencv-python hashlib requests
```

**Note:** 
- **CUDA (NVIDIA GPUs):** Ensure you have the correct CUDA drivers installed for your NVIDIA GPU to run the CUDA-based GPU benchmarks.
- **OpenCL:** You may need to install OpenCL drivers and the SDK for your system to execute the OpenCL GPU benchmarks.

## How to Run

1. Save the code as `fullbench.py`.
2. Open a terminal or command prompt and navigate to the directory where you saved the file.
3. Run the script using the following command:

```bash
python fullbench.py 
```

## Results

- **Console Output:** Provides detailed benchmark results, system information, and calculated scores.
- **`benchmark_results.json`:**  Stores the results in a JSON file for easy analysis and comparison.
- **`benchmark_results.png`:** Creates a bar chart visualization of the time taken for each benchmark test. 

## Customization

- The script allows you to modify workloads and parameters within each benchmark function to suit your specific needs.
- You can adjust data sizes, iteration counts, algorithms used, URLs, request types, image sizes, and operations, among other parameters.


## Disclaimer

- Benchmark results can vary depending on factors such as system configuration, background processes, thermal conditions, and software updates. 
- This script is intended for informational purposes and to provide a general overview of your system's performance. 
