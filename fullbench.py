# Made by @pyroalww
# Instagram @c4gwn
# M.I.T. License
# https://pyrollc.com.tr
# FullBench 
# v1.0.4

# ===========================
# - Fixed OpenCL
# - Fixed Pycuda skip for AMD GPUs
# - Reduced GPU load
# - Added HTML Result
# - Machine Learning: Reduced load .50

print("Importing libs.")
import psutil
import cpuinfo
import GPUtil
import numpy as np
import time
import subprocess
import os
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy import signal
import zlib
import multiprocessing
import socket
import sounddevice as sd
import cv2
import hashlib
import requests
import json
from PIL import Image
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import pyopencl as cl
import sqlite3
import threading
import ssl
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import random
    
try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
print("Benchmark started. This process takes 30 minutes to 1 hour depending on your hardware. You will be notified when the process is complete.")
class AdvancedBench:
    def __init__(self):
        self.scores = {
            'cpu': 0,
            'gpu': 0,
            'ram': 0,
            'ssd': 0,
            'network': 0,
            'audio': 0,
            'image': 0,
            'encryption': 0,
            'web': 0,
            'ml': 0,
            'database': 0,
            'parallel': 0,
            'network_security': 0,
            'graphics': 0
        }
        self.details = {}
        self.results = {}
        self.weights = {
            'cpu': 0.15,
            'gpu': 0.15,
            'ram': 0.1,
            'ssd': 0.1,
            'network': 0.05,
            'audio': 0.05,
            'image': 0.05,
            'encryption': 0.05,
            'web': 0.05,
            'ml': 0.1,
            'database': 0.05,
            'parallel': 0.05,
            'network_security': 0.025,
            'graphics': 0.025
        }

    def cpu_benchmark(self):
        try:
            print("Advanced CPU Benchmarking...")
            info = cpuinfo.get_cpu_info()
            self.details["CPU Model"] = info['brand_raw']
            
            def prime_sieve(n):
                sieve = [True] * n
                for i in range(2, int(n**0.5) + 1):
                    if sieve[i]:
                        for j in range(i*i, n, i):
                            sieve[j] = False
                return sum(sieve[2:])
            
            def matrix_operations():
                a = np.random.rand(5000, 5000)
                b = np.random.rand(5000, 5000)
                np.dot(a, b)
                np.linalg.inv(a)
            
            def fft_test():
                data = np.random.random(5000000)
                fft(data)
            
            def compression_test():
                data = b'x' * 10000000
                zlib.compress(data, level=9)
            
            def multi_core_test():
                def worker():
                    for _ in range(1000000):
                        _ = 1 + 1
                threads = []
                for _ in range(psutil.cpu_count()):
                    t = threading.Thread(target=worker)
                    t.start()
                    threads.append(t)
                for t in threads:
                    t.join()
            
            tests = [
                ("Prime Sieve", lambda: prime_sieve(10000000)),
                ("Matrix Operations", matrix_operations),
                ("FFT", fft_test),
                ("Compression", compression_test),
                ("Multi-core", multi_core_test)
            ]
            
            results = []
            for name, test in tests:
                start = time.time()
                test()
                end = time.time()
                results.append(end - start)
                self.results[f'CPU {name}'] = end - start
            
            self.scores['cpu'] = 10000000 / np.mean(results)
            print(f"CPU Benchmark Score: {self.scores['cpu']:.2f}")
        except Exception as e:
            print(f"CPU Benchmarking failed: {e}")

    def gpu_benchmark(self):
        try:
            print("Advanced GPU Benchmarking...")
            
            if CUDA_AVAILABLE:
                self.cuda_benchmark()
            else:
                self.opencl_benchmark()
            
            self.ray_tracing_benchmark()
            
        except Exception as e:
            print(f"GPU Benchmarking failed: {e}")

    def cuda_benchmark(self):
        gpus = GPUtil.getGPUs()
        if gpus:
            for gpu in gpus:
                self.details["GPU Model"] = gpu.name
                
                n = 20000
                a = np.random.randn(n, n).astype(np.float32)
                b = np.random.randn(n, n).astype(np.float32)
                
                a_gpu = cuda.mem_alloc(a.nbytes)
                b_gpu = cuda.mem_alloc(b.nbytes)
                c_gpu = cuda.mem_alloc(a.nbytes)
                
                cuda.memcpy_htod(a_gpu, a)
                cuda.memcpy_htod(b_gpu, b)
                
                mod = cuda.SourceModule("""
                __global__ void matmul(float *a, float *b, float *c, int n)
                {
                    int row = blockIdx.y * blockDim.y + threadIdx.y;
                    int col = blockIdx.x * blockDim.x + threadIdx.x;
                    float sum = 0.0f;
                    if (row < n && col < n) {
                        for (int i = 0; i < n; i++) {
                            sum += a[row * n + i] * b[i * n + col];
                        }
                        c[row * n + col] = sum;
                    }
                }
                """)
                
                func = mod.get_function("matmul")
                
                start = time.time()
                func(a_gpu, b_gpu, c_gpu, np.int32(n), block=(32, 32, 1), grid=(int(n/32 + 1), int(n/32 + 1)))
                end = time.time()
                
                cuda_time = end - start
                self.results['GPU CUDA Benchmark'] = cuda_time
                self.scores['gpu'] = 10000000 / cuda_time
                print(f"GPU CUDA Benchmark Score: {self.scores['gpu']:.2f}")

    def opencl_benchmark(self):
        platforms = cl.get_platforms()
        for platform in platforms:
            devices = platform.get_devices(device_type=cl.device_type.GPU)
            for device in devices:
                self.details["GPU Model"] = device.name
                ctx = cl.Context([device])
                queue = cl.CommandQueue(ctx)
                
                n = 10000
                a = np.random.rand(n, n).astype(np.float32)
                b = np.random.rand(n, n).astype(np.float32)
                
                a_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a)
                b_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b)
                c_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, a.nbytes)
                
                program = cl.Program(ctx, """
                __kernel void matmul(__global const float *a, __global const float *b, __global float *c, const int n) {
                    int row = get_global_id(1);
                    int col = get_global_id(0);
                    float sum = 0.0f;
                    if (row < n && col < n) {
                        for (int i = 0; i < n; i++) {
                            sum += a[row * n + i] * b[i * n + col];
                        }
                        c[row * n + col] = sum;
                    }
                }
                """).build()
                
                start = time.time()
                program.matmul(queue, (n, n), None, a_buf, b_buf, c_buf, np.int32(n))
                end = time.time()
                
                opencl_time = end - start
                self.results['GPU OpenCL Benchmark'] = opencl_time
                self.scores['gpu'] = 10000000 / opencl_time
                print(f"GPU OpenCL Benchmark Score: {self.scores['gpu']:.2f}")

    def ray_tracing_benchmark(self):
        try:
            print("Ray Tracing Benchmark...")
            
            def ray_sphere_intersect(ray_origin, ray_direction, sphere_center, sphere_radius):
                oc = ray_origin - sphere_center
                a = np.dot(ray_direction, ray_direction)
                b = 2.0 * np.dot(oc, ray_direction)
                c = np.dot(oc, oc) - sphere_radius * sphere_radius
                discriminant = b * b - 4 * a * c
                return discriminant > 0

            width, height = 1000, 1000
            sphere_center = np.array([0, 0, -1])
            sphere_radius = 0.5

            start = time.time()
            for y in range(height):
                for x in range(width):
                    u = x / width
                    v = y / height
                    ray_origin = np.array([0, 0, 0])
                    ray_direction = np.array([u - 0.5, v - 0.5, -1])
                    ray_direction /= np.linalg.norm(ray_direction)
                    ray_sphere_intersect(ray_origin, ray_direction, sphere_center, sphere_radius)
            end = time.time()

            ray_tracing_time = end - start
            self.results['GPU Ray Tracing Benchmark'] = ray_tracing_time
            self.scores['gpu'] = (self.scores['gpu'] + 10000000 / ray_tracing_time) / 2
            print(f"GPU Ray Tracing Benchmark Score: {10000000 / ray_tracing_time:.2f}")
        except Exception as e:
            print(f"Ray Tracing Benchmark failed: {e}")

    def ram_benchmark(self):
        try:
            print("Advanced RAM Benchmarking...")
            
            def sequential_access(size):
                arr = np.random.rand(size)
                start = time.time()
                np.sum(arr)
                end = time.time()
                return end - start
            
            def random_access(size):
                arr = np.random.rand(size)
                indices = np.random.randint(0, size, size=5000000)
                start = time.time()
                np.sum(arr[indices])
                end = time.time()
                return end - start
            
            sizes = [10000000, 100000000, 500000000]
            seq_times = []
            rand_times = []
            
            for size in sizes:
                seq_times.append(sequential_access(size))
                rand_times.append(random_access(size))
                self.results[f'RAM Sequential {size}'] = seq_times[-1]
                self.results[f'RAM Random {size}'] = rand_times[-1]
            
            self.scores['ram'] = 10000000 / (np.mean(seq_times) + np.mean(rand_times))
            
            ram = psutil.virtual_memory()
            self.details["RAM Size"] = ram.total / (1024 ** 3)  # Convert to GB
            print(f"RAM Size: {self.details['RAM Size']:.2f} GB")
            print(f"RAM Benchmark Score: {self.scores['ram']:.2f}")
        except Exception as e:
            print(f"RAM Benchmarking failed: {e}")

    def ssd_benchmark(self):
        try:
            print("Advanced SSD Benchmarking...")
            
            def write_test(size):
                start = time.time()
                with open("tempfile", "wb") as f:
                    f.write(os.urandom(size))
                end = time.time()
                return end - start
            
            def read_test(size):
                start = time.time()
                with open("tempfile", "rb") as f:
                    f.read()
                end = time.time()
                return end - start
            
            def random_io_test(size, block_size=4096, num_operations=1000):
                with open("tempfile", "r+b") as f:
                    start = time.time()
                    for _ in range(num_operations):
                        pos = random.randint(0, size - block_size)
                        f.seek(pos)
                        if random.choice([True, False]):
                            f.write(os.urandom(block_size))
                        else:
                            f.read(block_size)
                    end = time.time()
                return end - start

            sizes = [100 * 1024 * 1024, 500 * 1024 * 1024, 1024 * 1024 * 1024]  # 100MB, 500MB, 1GB
            write_times = []
            read_times = []
            random_io_times = []

            for size in sizes:
                write_times.append(write_test(size))
                read_times.append(read_test(size))
                random_io_times.append(random_io_test(size))
                self.results[f'SSD Write {size/1024/1024}MB'] = write_times[-1]
                self.results[f'SSD Read {size/1024/1024}MB'] = read_times[-1]
                self.results[f'SSD Random I/O {size/1024/1024}MB'] = random_io_times[-1]

            os.remove("tempfile")

            self.scores['ssd'] = 10000000 / (np.mean(write_times) + np.mean(read_times) + np.mean(random_io_times))
            print(f"SSD Benchmark Score: {self.scores['ssd']:.2f}")
        except Exception as e:
            print(f"SSD Benchmarking failed: {e}")

    def network_benchmark(self):
        try:
            print("Network Benchmarking...")

            def download_speed():
                start = time.time()
                response = requests.get("http://speedtest.ftp.otenet.gr/files/test10Mb.db")
                end = time.time()
                return end - start, len(response.content)

            def latency():
                start = time.time()
                for _ in range(20):
                    requests.get("https://www.google.com")
                end = time.time()
                return (end - start) / 20  # Average latency

            def upload_speed():
                data = 'x' * 1000000  # 1 MB of data
                start = time.time()
                requests.post("https://httpbin.org/post", data=data)
                end = time.time()
                return end - start

            download_time, file_size = download_speed()
            avg_latency = latency()
            upload_time = upload_speed()

            download_speed_mbps = (file_size * 8) / (1000000 * download_time)  # Convert to Mbps
            upload_speed_mbps = 8 / upload_time  # Convert to Mbps

            self.results['Network Download Speed (Mbps)'] = download_speed_mbps
            self.results['Network Upload Speed (Mbps)'] = upload_speed_mbps
            self.results['Network Latency (s)'] = avg_latency

            self.scores['network'] = (1000 * (download_speed_mbps + upload_speed_mbps) / avg_latency) ** 0.5
            print(f"Network Benchmark Score: {self.scores['network']:.2f}")
        except Exception as e:
            print(f"Network Benchmarking failed: {e}")

    def audio_benchmark(self):
        try:
            print("Audio Processing Benchmarking...")

            def generate_sine_wave(freq, duration, sample_rate):
                t = np.linspace(0, duration, int(sample_rate * duration), False)
                return np.sin(2 * np.pi * freq * t)

            sample_rate = 44100
            duration = 20
            freq = 440

            sine_wave = generate_sine_wave(freq, duration, sample_rate)

            start = time.time()
            sd.play(sine_wave, sample_rate)
            sd.wait()
            end = time.time()

            playback_time = end - start
            self.results['Audio Playback (s)'] = playback_time

            start = time.time()
            filtered = signal.lfilter([1, -0.98], [1], sine_wave)
            end = time.time()

            filter_time = end - start
            self.results['Audio Filtering (s)'] = filter_time

            start = time.time()
            fft_result = fft(sine_wave)
            end = time.time()

            fft_time = end - start
            self.results['Audio FFT (s)'] = fft_time

            self.scores['audio'] = 10000000 / (playback_time + filter_time + fft_time)
            print(f"Audio Benchmark Score: {self.scores['audio']:.2f}")
        except Exception as e:
            print(f"Audio Benchmarking failed: {e}")

    def image_benchmark(self):
        try:
            print("Image Processing Benchmarking...")

            image = np.random.randint(0, 256, (7680, 4320, 3), dtype=np.uint8)

            start = time.time()
            blurred = cv2.GaussianBlur(image, (5, 5), 0)
            end = time.time()

            blur_time = end - start
            self.results['Image Blur (s)'] = blur_time

            start = time.time()
            edges = cv2.Canny(image, 100, 200)
            end = time.time()

            edge_time = end - start
            self.results['Image Edge Detection (s)'] = edge_time

            start = time.time()
            resized = cv2.resize(image, (1920, 1080))
            end = time.time()

            resize_time = end - start
            self.results['Image Resize (s)'] = resize_time

            pil_image = Image.fromarray(image)


            filter_time = end - start
            self.results['Image Filter (s)'] = filter_time

            self.scores['image'] = 10000000 / (blur_time + edge_time + resize_time + filter_time)
            print(f"Image Benchmark Score: {self.scores['image']:.2f}")
        except Exception as e:
            print(f"Image Benchmarking failed: {e}")

    def encryption_benchmark(self):
        try:
            print("Encryption Benchmarking...")

            data = b'x' * 50000000  # 50 MB of data

            start = time.time()
            hashlib.sha512(data).hexdigest()
            end = time.time()

            sha512_time = end - start
            self.results['SHA512 Encryption (s)'] = sha512_time

            start = time.time()
            hashlib.blake2b(data).hexdigest()
            end = time.time()

            blake2b_time = end - start
            self.results['BLAKE2b Encryption (s)'] = blake2b_time

            key = os.urandom(32)
            iv = os.urandom(16)
            cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
            encryptor = cipher.encryptor()

            start = time.time()
            encryptor.update(data) + encryptor.finalize()
            end = time.time()

            aes_time = end - start
            self.results['AES Encryption (s)'] = aes_time

            self.scores['encryption'] = 10000000 / (sha512_time + blake2b_time + aes_time)
            print(f"Encryption Benchmark Score: {self.scores['encryption']:.2f}")
        except Exception as e:
            print(f"Encryption Benchmarking failed: {e}")

    def web_benchmark(self):
        try:
            print("Web Performance Benchmarking...")

            def api_request():
                response = requests.get("https://api.github.com/events")
                return response.json()

            start = time.time()
            for _ in range(20):
                api_request()
            end = time.time()

            api_time = (end - start) / 20
            self.results['API Request (s)'] = api_time

            start = time.time()
            response = requests.get("https://www.example.com")
            html_content = response.text
            end = time.time()

            html_time = end - start
            self.results['HTML Fetch (s)'] = html_time

            self.scores['web'] = 10000000 / (api_time * 20 + html_time)
            print(f"Web Benchmark Score: {self.scores['web']:.2f}")
        except Exception as e:
            print(f"Web Benchmarking failed: {e}")

    def ml_benchmark(self):
        try:
            print("Machine Learning Benchmarking... | ALERT: This operation may take a long time. Really long if your hardware does not have an AI processing module.")

            from sklearn.datasets import make_classification
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.neural_network import MLPClassifier

            # Generate a large dataset
            X, y = make_classification(n_samples=100000, n_features=50, n_informative=25, n_redundant=25, random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Random Forest
            start = time.time()
            rf = RandomForestClassifier(n_estimators=50, random_state=42)  # Tahmin edici sayısını azaltın
            rf.fit(X_train, y_train)
            rf.predict(X_test)
            end = time.time()

            rf_time = end - start
            self.results['Random Forest (s)'] = rf_time

            # Logistic Regression
            start = time.time()
            lr = LogisticRegression(random_state=42, max_iter=500)
            lr.fit(X_train, y_train)
            lr.predict(X_test)
            end = time.time()

            lr_time = end - start
            self.results['Logistic Regression (s)'] = lr_time

            # Neural Network
            start = time.time()
            nn = MLPClassifier(hidden_layer_sizes=(50, 25), random_state=42, max_iter=50)
            nn.fit(X_train, y_train)
            nn.predict(X_test)
            end = time.time()

            nn_time = end - start
            self.results['Neural Network (s)'] = nn_time

            self.scores['ml'] = 10000000 / (rf_time + lr_time + nn_time)
            print(f"Machine Learning Benchmark Score: {self.scores['ml']:.2f}")
        except Exception as e:
            print(f"Machine Learning Benchmarking failed: {e}")

    def database_benchmark(self):
        try:
            print("Database Benchmarking...")

            conn = sqlite3.connect(':memory:')
            c = conn.cursor()

            # Create table
            c.execute('''CREATE TABLE benchmark
                        (id INTEGER PRIMARY KEY, data TEXT)''')

            # Insert many rows
            start = time.time()
            for i in range(100000):
                c.execute("INSERT INTO benchmark (data) VALUES (?)", (f'data{i}',))
            conn.commit()
            end = time.time()

            insert_time = end - start
            self.results['Database Insert (s)'] = insert_time

            # Select many rows
            start = time.time()
            c.execute("SELECT * FROM benchmark")
            rows = c.fetchall()
            end = time.time()

            select_time = end - start
            self.results['Database Select (s)'] = select_time

            # Update many rows
            start = time.time()
            c.execute("UPDATE benchmark SET data = 'updated'")
            conn.commit()
            end = time.time()

            update_time = end - start
            self.results['Database Update (s)'] = update_time

            conn.close()

            self.scores['database'] = 10000000 / (insert_time + select_time + update_time)
            print(f"Database Benchmark Score: {self.scores['database']:.2f}")
        except Exception as e:
            print(f"Database Benchmarking failed: {e}")

    def parallel_benchmark(self):
        try:
            print("Parallel Processing Benchmarking...")

            def worker(n):
                for i in range(n):
                    _ = i ** 2

            n = 10000000
            num_processes = multiprocessing.cpu_count()

            start = time.time()
            with multiprocessing.Pool(processes=num_processes) as pool:
                pool.map(worker, [n//num_processes] * num_processes)
            end = time.time()

            parallel_time = end - start
            self.results['Parallel Processing (s)'] = parallel_time

            start = time.time()
            worker(n)
            end = time.time()

            sequential_time = end - start
            self.results['Sequential Processing (s)'] = sequential_time

            speedup = sequential_time / parallel_time
            self.scores['parallel'] = speedup * 1000
            print(f"Parallel Processing Benchmark Score: {self.scores['parallel']:.2f}")
        except Exception as e:
            print(f"Parallel Processing Benchmarking failed: {e}")

    def network_security_benchmark(self):
        try:
            print("Network Security Benchmarking...")

            def ssl_handshake():
                context = ssl.create_default_context()
                with socket.create_connection(('www.python.org', 443)) as sock:
                    with context.wrap_socket(sock, server_hostname='www.python.org') as secure_sock:
                        secure_sock.do_handshake()

            start = time.time()
            for _ in range(10):
                ssl_handshake()
            end = time.time()

            ssl_time = (end - start) / 10
            self.results['SSL Handshake (s)'] = ssl_time

            self.scores['network_security'] = 1000 / ssl_time
            print(f"Network Security Benchmark Score: {self.scores['network_security']:.2f}")
        except Exception as e:
            print(f"Network Security Benchmarking failed: {e}")

    def graphics_benchmark(self):
        try:
            print("Graphics Benchmarking...")

            def draw():
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                glLoadIdentity()
                glTranslatef(-1.5, 0.0, -6.0)
                glBegin(GL_TRIANGLES)
                glVertex3f(0.0, 1.0, 0.0)
                glVertex3f(-1.0, -1.0, 0.0)
                glVertex3f(1.0, -1.0, 0.0)
                glEnd()
                glTranslatef(3.0, 0.0, 0.0)
                glBegin(GL_QUADS)
                glVertex3f(-1.0, 1.0, 0.0)
                glVertex3f(1.0, 1.0, 0.0)
                glVertex3f(1.0, -1.0, 0.0)
                glVertex3f(-1.0, -1.0, 0.0)
                glEnd()
                glutSwapBuffers()

            glutInit()
            glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
            glutInitWindowSize(800, 600)
            glutCreateWindow(b"OpenGL Benchmark")

            glutDisplayFunc(draw)

            start = time.time()
            for _ in range(1000):
                draw()
            end = time.time()

            graphics_time = end - start
            self.results['Graphics Rendering (s)'] = graphics_time

            self.scores['graphics'] = 10000 / graphics_time
            print(f"Graphics Benchmark Score: {self.scores['graphics']:.2f}")
        except Exception as e:
            print(f"Graphics Benchmarking failed: {e}")

    def calculate_total_score(self):
        total_weight = sum(self.weights.values())
        self.total_score = sum(self.scores[key] * self.weights[key] / total_weight for key in self.scores)
        self.details["Total Score"] = self.total_score
        print(f"Total Benchmark Score: {self.total_score:.2f}")

    def plot_results(self):
        categories = list(self.results.keys())
        times = list(self.results.values())

        plt.figure(figsize=(15, 10))
        bars = plt.barh(categories, times, color='skyblue')
        plt.xlabel('Time (s)')
        plt.title('Benchmark Results')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        for bar in bars:
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2, f'{width:.4f}s', 
                    ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('benchmark_results.png', dpi=300)
        plt.close()

    def run_all_benchmarks(self):
        self.cpu_benchmark()
        self.gpu_benchmark()
        self.ram_benchmark()
        self.ssd_benchmark()
        self.network_benchmark()
        self.audio_benchmark()
        self.image_benchmark()
        self.encryption_benchmark()
        self.web_benchmark()
        self.ml_benchmark()
        self.database_benchmark()
        self.parallel_benchmark()
        self.network_security_benchmark()
        self.graphics_benchmark()
        self.calculate_total_score()

    def display_results(self):
        print("\nDetailed Benchmark Results:")
        for key, value in self.details.items():
            print(f"{key}: {value}")
        
        print("\nBenchmark Scores:")
        for key, value in self.scores.items():
            print(f"{key.upper()} Score: {value:.2f}")

    def save_results(self):
        results = {
            "details": self.details,
            "scores": self.scores,
            "results": self.results
        }
        with open("benchmark_results.json", "w") as f:
            json.dump(results, f, indent=4)


    def generate_html_report(self):
        html_content = f"""
        <html>
        <head>
            <title>Benchmark Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                th {{ background-color: #4CAF50; color: white; }}
            </style>
        </head>
        <body>
            <h1>Benchmark Results</h1>
            <h2>System Details</h2>
            <table>
                <tr><th>Detail</th><th>Value</th></tr>
                {''.join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in self.details.items())}
            </table>
            <h2>Benchmark Scores</h2>
            <table>
                <tr><th>Benchmark</th><th>Score</th></tr>
                {''.join(f"<tr><td>{k.upper()}</td><td>{v:.2f}</td></tr>" for k, v in self.scores.items())}
            </table>
            <h2>Detailed Results</h2>
            <table>
                <tr><th>Test</th><th>Time (s)</th></tr>
                {''.join(f"<tr><td>{k}</td><td>{v:.4f}</td></tr>" for k, v in self.results.items())}
            </table>
            <img src="benchmark_results.png" alt="Benchmark Results Graph">
        </body>
        </html>
        """
        with open("benchmark_report.html", "w") as f:
            f.write(html_content)

if __name__ == "__main__":
    bench = AdvancedBench()
    bench.run_all_benchmarks()
    bench.display_results()
    bench.plot_results()
    bench.save_results()
    bench.generate_html_report()
    print("\nBenchmark complete. Results saved to 'benchmark_results.json', 'benchmark_results.png', and 'benchmark_report.html'.")
