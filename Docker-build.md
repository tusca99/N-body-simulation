---

## 🚀 **Quick Guide: Build & Run N-Body Simulation (with Docker)**

### **1. Requirements**

- Docker installed on your system
- NVIDIA drivers installed (only needed for GPU execution)
- Project files in your local folder (with Dockerfile, CMakeLists.txt, etc.)

---

### **2. Build the Docker image**

From the project root (where the Dockerfile is):

```bash
sudo docker build -t nbody-builder .
```

---

### **3. Compile the project**

Run the build (the executable will be written to your project root):

```bash
sudo docker run --rm -v "$PWD":/workspace nbody-builder
```

---

### **4. Where is the executable?**

After building, you will find the executable at:

```
./n-body-simulation
```

---

### **5. Simulation configuration**

Edit the `config.json` file to set:

- **Initialization**:  
  `"selected": "SPIRAL_GALAXY"` (or `"GALAXY"`, `"STELLAR_SYSTEM"`, `"RANDOM"`, `"FROM_FILE"`)
- **Number of particles**:  
  E.g. `"nStars": 15000` for the spiral galaxy
- **Execution mode**:  
  `"executionMode": "CPU"` or `"GPU"`
- **Output**:  
  `"outputMode": "BENCHMARK"` (timing only) or `"FILE_CSV"` (save data)
- **Other parameters**:  
  `"years"`, `"dtYears"`, `"threads"`, etc.

---

### **6. Run the simulation**

#### **CPU**
```bash
./n-body-simulation
```

#### **GPU (if you have NVIDIA and want to force offloading)**
```bash
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia ./n-body-simulation
```

---

### **7. Debug and test inside the container**

To enter the container and test the executable:

```bash
sudo docker run --rm -it --entrypoint /bin/bash -v "$PWD":/workspace nbody-builder
cd /workspace
./n-body-simulation
```

---

### **8. Clean build**

To force a clean build:

```bash
sudo rm -rf build n-body-simulation
```
Then re-run the Docker build.

---

### **9. Notes**

- The Linux executable works on any x86_64 PC with NVIDIA drivers and required libraries.
- For maximum portability, you can create an AppImage (see previous answers).
- For Windows, you need to build natively on Windows.