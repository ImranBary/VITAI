#!/bin/bash
echo "Building GenerateAndPredict for Raspberry Pi 4B..."

# Clean previous object files
rm -f *.o

# List of source files to compile - explicitly mention Utilities.cpp first
SOURCE_FILES="Utilities.cpp GenerateAndPredict.cpp MedicalDictionaries.cpp HealthIndex.cpp DataStructures.cpp FileProcessing.cpp BatchProcessor.cpp PatientSubsets.cpp FeatureUtils.cpp ResourceMonitor.cpp ThreadPool.cpp SystemResources.cpp BatchProcessorTuner.cpp"

# Find Python include path and library path
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d. -f1-2)
PYTHON_INCLUDE_PATH=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
PYTHON_LIB_PATH=$(python3 -c "import sysconfig; print(sysconfig.get_path('stdlib'))")
PYTHON_CONFIG=$(which python3-config)
PYTHON_LDFLAGS=$(${PYTHON_CONFIG} --ldflags)
PYTHON_INCLUDES=$(${PYTHON_CONFIG} --includes)

# Set compiler and linker flags for Raspberry Pi
INCLUDE_PATH="${PYTHON_INCLUDES}"
COMPILER_FLAGS="-std=c++17 -O3 -mcpu=cortex-a72 -mtune=cortex-a72 -mfpu=neon-fp-armv8 -mfloat-abi=hard"
# Add ARM-specific optimizations
COMPILER_FLAGS="${COMPILER_FLAGS} -ftree-vectorize -ffast-math"
LINKER_FLAGS="${PYTHON_LDFLAGS}"

echo "Using Python ${PYTHON_VERSION}"
echo "Python include path: ${PYTHON_INCLUDE_PATH}"
echo "Python library path: ${PYTHON_LIB_PATH}"

# Compile all files
echo "Compiling with these flags: ${COMPILER_FLAGS} ${INCLUDE_PATH}"
g++ ${COMPILER_FLAGS} ${INCLUDE_PATH} ${SOURCE_FILES} -o GenerateAndPredict ${LINKER_FLAGS}

if [ $? -eq 0 ]; then
  echo "Build successful."
  
  # Copy Python helper scripts to the output directory
  echo "Copying Python helper scripts..."
  for SCRIPT in tabnet_adapter.py model_inspector.py run_feature_verification.py; do
    if [ -f "${SCRIPT}" ]; then
      echo "${SCRIPT} already exists"
    else
      cp "$(dirname "$0")/${SCRIPT}" .
      if [ $? -eq 0 ]; then
        echo "Copied ${SCRIPT}"
      else
        echo "Failed to copy ${SCRIPT}"
      fi
    fi
  done
  
  echo "Python helper scripts ready."
  
  # Optionally add execution permission to the built binary
  chmod +x GenerateAndPredict
  
else
  echo "Build failed with error code $?"
fi

echo
echo "NOTE: This application requires Python with pytorch_tabnet, torch, and sklearn packages."
echo "Please ensure these are installed in your Python environment before running ./GenerateAndPredict"
echo
echo "You can install required packages with:"
echo "pip3 install pytorch_tabnet torch torchvision torchaudio scikit-learn pandas numpy"
