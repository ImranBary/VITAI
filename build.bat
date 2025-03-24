REM filepath: c:\Users\imran\Documents\VITAI\build.bat
@echo off
echo Building GenerateAndPredict...

REM Clean previous object files
if exist *.obj del *.obj

REM List of source files to compile - explicitly mention Utilities.cpp first
set SOURCE_FILES=Utilities.cpp GenerateAndPredict.cpp MedicalDictionaries.cpp HealthIndex.cpp DataStructures.cpp FileProcessing.cpp BatchProcessor.cpp PatientSubsets.cpp FeatureUtils.cpp ResourceMonitor.cpp ThreadPool.cpp SystemResources.cpp BatchProcessorTuner.cpp

REM Set compiler and linker flags
set INCLUDE_PATH=/I"C:\Users\imran\miniconda3\envs\tf_gpu_env\include"
set LIB_PATH=/LIBPATH:"C:\Users\imran\miniconda3\envs\tf_gpu_env\libs"
set COMPILER_FLAGS=/EHsc /std:c++17
set LINKER_FLAGS=python39.lib /MACHINE:X64

REM Compile all files in a single command
echo Compiling with these flags: %COMPILER_FLAGS% %INCLUDE_PATH%
cl %COMPILER_FLAGS% %INCLUDE_PATH% %SOURCE_FILES% /Fe:GenerateAndPredict.exe /link %LIB_PATH% %LINKER_FLAGS%

if %ERRORLEVEL% == 0 (
  echo Build successful.
  
  REM Copy Python helper scripts to the output directory
  echo Copying Python helper scripts...
  if exist "tabnet_adapter.py" echo tabnet_adapter.py already exists
  if not exist "tabnet_adapter.py" copy "%~dp0tabnet_adapter.py" .
  
  if exist "model_inspector.py" echo model_inspector.py already exists
  if not exist "model_inspector.py" copy "%~dp0model_inspector.py" .
  
  if exist "run_feature_verification.py" echo run_feature_verification.py already exists
  if not exist "run_feature_verification.py" copy "%~dp0run_feature_verification.py" .
  
  echo Python helper scripts ready.
) else (
  echo Build failed with error level %ERRORLEVEL%.
)

echo.
echo NOTE: This application requires Python with pytorch_tabnet, torch, and sklearn packages.
echo Please ensure these are installed in your Python environment before running GenerateAndPredict.exe