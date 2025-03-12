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
) else (
  echo Build failed with error level %ERRORLEVEL%.
)