CXX = cl
CXXFLAGS = /EHsc /std:c++17 /I"C:\Users\imran\miniconda3\envs\tf_gpu_env\include"
LDFLAGS = /LIBPATH:"C:\Users\imran\miniconda3\envs\tf_gpu_env\libs" python39.lib /MACHINE:X64

all: GenerateAndPredict.exe

GenerateAndPredict.exe: GenerateAndPredict.obj Utilities.obj
	$(CXX) $^ /link $(LDFLAGS) /out:$@

GenerateAndPredict.obj: GenerateAndPredict.cpp
	$(CXX) $(CXXFLAGS) /c $<

Utilities.obj: Utilities.cpp
	$(CXX) $(CXXFLAGS) /c $<

clean:
	del *.obj *.exe
