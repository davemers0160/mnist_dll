#MNIST Net Library#

This repo is an experiment to build a windows/linux library (.dll/.so) that uses [dlib](dlib.net) built network to do MNIST digit classification.

The purpose of this project is to build a library that can be used by other programs without having to complie dlib or the network each time.  Thi library can be used in C/C++, Matlab, Python, etc...

#Dependencies#

The following is libraries/repos are required to compile this project:
- [dlib](http://dlib.net/files/dlib-19.18.zip) or the github repository https://github.com/davisking/dlib
- [Common Repo](https://github.com/davemers0160/Common.git)

#Building the Project#

To build the project ensure that you have the correct dependencies.  You will need to slightly modify the CMakeList.txt file to point to the correct locations for the dlib library can the Common repo.

Once the CMakeLists.txt file is correct 

Windows/Visual Studio:
- Make a folder called "build" in the project folder
- Open up the cmake-gui and point to the locations for the CMakeLists.txt file and the build directory
  - Select the 64-bit version of the VS20XX compiler
  - Enter host=x64
  - Select the "Configure" Button
    - Make any modifications (use AVX, CUDA, etc...)
  - Redo the generation if you make any changes  
  - Select the "Generate" Button
  - Select the "Open Projct" Button
- In Visual Studio build the release version of the library.  A dll file should be created. 


Linux
```
mkdir build
cd build
cmake ..
cmake --build . --config Release -- -j4
```
