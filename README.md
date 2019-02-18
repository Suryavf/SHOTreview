# SHOT
C++ implementation of the SHOT 3D descriptor
	
SHOT is an implementation of the algorithms described in: 
* F. Tombari, S. Salti and L. Di Stefano "Unique Signatures of Histograms for Local Surface Description", The 11th IEEE European Conference on Computer Vision (ECCV) 2010.
* F. Tombari, S. Salti, L. Di Stefano, "A combined texture-shape descriptor for enhanced 3D feature matching", IEEE International Conference on Image Processing (ICIP), September 11-14, Brussels, Belgium, 2011.
* S. Salti, F. Tombari, L. Di Stefano, "SHOT: Unique Signatures of Histograms for Surface and Texture Description", Computer Vision and Image Understanding, May, 2014.

SHOT has been developed by the Computer Vision Laboratory of the University of Bologna (http://www.vision.disi.unibo.it)
Datasets used in the experiments reported in the papers are available at the SHOT project [official page](http://vision.disi.unibo.it/research/80-shot)

# Dependencies
* OpenCV (3.0 and above)
* VTK (5.10 and above)

# Compilation and execution
Compile:
```console
mkdir build
cd build
cmake ..
make
```
Execute:
```console
../bin/SHOT
```
