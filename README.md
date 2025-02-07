# SHOT
C++ implementation of the SHOT 3D descriptor based on https://github.com/fedassa/SHOT
	
SHOT is an implementation of the algorithms described in: 
* F. Tombari, S. Salti and L. Di Stefano "Unique Signatures of Histograms for Local Surface Description", The 11th IEEE European Conference on Computer Vision (ECCV) 2010.
* F. Tombari, S. Salti, L. Di Stefano, "A combined texture-shape descriptor for enhanced 3D feature matching", IEEE International Conference on Image Processing (ICIP), September 11-14, Brussels, Belgium, 2011.
* S. Salti, F. Tombari, L. Di Stefano, "SHOT: Unique Signatures of Histograms for Surface and Texture Description", Computer Vision and Image Understanding, May, 2014.

# Dependencies
* OpenCV (3.0 and above)
* VTK (6.0 and above)

# Compilation
```console
mkdir build
cd build
cmake ..
make
```
# Execution
```console
../bin/SHOT
```

Click on the image to see the test video:
[![Watch the video](https://i.imgur.com/pLNcP0I.png)](https://youtu.be/uKc07AzNMa0)