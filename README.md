# Structure-from-motion-python
Implementation based on SFMedu Princeton COS429: Computer Vision http://vision.princeton.edu/courses/SFMedu/ but on python + numpy

The objective of this project was to understand  the structure from motion problem so i take the MATLAB code from http://vision.princeton.edu/courses/SFMedu/
and translate it in python + numpy. The initial version is just a literal translation from the MATLAB code to python (so expect higher run times, if you want a fast and easy to use software see http://ccwu.me/vsfm/)

Requeriments
Numpy, cv2, https://github.com/dranjan/python-plyfile

For an example just run main.py without any changes. It will generate a point cloud of a jirafe from 5 images included in examples folder. (can take up to 30 m)
