# Welcome to FLOW

## Usage

Flow is designed to estimate velocity fields from successive frames in a video.
Usage is straightforward. 

Determine the frame rate for a given video: 
```unix
$ ffmpeg -i input.mp4 
```
Create consecutively numbered images for the given video, providing the previously determined frame rate: 
```unix
$ ffmpeg -i input.mp4 -vf fps=29.97 img%d.jpeg
```
Run `predict_vector_field.py` script with arguments `FOLDER_NAME` where images are located and `FRAME_RATE`.

This script will loop through the available images, estimating a velocity vector field 
at grid points distributed across the image. It saves the field data in a data file
called `fields.dat`.

Running `generate_video_frames.py` will load this data file and produce new video frames 
with the estimated vector field superimposed, which are by default stored in `/videos`.

Once you have generated the raw frames, they can be compiled into a video using FFMPEG:

```unix
$ ffmpeg -f image2 -r 30 -i image_%04d.png -vb 20M -vcodec mpeg4 -y movie_name.mp4
```


