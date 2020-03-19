# Welcome to FLOW

## Usage

Flow is designed to estimate velocity fields from successive frames in a video.
Usage is straightforward. Put your consecutively numbered images in the
`waves2997fps/` folder (or other designated spot) and run the
`predict_vector_field.py` script. 

This script will loop through the available images, estimating a velocity vector field 
at grid points distributed across the image. It saves the field data in a data file
called `fields.dat`.

Running `generate_video_frames.py` will load this data file and produce new video frames 
with the estimated vector field superimposed, which are by default stored in `/videos`.

Once you have generated the raw frames, they can be compiled into a video using FFMPEG:

```unix
$ ffmpeg -f image2 -r 30 -i image_%04d.png -vb 20M -vcodec mpeg4 -y movie_name.mp4
```


