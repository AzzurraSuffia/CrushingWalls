# CrushingWalls
**CrushingWalls** is an interactive system. It displays two virtual walls on the right and left sides of a visitor’s bounding rectangle, attempting to crush their body. The visitor can expand this rectangle to gradually reveal the background of the room. When the visitor’s energy falls below a predefined threshold, the visitor disappears and the two walls collapse toward the center, merging in the middle.

## Requirements

To run this project, the following software and libraries are required:

- **Python**: 3.11.9 or higher
- **MediaPipe**: 0.10.32
- **SciPy**: 1.17.1
- **NumPy**: 2.4.2
- **OpenCV (opencv-python)**: 4.13.0.92

The Python packages can be installed using `pip`:

```bash
pip install mediapipe==0.10.32 scipy==1.17.1 numpy==2.4.2 opencv-python==4.13.0.92
```

## Installation Instructions
Clone this repository:
```bash
git clone https://github.com/AzzurraSuffia/CrushingWalls.git
```

## Run the application
To run the project:

1. Navigate to the project folder:

```bash
cd <repository_folder>
```
2. Change input options:
    - Live camera input: <br>
    Make sure the `LIVE_INPUT` constant in `config/constants.py` is set to `True`.
    - Pre-recorded video input: <br>
        1. Copy the video file into the `videos` folder.
        2. Set the `VIDEO_PATH` constant in `config/constants.py` to the relative path of your video, e.g.:
        ```bash
        VIDEO_PATH = "videos\\my_video.mp4"
        ```
3. Set camera parameters and background: <br>
    - Adjust the `FPS` constant in `config/constants.py` to match your camera’s frame rate.
    - To use a fixed background image, set it in `main.py` (resize the image to 640×480).
Otherwise, the system will automatically capture the first frame as the background.
4. (Optional) Enable debug mode:<br>
Set the `DEBUG` constant in `config/constants.py` to `True` if you want extra debug output.
5. Run the main script:
```bash
python src/main.py
```