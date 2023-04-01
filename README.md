# 3DTalkingFaceMesh: Generate Lip-Synchronized 3D Face Mesh from Speech
3DTalkingFaceMesh is a deep learning model that generates 3D facial meshes, which are lip-synchronized to speech input. Given a raw audio file, the model outputs a sequence of facial landmarks that adhere to the [MediaPipe FaceMesh](https://google.github.io/mediapipe/solutions/face_mesh.html#:~:text=MediaPipe%20Face%20Mesh%20is%20a,for%20a%20dedicated%20depth%20sensor.) standard. The model is built upon a Transformer Encoder-Decoder architecture, with the Encoder utilizing [OpenAI Whisper](https://huggingface.co/openai/whisper-base) for speech recognition.


## Dependencies
To set up the necessary environment, please install the dependencies listed in the `requirements.txt` file. Additionally, you will need `ffmpeg` for video and audio processing.

## Dataset

### Obama Weekly Address
The [Obama Weekly Address](https://obamawhitehouse.archives.gov/briefing-room/weekly-address) dataset consists of high-quality, fixed-pose speech videos featuring former President Barack Obama. You can find the list of YouTube video IDs in `data/video_links.txt`.

To download the raw videos, execute the following command:

```
python download_video.py
```

This command creates a `data/video` folder that contains all the downloaded video files.

### VoxCeleb2
The [VoxCeleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html) dataset comprises over 1 million wild YouTube videos featuring more than 6,000 celebrities speaking. However, we found that training the model solely on Obama speech videos was sufficient for zero-shot adaptation to other voices and languages, thanks to the utilization of Whisper Speech Embeddings.

## Data Preparation

1. **Extract Landmarks**: To extract facial landmarks from the videos, run:

    ```
    python utils/extract_landmarks.py
    ```

    This command creates a `data/landmarks` folder containing extracted landmarks for each video, and a `data/validity` folder containing frame-specific labels that indicate whether the corresponding landmarks are valid. A frame is considered invalid if no faces are detected or if multiple faces are present.

2. **Extract Audio**: To extract audio from the raw videos and save it as a numpy npy array format, run: 

    ```
    python utils/load_audio.py
    ```

    This will create the `data/audio` folder containing extracted audio wav npy files from raw videos


## Model Training

To train the 3DTalkingFaceMesh model, execute the following command:

```
python main.py
```

You can find a list of available arguments by checking the main function. For example:

```
python main.py --seed 42 --lr 0.003 --feature_dim 256 --optimizer lion --audio_encoder whisper
```

This command trains the model with a fixed seed of 42, a learning rate of 0.003, a hidden Transformer decoder dimension of 256, the LION optimizer, and Whisper speech embeddings.



