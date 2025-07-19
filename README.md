# YouTube Piano Parser

This project provides a Python script to download audio from YouTube piano videos, transcribe the piano performance into a MIDI file, and then generate simplified sheet music (MusicXML). It also supports processing specific segments of a video using start and end timestamps.

## Features

*   **YouTube Audio Download:** Downloads the audio track from a given YouTube video.
*   **Custom Naming:** Specify a custom name for the output files.
*   **Timestamp Support:** Process only a specific segment of the video by providing start and end times in seconds.
*   **Multiple Transcription Models:** Converts the downloaded audio into a MIDI file using either the `basic-pitch` model locally or the more advanced `MT3` model via a public API.
*   **Piano Track Filtering:** Automatically isolates the piano track from multi-instrument transcriptions.
*   **Optional MIDI Simplification:** Automatically quantizes notes, merges consecutive notes of the same pitch, and removes very short notes to produce cleaner, more readable MIDI and sheet music. This step is optional.
*   **Sheet Music Generation:** Creates MusicXML files from the simplified MIDI, which can be opened in music notation software.

## How to Use

1.  **Setup:** Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the Script:**
    ```bash
    .\venv\Scripts\python.exe main.py <youtube_url> [--name <filename>] [--start <seconds>] [--end <seconds>] [--model <model_name>] [--no-simplification] [--keep-all-instruments]
    ```
    *   `<youtube_url>`: The URL of the YouTube video you want to process.
    *   `--name <filename>` (optional): A custom name for the output files (without extension). Defaults to the YouTube video ID.
    *   `--start <seconds>` (optional): The start time of the audio segment in seconds.
    *   `--end <seconds>` (optional): The end time of the audio segment in seconds.
    *   `--model <model_name>` (optional): The transcription model to use. Choose between `mt3` (default, higher quality, requires internet) and `basic-pitch` (local, faster).
    *   `--no-simplification` (optional): Skip the MIDI simplification process.
    *   `--keep-all-instruments` (optional): Keep all instrument tracks from the transcription (by default, only the piano track is kept).

    **Examples:**
    *   Process the entire video, keeping only the piano track and skipping simplification:
        ```bash
        .\venv\Scripts\python.exe main.py https://www.youtube.com/watch?v=XY2lN9CAuuQ --name my-song --no-simplification
        ```
    *   Process a segment with the `basic-pitch` model:
        ```bash
        .\venv\Scripts\python.exe main.py https://www.youtube.com/watch?v=XY2lN9CAuuQ --start 10 --end 30 --model basic-pitch
        ```

## Output Files

The script will create the following directories and files, using either the custom name you provide or the YouTube video ID:

*   `audio/`: Contains the downloaded MP3 audio file.
*   `midi/`: Contains the raw MIDI transcription. By default, this will be filtered to only include the piano track.
*   `midi_simplified/`: Contains the cleaned and simplified MIDI file (if simplification is not skipped).
*   `partitures/`: Contains the generated MusicXML sheet music file.

## Viewing Sheet Music

The generated `.xml` files are in MusicXML format. To view, play, or print them, you will need a music notation software. We recommend **MuseScore**, which is free and open-source:

1.  **Download MuseScore:** [https://musescore.org/en/download](https://musescore.org/en/download)
2.  **Open the file:** In MuseScore, go to `File` > `Open...` and select the `.xml` file from the `partitures/` directory.
3.  **Export to PDF:** Once opened, you can export the score to PDF via `File` > `Export...`.
