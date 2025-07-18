# YouTube Piano Parser

This project provides a Python script to download audio from YouTube piano videos, transcribe the piano performance into a MIDI file, and then generate simplified sheet music (MusicXML). It also supports processing specific segments of a video using start and end timestamps.

## Features

*   **YouTube Audio Download:** Downloads the audio track from a given YouTube video.
*   **Timestamp Support:** Process only a specific segment of the video by providing start and end times in seconds.
*   **MIDI Transcription:** Converts the downloaded audio into a MIDI file using the `basic-pitch` machine learning model.
*   **MIDI Simplification:** Automatically quantizes notes, merges consecutive notes of the same pitch, and removes very short notes to produce cleaner, more readable MIDI and sheet music.
*   **Sheet Music Generation:** Creates MusicXML files from the simplified MIDI, which can be opened in music notation software.

## How to Use

1.  **Setup:** Follow the instructions in `SETUP.md` to set up your environment and install dependencies.
2.  **Run the Script:**
    ```bash
    .\venv\Scripts\python.exe main.py <youtube_url> [--start <seconds>] [--end <seconds>]
    ```
    *   `<youtube_url>`: The URL of the YouTube video you want to process.
    *   `--start <seconds>` (optional): The start time of the audio segment in seconds.
    *   `--end <seconds>` (optional): The end time of the audio segment in seconds.

    **Examples:**
    *   Process the entire video:
        ```bash
        .\venv\Scripts\python.exe main.py https://www.youtube.com/watch?v=XY2lN9CAuuQ
        ```
    *   Process from 10 seconds to 30 seconds:
        ```bash
        .\venv\Scripts\python.exe main.py https://www.youtube.com/watch?v=XY2lN9CAuuQ --start 10 --end 30
        ```

## Output Files

The script will create the following directories and files:

*   `audio/`: Contains the downloaded MP3 audio file (named after the YouTube video ID).
*   `midi/`: Contains the raw MIDI transcription (named after the YouTube video ID).
*   `midi_simplified/`: Contains the cleaned and simplified MIDI file (named `[video_id]_simplified.mid`).
*   `partitures/`: Contains the generated MusicXML sheet music file (named `[video_id]_simplified.xml`).

## Viewing Sheet Music

The generated `.xml` files are in MusicXML format. To view, play, or print them, you will need a music notation software. We recommend **MuseScore**, which is free and open-source:

1.  **Download MuseScore:** [https://musescore.org/en/download](https://musescore.org/en/download)
2.  **Open the file:** In MuseScore, go to `File` > `Open...` and select the `.xml` file from the `partitures/` directory.
3.  **Export to PDF:** Once opened, you can export the score to PDF via `File` > `Export...`.
