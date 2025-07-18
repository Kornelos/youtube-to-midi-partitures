
import os
from yt_dlp import YoutubeDL
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
import music21

def download_audio(url, output_path="audio"):
    """
    Downloads the audio from a YouTube video.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': os.path.join(output_path, '%(id)s.%(ext)s'),
    }
    with YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info_dict)
        base, _ = os.path.splitext(filename)
        return base + '.mp3'

def transcribe_audio(audio_path, output_path="midi"):
    """
    Transcribes the audio to a MIDI file.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    model_output, midi_data, note_events = predict(audio_path, ICASSP_2022_MODEL_PATH)
    
    output_midi_path = os.path.join(output_path, os.path.splitext(os.path.basename(audio_path))[0] + ".mid")
    midi_data.write(output_midi_path)
    return output_midi_path

def create_sheet_music(midi_path, output_path="partitures"):
    """
    Creates sheet music from a MIDI file.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    score = music21.converter.parse(midi_path)
    output_sheet_path = os.path.join(output_path, os.path.splitext(os.path.basename(midi_path))[0] + ".xml")
    score.write('musicxml', fp=output_sheet_path)
    return output_sheet_path

def simplify_midi(midi_path, output_path="midi_simplified"):
    """
    Simplifies a MIDI file by quantizing and merging notes.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    score = music21.converter.parse(midi_path)
    
    # Quantize the score to the nearest 16th note
    quantized_score = score.quantize(quarterLengthDivisors=(4,))

    # Merge consecutive notes/chords with the same pitch
    for part in quantized_score.parts:
        # Get a flat list of notes and chords
        elements = list(part.flatten().notesAndRests)
        notes_to_remove = []
        
        i = 0
        while i < len(elements) - 1:
            el1 = elements[i]
            el2 = elements[i+1]

            # Skip rests
            if isinstance(el1, music21.note.Rest) or isinstance(el2, music21.note.Rest):
                i += 1
                continue

            # Check if elements are close enough to merge
            if el2.offset > el1.offset + el1.duration.quarterLength + 0.1:
                i += 1
                continue

            are_notes = isinstance(el1, music21.note.Note) and isinstance(el2, music21.note.Note)
            are_chords = isinstance(el1, music21.chord.Chord) and isinstance(el2, music21.chord.Chord)

            if (are_notes and el1.pitch == el2.pitch) or \
               (are_chords and el1.normalOrder == el2.normalOrder):
                # Extend the duration of the first element
                el1.duration.quarterLength += el2.duration.quarterLength
                # Mark the second element for removal from the original stream
                notes_to_remove.append(el2)
                # Remove the second element from our temporary list so we don't process it again
                elements.pop(i + 1)
            else:
                i += 1
        
        # Remove the merged notes from the actual score part
        for note in notes_to_remove:
            part.remove(note, recurse=True)

    # Remove very short notes (less than a 32nd note)
    for el in quantized_score.flatten().notesAndRests:
        if isinstance(el, music21.note.Note) and el.duration.quarterLength < 0.125:
            el.activeSite.remove(el)

    simplified_midi_path = os.path.join(output_path, os.path.splitext(os.path.basename(midi_path))[0] + "_simplified.mid")
    quantized_score.write('midi', fp=simplified_midi_path)
    return simplified_midi_path

import argparse

def download_audio(url, output_path="audio", start_time=None, end_time=None):
    """
    Downloads a specific segment of audio from a YouTube video.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Use the video ID for a clean filename
    output_template = os.path.join(output_path, '%(id)s.%(ext)s')

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_template,
    }

    # If start or end times are provided, download a clip
    if start_time is not None or end_time is not None:
        postprocessor_args = []
        if start_time is not None:
            postprocessor_args.extend(['-ss', str(start_time)])
        if end_time is not None:
            postprocessor_args.extend(['-to', str(end_time)])

        ydl_opts['postprocessors'] = [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }]
        ydl_opts['postprocessor_args'] = postprocessor_args

    else:
        # Default behavior: download the whole audio
        ydl_opts['postprocessors'] = [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }]

    with YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        # Construct the final path to the mp3 file
        base, _ = os.path.splitext(ydl.prepare_filename(info_dict))
        return base + '.mp3'

def transcribe_audio(audio_path, output_path="midi"):
    """
    Transcribes the audio to a MIDI file.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    model_output, midi_data, note_events = predict(audio_path, ICASSP_2022_MODEL_PATH)
    
    output_midi_path = os.path.join(output_path, os.path.splitext(os.path.basename(audio_path))[0] + ".mid")
    midi_data.write(output_midi_path)
    return output_midi_path

def create_sheet_music(midi_path, output_path="partitures"):
    """
    Creates sheet music from a MIDI file.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    score = music21.converter.parse(midi_path)
    output_sheet_path = os.path.join(output_path, os.path.splitext(os.path.basename(midi_path))[0] + ".xml")
    score.write('musicxml', fp=output_sheet_path)
    return output_sheet_path

def simplify_midi(midi_path, output_path="midi_simplified"):
    """
    Simplifies a MIDI file by quantizing and merging notes.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    score = music21.converter.parse(midi_path)
    
    # Quantize the score to the nearest 16th note
    quantized_score = score.quantize(quarterLengthDivisors=(4,))

    # Merge consecutive notes/chords with the same pitch
    for part in quantized_score.parts:
        # Get a flat list of notes and chords
        elements = list(part.flatten().notes)
        notes_to_remove = []
        
        i = 0
        while i < len(elements) - 1:
            el1 = elements[i]
            el2 = elements[i+1]

            # Check if elements are close enough to merge
            if el2.offset > el1.offset + el1.duration.quarterLength + 0.1:
                i += 1
                continue

            are_notes = isinstance(el1, music21.note.Note) and isinstance(el2, music21.note.Note)
            are_chords = isinstance(el1, music21.chord.Chord) and isinstance(el2, music21.chord.Chord)

            if (are_notes and el1.pitch == el2.pitch) or \
               (are_chords and el1.normalOrder == el2.normalOrder):
                # Extend the duration of the first element
                el1.duration.quarterLength += el2.duration.quarterLength
                # Mark the second element for removal from the original stream
                notes_to_remove.append(el2)
                # Remove the second element from our temporary list so we don't process it again
                elements.pop(i + 1)
            else:
                i += 1
        
        # Remove the merged notes from the actual score part
        for note in notes_to_remove:
            part.remove(note, recurse=True)

    # Remove very short notes (less than a 32nd note)
    for el in quantized_score.flatten().notes:
        if el.duration.quarterLength < 0.125:
            el.activeSite.remove(el)

    simplified_midi_path = os.path.join(output_path, os.path.splitext(os.path.basename(midi_path))[0] + "_simplified.mid")
    quantized_score.write('midi', fp=simplified_midi_path)
    return simplified_midi_path

def main():
    """
    Main function to process the YouTube video.
    """
    parser = argparse.ArgumentParser(description='Transcribe piano audio from YouTube to MIDI and sheet music.')
    parser.add_argument('url', type=str, help='The YouTube URL to process.')
    parser.add_argument('--start', type=int, help='Start time in seconds.')
    parser.add_argument('--end', type=int, help='End time in seconds.')
    args = parser.parse_args()

    print("Downloading audio...")
    audio_file = download_audio(args.url, start_time=args.start, end_time=args.end)
    
    print("Transcribing audio to MIDI...")
    midi_file = transcribe_audio(audio_file)

    print("Simplifying MIDI...")
    simplified_midi_file = simplify_midi(midi_file)
    
    print("Creating sheet music from simplified MIDI...")
    sheet_music_file = create_sheet_music(simplified_midi_file)
    
    print(f"\nSuccessfully created MIDI file: {midi_file}")
    print(f"Successfully created simplified MIDI file: {simplified_midi_file}")
    print(f"Successfully created sheet music: {sheet_music_file}")

if __name__ == "__main__":
    main()
