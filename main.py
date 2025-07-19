import os
from yt_dlp import YoutubeDL
import music21
import argparse
import numpy as np
from collections import Counter
from gradio_client import Client, handle_file
import shutil

def download_audio(url, output_path="audio", start_time=None, end_time=None, name=None):
    """
    Downloads a specific segment of audio from a YouTube video.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if name:
        output_template = os.path.join(output_path, f'{name}.%(ext)s')
    else:
        output_template = os.path.join(output_path, '%(id)s.%(ext)s')

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_template,
    }

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
        ydl_opts['postprocessors'] = [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }]

    with YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        base, _ = os.path.splitext(ydl.prepare_filename(info_dict))
        return base + '.mp3'

def transcribe_audio(audio_path, output_path="midi"):
    """
    Transcribes the audio to a MIDI file using basic-pitch.
    """
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    model_output, midi_data, note_events = predict(audio_path, ICASSP_2022_MODEL_PATH)
    
    output_midi_path = os.path.join(output_path, os.path.splitext(os.path.basename(audio_path))[0] + ".mid")
    midi_data.write(output_midi_path)
    return output_midi_path

def transcribe_audio_mt3_api(audio_path, output_path="midi"):
    """
    Transcribes the audio to a MIDI file using the MT3 Hugging Face Space.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print("Connecting to MT3 API...")
    client = Client("NeoPy/MT3", verbose=False)
    print("Transcribing... (this may take a while)")
    result = client.predict(
        audio=handle_file(audio_path),
        api_name="/inference"
    )
    
    output_midi_path = os.path.join(output_path, os.path.splitext(os.path.basename(audio_path))[0] + ".mid")
    # The result is a path to a temporary file, so we need to copy it
    shutil.copyfile(result, output_midi_path)
    
    return output_midi_path

def filter_piano_tracks(midi_path):
    """
    Filters a MIDI file to keep only the piano track(s).
    If no specific piano track is found, it keeps the track with the most notes.
    """
    score = music21.converter.parse(midi_path)
    piano_parts = []
    
    for part in score.parts:
        if part.instrument is not None and 'Piano' in part.instrument.instrumentName:
            piano_parts.append(part)

    if not piano_parts:
        # If no instrument is explicitly named "Piano", find the part with the most notes
        max_notes = 0
        best_part = None
        for part in score.parts:
            num_notes = len(part.flatten().notes)
            if num_notes > max_notes:
                max_notes = num_notes
                best_part = part
        if best_part is not None:
            piano_parts.append(best_part)

    if piano_parts:
        new_score = music21.stream.Score()
        for part in piano_parts:
            new_score.insert(0, part)
        new_score.write('midi', fp=midi_path)

def detect_key_signature(score):
    """
    Detect the key signature of the piece using music21's key analysis.
    """
    try:
        # Analyze the key using music21's built-in key detection
        key = score.analyze('key')
        return key
    except:
        # Fallback to C major if detection fails
        return music21.key.Key('C', 'major')

def filter_notes_by_confidence(score, velocity_threshold=40):
    """
    Remove notes with low velocity (confidence) from the score.
    """
    for part in score.parts:
        notes_to_remove = []
        for element in part.flatten().notes:
            if hasattr(element, 'volume') and element.volume.velocity:
                if element.volume.velocity < velocity_threshold:
                    notes_to_remove.append(element)
            elif isinstance(element, music21.chord.Chord):
                # For chords, check if most notes have low velocity
                low_velocity_count = 0
                for note in element.notes:
                    if hasattr(note, 'volume') and note.volume.velocity:
                        if note.volume.velocity < velocity_threshold:
                            low_velocity_count += 1
                
                if low_velocity_count > len(element.notes) / 2:
                    notes_to_remove.append(element)
        
        for note in notes_to_remove:
            part.remove(note, recurse=True)
    
    return score

def simplify_chords(score, max_chord_notes=3):
    """
    Simplify chords by keeping only the most important notes.
    """
    for part in score.parts:
        for element in part.flatten().notes:
            if isinstance(element, music21.chord.Chord) and len(element.notes) > max_chord_notes:
                # Keep root, third, fifth, and seventh if present
                pitches = [n.pitch for n in element.notes]
                pitches.sort(key=lambda x: x.midi)
                
                # Try to keep important chord tones
                simplified_pitches = []
                if len(pitches) >= 1:
                    simplified_pitches.append(pitches[0])  # Root (lowest)
                if len(pitches) >= 3:
                    simplified_pitches.append(pitches[2])  # Third
                if len(pitches) >= 5 and len(simplified_pitches) < max_chord_notes:
                    simplified_pitches.append(pitches[4])  # Fifth
                
                # Fill remaining slots with highest notes
                remaining_slots = max_chord_notes - len(simplified_pitches)
                for i in range(remaining_slots):
                    if len(pitches) - 1 - i >= 0:
                        pitch = pitches[-(i+1)]
                        if pitch not in simplified_pitches:
                            simplified_pitches.append(pitch)
                
                # Replace the chord
                element.pitches = simplified_pitches
    
    return score

def quantize_to_common_rhythms(score):
    """
    Quantize to common rhythms (whole, half, quarter, eighth, sixteenth notes).
    """
    common_durations = [4.0, 2.0, 1.0, 0.5, 0.25]  # Whole, half, quarter, eighth, sixteenth
    
    for part in score.parts:
        for element in part.flatten().notes:
            current_duration = element.duration.quarterLength
            
            # Find the closest common duration
            closest_duration = min(common_durations, key=lambda x: abs(x - current_duration))
            
            # Only change if it's reasonably close
            if abs(closest_duration - current_duration) < current_duration * 0.3:
                element.duration.quarterLength = closest_duration
    
    return score

def remove_ornaments_and_grace_notes(score):
    """
    Remove very short notes that are likely ornaments or grace notes.
    """
    min_duration = 0.1  # Remove notes shorter than this
    
    for part in score.parts:
        notes_to_remove = []
        for element in part.flatten().notes:
            if element.duration.quarterLength < min_duration:
                notes_to_remove.append(element)
        
        for note in notes_to_remove:
            part.remove(note, recurse=True)
    
    return score

def merge_tied_notes(score):
    """
    Merge notes that should be tied together.
    """
    for part in score.parts:
        elements = list(part.flatten().notes)
        notes_to_remove = []
        
        i = 0
        while i < len(elements) - 1:
            el1 = elements[i]
            el2 = elements[i + 1]
            
            # Check if we can merge these notes
            can_merge = False
            
            if isinstance(el1, music21.note.Note) and isinstance(el2, music21.note.Note):
                # Same pitch and very close in time
                if (el1.pitch == el2.pitch and 
                    abs(el2.offset - (el1.offset + el1.duration.quarterLength)) < 0.1):
                    can_merge = True
            
            if can_merge:
                # Extend the duration of the first note
                el1.duration.quarterLength += el2.duration.quarterLength
                notes_to_remove.append(el2)
                elements.pop(i + 1)
            else:
                i += 1
        
        for note in notes_to_remove:
            part.remove(note, recurse=True)
    
    return score

def separate_hands(score):
    """
    Attempt to separate notes into left and right hand parts based on pitch.
    """
    # Find the median pitch to use as a split point
    all_pitches = []
    for part in score.parts:
        for element in part.flatten().notes:
            if isinstance(element, music21.note.Note):
                all_pitches.append(element.pitch.midi)
            elif isinstance(element, music21.chord.Chord):
                for note in element.notes:
                    all_pitches.append(note.pitch.midi)
    
    if not all_pitches:
        return score
    
    # Use middle C (60) as default split, or median if it's reasonable
    split_point = 60  # Middle C
    if all_pitches:
        median_pitch = np.median(all_pitches)
        if 48 <= median_pitch <= 72:  # Reasonable range around middle C
            split_point = median_pitch
    
    # Create new parts
    right_hand = music21.stream.Part()
    left_hand = music21.stream.Part()
    
    # Add instruments
    right_hand.insert(0, music21.instrument.Piano())
    left_hand.insert(0, music21.instrument.Piano())
    
    # Distribute notes
    for part in score.parts:
        for element in part.flatten().notes:
            element_copy = element.__deepcopy__()
            
            if isinstance(element, music21.note.Note):
                if element.pitch.midi >= split_point:
                    right_hand.insert(element.offset, element_copy)
                else:
                    left_hand.insert(element.offset, element_copy)
            elif isinstance(element, music21.chord.Chord):
                # Split chords based on average pitch
                avg_pitch = sum(n.pitch.midi for n in element.notes) / len(element.notes)
                if avg_pitch >= split_point:
                    right_hand.insert(element.offset, element_copy)
                else:
                    left_hand.insert(element.offset, element_copy)
    
    # Create new score with separated hands
    new_score = music21.stream.Score()
    if len(right_hand.notes) > 0:
        new_score.append(right_hand)
    if len(left_hand.notes) > 0:
        new_score.append(left_hand)
    
    # Copy metadata
    for element in score.flatten():
        if isinstance(element, (music21.tempo.MetronomeMark, music21.key.KeySignature, 
                              music21.meter.TimeSignature)):
            new_score.insert(0, element)
    
    return new_score if len(new_score.parts) > 0 else score

def simplify_midi(midi_path, output_path="midi_simplified"):
    """
    Comprehensively simplify a MIDI file for easier sheet music reading.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print("Loading MIDI file...")
    score = music21.converter.parse(midi_path)
    
    print("Detecting key signature...")
    key = detect_key_signature(score)
    score.insert(0, key)
    
    print("Filtering low-confidence notes...")
    score = filter_notes_by_confidence(score, velocity_threshold=30)
    
    print("Removing ornaments and grace notes...")
    score = remove_ornaments_and_grace_notes(score)
    
    print("Quantizing to common rhythms...")
    score = quantize_to_common_rhythms(score)
    
    print("Merging tied notes...")
    score = merge_tied_notes(score)
    
    print("Simplifying chords...")
    score = simplify_chords(score, max_chord_notes=3)
    
    print("Separating hands...")
    score = separate_hands(score)
    
    # Final quantization
    print("Final quantization...")
    score = score.quantize(quarterLengthDivisors=[4, 3])  # Allow triplets
    
    # Add time signature if missing
    if not score.getElementsByClass(music21.meter.TimeSignature):
        score.insert(0, music21.meter.TimeSignature('4/4'))
    
    simplified_midi_path = os.path.join(output_path, os.path.splitext(os.path.basename(midi_path))[0] + "_simplified.mid")
    score.write('midi', fp=simplified_midi_path)
    return simplified_midi_path

def create_sheet_music(midi_path, output_path="partitures"):
    """
    Creates sheet music from a MIDI file with better formatting.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    score = music21.converter.parse(midi_path)
    
    # Add title if missing
    if not score.metadata:
        score.metadata = music21.metadata.Metadata()
        score.metadata.title = 'Transcribed Piano Music'
    
    # Ensure proper clefs are set
    for i, part in enumerate(score.parts):
        if i == 0:  # First part (right hand)
            part.insert(0, music21.clef.TrebleClef())
        elif i == 1:  # Second part (left hand)
            part.insert(0, music21.clef.BassClef())
    
    output_sheet_path = os.path.join(output_path, os.path.splitext(os.path.basename(midi_path))[0] + ".xml")
    score.write('musicxml', fp=output_sheet_path)
    return output_sheet_path

def main():
    """
    Main function to process the YouTube video with improved transcription.
    """
    parser = argparse.ArgumentParser(description='Transcribe piano audio from YouTube to simplified MIDI and sheet music.')
    parser.add_argument('url', type=str, help='The YouTube URL to process.')
    parser.add_argument('--name', type=str, help='A custom name for the output files.')
    parser.add_argument('--start', type=int, help='Start time in seconds.')
    parser.add_argument('--end', type=int, help='End time in seconds.')
    parser.add_argument('--model', type=str, default='mt3', choices=['basic-pitch', 'mt3'],
                       help='The transcription model to use.')
    parser.add_argument('--no-simplification', action='store_true', help='Skip the MIDI simplification process.')
    parser.add_argument('--keep-all-instruments', action='store_true', help='Keep all instrument tracks from the transcription.')
    parser.add_argument('--velocity-threshold', type=int, default=30,
                       help='Minimum velocity for notes to be included (default: 30).')
    parser.add_argument('--max-chord-notes', type=int, default=3,
                       help='Maximum number of notes in a chord (default: 3).')

    args = parser.parse_args()

    print("Downloading audio...")
    audio_file = download_audio(args.url, name=args.name, start_time=args.start, end_time=args.end)

    print(f"Transcribing audio to MIDI using {args.model}...")
    if args.model == 'mt3':
        midi_file = transcribe_audio_mt3_api(audio_file)
    else:
        midi_file = transcribe_audio(audio_file)

    if not args.keep_all_instruments:
        print("Filtering for piano tracks...")
        filter_piano_tracks(midi_file)

    simplified_midi_file = None
    if not args.no_simplification:
        print("Applying advanced MIDI simplification...")
        simplified_midi_file = simplify_midi(midi_file)
        sheet_music_source = simplified_midi_file
    else:
        print("Skipping MIDI simplification.")
        sheet_music_source = midi_file

    print("Creating sheet music...")
    sheet_music_file = create_sheet_music(sheet_music_source)

    print("\nSuccessfully created:")
    print(f"  Original MIDI: {midi_file}")
    if simplified_midi_file:
        print(f"  Simplified MIDI: {simplified_midi_file}")
    print(f"  Sheet music: {sheet_music_file}")

    print("\nTips for better results:")
    print("  - Use shorter audio clips (30-60 seconds)")
    print("  - Ensure audio has minimal background noise")
    print("  - Try adjusting --velocity-threshold (lower = more notes)")
    print("  - Try adjusting --max-chord-notes for simpler chords")
    print("  - Use the --no-simplification flag if the transcription is already clean.")
    print("  - Use the --keep-all-instruments flag to keep all transcribed instrument tracks.")

if __name__ == "__main__":
    main()