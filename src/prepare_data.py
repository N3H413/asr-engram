import argparse
import os
import json
import soundfile as sf
import librosa
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="google/fleurs")
    parser.add_argument("--subset", type=str, default="hi_in")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max_hours", type=float, default=5.0)
    parser.add_argument("--output_dir", type=str, default="./data")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    audio_dir = os.path.join(args.output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    print(f"Loading dataset {args.dataset_name} ({args.subset})...")
    # Streaming mode avoids memory bottlenecks on Colab when loading large datasets
    dataset = load_dataset(args.dataset_name, args.subset, split=args.split, streaming=True)

    manifest_path = os.path.join(args.output_dir, f"{args.split}_manifest.json")
    
    total_duration = 0.0
    max_duration_sec = args.max_hours * 3600
    written_samples = 0

    print(f"Processing and writing up to {args.max_hours} hours of audio...")
    with open(manifest_path, "w", encoding="utf-8") as f_out:
        for idx, item in enumerate(dataset):
            audio_data = item.get("audio")
            if not audio_data:
                continue
                
            audio_array = audio_data["array"]
            sr = audio_data["sampling_rate"]
            
            # Ensure 16kHz sample rate for NeMo compatibility
            if sr != 16000:
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
                sr = 16000

            # Correctly map Google FLEURS transcription keys
            text = item.get("transcription", item.get("raw_transcription", ""))

            if not text.strip():
                continue

            duration = len(audio_array) / sr
            
            # Save individual WAV file
            file_name = f"sample_{idx:06d}.wav"
            file_path = os.path.join(audio_dir, file_name)
            sf.write(file_path, audio_array, sr)

            # NeMo Manifest Entry Format
            entry = {
                "audio_filepath": os.path.abspath(file_path),
                "duration": duration,
                "text": text.strip()
            }
            f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")

            total_duration += duration
            written_samples += 1

            if total_duration >= max_duration_sec:
                break

    print(f"Done! Created manifest at {manifest_path} with {written_samples} samples (~{total_duration/3600:.2f} hours).")

if __name__ == "__main__":
    main()