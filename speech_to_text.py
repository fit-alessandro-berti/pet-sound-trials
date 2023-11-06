from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
import torch
from datasets import load_dataset
import soundfile

# Load the model and processor
model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")
processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")

# Load an example dataset and get the first example
#dataset = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
#example = dataset[0]

audio_input, sampling_rate = soundfile.read("output.wav")

# Process the raw input
#inputs = processor(example["audio"]["array"], sampling_rate=example["audio"]["sampling_rate"], return_tensors="pt")
inputs = processor(audio_input, sampling_rate=sampling_rate, return_tensors="pt")
input_features = inputs.input_features  # get the input features

# Perform the speech-to-text task
with torch.no_grad():
    generated_ids = model.generate(input_features)

# Convert the generated ids to text
transcription = processor.batch_decode(generated_ids)

print(transcription)
