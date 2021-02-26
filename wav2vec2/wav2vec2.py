import os
import librosa
import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import nltk

from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def load_model():
    tokenizer = Wav2Vec2Tokenizer.from_pretrained(
        "facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    return tokenizer, model


def correct_sentence(input_text):
    sentences = nltk.sent_tokenize(input_text)
    return (' '.join([s.replace(s[0], s[0].capitalize(), 1) for s in sentences]))


def asr_transcript(model, tokenizer, input_file):

    if not os.path.isfile(input_file):
        raise FileNotFoundError

    # tokenizer, model = load_model()

    speech, fs = sf.read(input_file)

    if len(speech.shape) > 1:
        speech = speech[:, 0] + speech[:, 1]

    if fs != 16000:
        speech = librosa.resample(speech, fs, 16000)

    input_values = tokenizer(speech, return_tensors="pt").input_values
    input_values = input_values.to(device)
    logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)

    transcription = tokenizer.decode(predicted_ids[0])

    return correct_sentence(transcription.lower())


def to_onnx(torch_model, tokenizer, x):

    if not os.path.isfile(x):
        raise FileNotFoundError

    # tokenizer, model = load_model()

    speech, fs = sf.read(x)

    if len(speech.shape) > 1:
        speech = speech[:, 0] + speech[:, 1]

    if fs != 16000:
        speech = librosa.resample(speech, fs, 16000)

    input_values = tokenizer(speech, return_tensors="pt").input_values
    # input_values = input_values.to(device)

    # Export the model
    # torch.onnx.export(torch_model,  # model being run
    #                   # model input (or a tuple for multiple inputs)
    #                   input_values,
    #                   # where to save the model (can be a file or file-like object)
    #                   "wav2vec2.onnx",
    #                   export_params=True,        # store the trained parameter weights inside the model file
    #                   opset_version=11,          # the ONNX version to export the model to
    #                   do_constant_folding=True,  # whether to execute constant folding for optimization
    #                   input_names=['input'],   # the model's input names
    #                   output_names=['output'],  # the model's output names
    #                   dynamic_axes={'input': {0: 'batch_size', 1: 'nfeat'},
    #                                 'output': {0: 'batch_size'}}     # variable lenght axes
    #                   )
    torch.onnx.export(torch_model,  # model being run
                      # model input (or a tuple for multiple inputs)
                      input_values,
                      # where to save the model (can be a file or file-like object)
                      "wav2vec2.onnx",
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=11,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],   # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {1: 'nfeat'}}     # variable lenght axes
                      )


def main():
    # prereq
    nltk.download('punkt')

    # f = 'data/LibriSpeech/test-clean/61/70968/61-70968-0000.flac'
    data_root = "data/LibriSpeech/test-clean"
    all_chapters = [os.path.join(data_root, f) for f in os.listdir(data_root)]
    sub_chapters = [os.listdir(d) for i, d in enumerate(all_chapters)]

    full_chapters = []
    for i, c in enumerate(sub_chapters):
        for k in c:
            full_chapters.append(os.path.join(all_chapters[i], k))

    # full_chapters = [os.path.join(all_chapters[i], k) for k in c for i,c in enumerate(sub_chapters)]
    all_rows = [os.path.join(d, f)
                for d in full_chapters for f in os.listdir(d)]

    extensions = ['.flac', '.mp3']
    flacs = [f for f in all_rows if os.path.splitext(f)[1] in extensions]

    tokenizer, model = load_model()
    # model.to(device)
    # for f in tqdm(flacs):
    #     print(asr_transcript(model, tokenizer, f))
    x = flacs[0]
    to_onnx(model, tokenizer, x)


if __name__ == "__main__":
    main()
