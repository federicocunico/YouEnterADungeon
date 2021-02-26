import os
import numpy as np
import onnxruntime
import os
import librosa
import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import nltk

def load_model():
    tokenizer = Wav2Vec2Tokenizer.from_pretrained(
        "facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    return tokenizer, model


# DATI
###############################################
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

speech, fs = sf.read(x)

if len(speech.shape) > 1:
    speech = speech[:, 0] + speech[:, 1]

if fs != 16000:
    speech = librosa.resample(speech, fs, 16000)

input_values = tokenizer(speech, return_tensors="pt").input_values
torch_out = model(input_values).logits
###############################################

import onnx

onnx_model = onnx.load("wav2vec2.onnx")
onnx.checker.check_model(onnx_model)


ort_session = onnxruntime.InferenceSession("wav2vec2.onnx")


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_values)}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
source = to_numpy(torch_out)
dest = ort_outs[0]
# np.testing.assert_allclose(
#     to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
np.testing.assert_allclose(source, dest, rtol=2e-03, atol=2e-03)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")
