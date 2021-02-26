
# /usr/src/tensorrt/bin/trtexec --onnx=wav2vec2.onnx --saveEngine=wav2vec2_fp16.engine --maxBatch=16 --workspace=4096 --fp16


#trtexec --onnx=wav2vec2.onnx --saveEngine=wav2vec2_fp16.engine --explicitBatch --workspace=4096 --fp16
trtexec --onnx=wav2vec2.onnx --saveEngine=wav2vec2_fp16.engine --minShapes=input:1x500 --optShapes=input:1x30720 --maxShapes=input:1x3000000 --workspace=4096 --fp16

