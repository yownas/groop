import onnxruntime
import insightface
import imageio.v3 as iio
from PIL import Image
import argparse
import cv2

FACE_ANALYSER = None
FACE_SWAPPER = None
ONNX_PROVIDERS = None

def get_face_swapper():
    global FACE_SWAPPER
    if FACE_SWAPPER is None:
        model_path = 'inswapper_128.onnx'
        FACE_SWAPPER = insightface.model_zoo.get_model(model_path, download=False, download_zip=False, providers=ONNX_PROVIDERS)
    return FACE_SWAPPER

def get_face_analyser():
    global FACE_ANALYSER
    if FACE_ANALYSER is None:
        FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers=ONNX_PROVIDERS)
        FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))
    return FACE_ANALYSER

if __name__ == "__main__":
    out_imgs = []
    ONNX_PROVIDERS = onnxruntime.get_available_providers()
    print(f"Providers: {ONNX_PROVIDERS}")

    parser = argparse.ArgumentParser(description="groop")
    parser.add_argument("-t", "--target", type = str, required = True, help="target face")
    parser.add_argument("-i", "--input", type = str, required = True, help="input gif")
    parser.add_argument("-o", "--output", type = str, required = True, help="output gif")
    params = parser.parse_args()

    # Read target
    tgt_img = cv2.imread(params.target)
    tgt_face = sorted(get_face_analyser().get(tgt_img), key=lambda x: x.bbox[0])[0]

    # Read input
    x = iio.immeta(params.input)
    duration = x['duration']
    loop = x['loop']

    gif = cv2.VideoCapture(params.input) 

    # Swap
    while(True):
        ret, frame = gif.read()
        if not ret:
            break
        out_face = sorted(get_face_analyser().get(frame), key=lambda x: x.bbox[0])[0]
        frame = get_face_swapper().get(frame, out_face, tgt_face, paste_back=True)
        out_imgs.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

    # Write output
    out_imgs[0].save(params.output, save_all=True, append_images=out_imgs[1:], optimize=True, duration=duration, loop=loop)

