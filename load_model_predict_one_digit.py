from PIL import Image, ImageOps
import cv2, numpy as np
import tensorflow as tf

try:
    resample_filter = Image.Resampling.LANCZOS
except AttributeError:
    resample_filter = Image.LANCZOS
    
    
# 1. Load saved model
model = tf.keras.models.load_model('mnist_digit_classifier.h5')

# 2. Load & preprocess custom digit image
def preprocess_digit(path):
    # 1) Load as grayscale array
    pil   = Image.open(path).convert('L')
    raw   = np.array(pil)                # uint8 [0..255]
    
    # 2) Auto-invert if the background is brighter than the digit
    if raw.mean() > 127:
        raw = 255 - raw

    # 3) Binarize (foreground will be 255)
    _, bw = cv2.threshold(raw, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 4) Find and crop to the digit’s bounding-box
    ys, xs   = np.where(bw == 255)
    x0, x1   = xs.min(), xs.max()
    y0, y1   = ys.min(), ys.max()
    digit    = bw[y0:y1+1, x0:x1+1]

    # 5) Resize into a 20×20 box, preserving aspect ratio
    h, w     = digit.shape
    if h > w:
        new_h  = 20
        new_w  = max(1, int(w * (20.0 / h)))
    else:
        new_w  = 20
        new_h  = max(1, int(h * (20.0 / w)))
    digit_img = Image.fromarray(digit).resize(
        (new_w, new_h),
        resample=resample_filter
    )
    digit = np.array(digit_img)

    # 6) Pad into a 28×28 canvas, symmetrically
    canvas = np.zeros((28, 28), dtype='uint8')
    top    = (28 - new_h) // 2
    left   = (28 - new_w) // 2
    canvas[top:top+new_h, left:left+new_w] = digit

    # 7) Recentering via center-of-mass shift
    M = cv2.moments(canvas)
    if M['m00'] != 0:
        cx      = M['m10'] / M['m00']
        cy      = M['m01'] / M['m00']
        shiftx  = int(round(14.0 - cx))
        shifty  = int(round(14.0 - cy))
        canvas  = cv2.warpAffine(
            canvas,
            np.float32([[1, 0, shiftx], [0, 1, shifty]]),
            (28, 28),
            borderValue=0
        )

    # 8) Normalize 
    norm = canvas.astype('float32') / 255.0
    
    #display preprocessed image for debugging
    Image.fromarray((norm.squeeze() * 255).astype('uint8')).show()
    
    return norm.reshape(1, 28, 28, 1)


# 3. Predict
image_path = 'my_digit6_rotated_lightb.png'
input_array = preprocess_digit(image_path)
pred_probs = model.predict(input_array)        # shape (1, 10)
pred_label = np.argmax(pred_probs, axis=1)[0]  # highest-probability digit

print(f'I think this digit is: {pred_label}')
