'''
Usage: 
    h5_to_onnx.py --model="pilot_23-02-15_29.h5" --saved_model="mymodel.saved_model"  --out="mymodel.onnx"
    h5_to_onnx.py --model="pilot_23-02-15_29.h5" --saved_model="pilot_23-02-15_29_test.saved_model"  --out="pilot_23-02-15_29_test.trt"

Note:
    pass
'''

import os

from docopt import docopt
# from tensorflow import keras
from donkeycar.parts.interpreter import keras_model_to_onnx

args = docopt(__doc__)

in_model = os.path.expanduser(args['--model'])
saved_model = os.path.expanduser(args['--saved_model'])
out_model = os.path.expanduser(args['--out']) 

# keras_model = keras.models.load_model(in_model)
# keras_model.save(saved_model)

# saved_model_to_tensor_rt(saved_model, out_model)


# keras_model_to_onnx(saved_model, f'{out_model}.onnx')
keras_model_to_onnx(saved_model, out_model)