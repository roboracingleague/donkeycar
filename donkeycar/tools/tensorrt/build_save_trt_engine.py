
"""
Generate tensorrt model from onnx
Usage:
  build_save_trt_engine.py (--onnx <onnx_model_path>) (--savedtrt <trt_engine_path>) [--limit-workspace]

  # build_save_trt_engine.py --onnx new_model.onnx --savedtrt new_model.trt
Options:
  -h --help                Show this screen.
     --limit-workspace     Specify if the workspace size must be limited
"""

from docopt import docopt
from donkeycar.tools.tensorrt import engine as eng
from onnx import ModelProto

def build(onnx_path,engine_name, limit_workspace):
    # engine_name = "~/mycar/models/new_model.plan"
    # onnx_path = "~/mycar/models/new_model.onnx"
    batch_size = 1

    model = ModelProto()
    with open(onnx_path, "rb") as f:
        model.ParseFromString(f.read())

    d0 = model.graph.input[0].type.tensor_type.shape.dim[1].dim_value
    d1 = model.graph.input[0].type.tensor_type.shape.dim[2].dim_value
    d2 = model.graph.input[0].type.tensor_type.shape.dim[3].dim_value
    shape = [batch_size , d0, d1 ,d2]
    engine = eng.build_engine(onnx_path, limit_workspace, shape= shape)
    eng.save_engine(engine, engine_name)

if __name__=='__main__':
    args = docopt(__doc__)
    onnx_model_path = args['<onnx_model_path>']
    trt_engine_path = args['<trt_engine_path>']
    limit_workspace = args['--limit-workspace']

    build(onnx_model_path,trt_engine_path, limit_workspace)
