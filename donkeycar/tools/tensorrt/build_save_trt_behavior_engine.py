"""build_save_trt_behavior_engine

Usage:
  build_save_trt_behavior_engine.py (--onnx <onnx_model_path>) (--savedtrt <trt_engine_path>) [--orin]

Options:
  -h --help     Show this screen.
     --orin     Specify that this script runs on an orin platform
"""

import tensorrt as trt
from docopt import docopt

# TensorRT logger setup
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_file_path, engine_file_path, is_orin):
    # Create a builder
    builder = trt.Builder(TRT_LOGGER)

    # Set up the network and the parser
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse the ONNX model
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    # Define optimization profile for dynamic input shapes
    profile = builder.create_optimization_profile()
    profile.set_shape('img_in', (1, 165, 320, 3), (1, 165, 320, 3), (1, 165, 320, 3))
    profile.set_shape('xbehavior_in', (1, 3), (1, 3), (1, 3))
    config = builder.create_builder_config()
    config.add_optimization_profile(profile)

    # Build the engine
    builder.max_batch_size = 1
    if not is_orin:
        builder.max_workspace_size = 1 << 30  # 1GB
    engine = builder.build_engine(network, config)

    # Save the engine to a file
    with open(engine_file_path, 'wb') as f:
        f.write(engine.serialize())

def main():
    arguments = docopt(__doc__)
    onnx_model_path = arguments['<onnx_model_path>']
    trt_engine_path = arguments['<trt_engine_path>']
    is_orin = arguments['--orin']

    build_engine(onnx_model_path, trt_engine_path, is_orin)

if __name__ == "__main__":
    main()
