import atexit
import os
import time
from datetime import datetime
import json

import numpy as np
from PIL import Image

from donkeycar.parts.datastore_v2 import Manifest, ManifestIterator
# import zmq
# from io import BytesIO
# import cv2

class Tub(object):
    """
    A datastore to store sensor data in a key, value format. \n
    Accepts str, int, float, image_array, image, and array data types.
    """

    def __init__(self, base_path, inputs=[], types=[], metadata=[],
                 max_catalog_len=1000, read_only=False):
        self.base_path = base_path
        self.images_base_path = os.path.join(self.base_path, Tub.images())
        self.depths_base_path = self.images_base_path.replace("images","depths")
        self.inputs = inputs
        self.types = types
        self.metadata = metadata
        self.manifest = Manifest(base_path, inputs=inputs, types=types,
                                 metadata=metadata, max_len=max_catalog_len,
                                 read_only=read_only)
        self.input_types = dict(zip(self.inputs, self.types))
        # Create images folder if necessary
        if not os.path.exists(self.images_base_path):
            os.makedirs(self.images_base_path, exist_ok=True)
        if not os.path.exists(self.depths_base_path):
            os.makedirs(self.depths_base_path, exist_ok=True)
        
        # set up the publisher
        # self.context = zmq.Context()
        # self.socket = self.context.socket(zmq.PUB)
        # self.socket.bind("tcp://*:5555")

    def write_record(self, record=None):
        """
        Can handle various data types including images.
        """
        contents = dict()
        for key, value in record.items():
            if value is None:
                continue
            elif key not in self.input_types:
                continue
            else:
                input_type = self.input_types[key]
                if input_type == 'float':
                    # Handle np.float() types gracefully
                    contents[key] = float(value)
                elif input_type == 'str':
                    contents[key] = value
                elif input_type == 'int':
                    contents[key] = int(value)
                elif input_type == 'boolean':
                    contents[key] = bool(value)
                elif input_type == 'nparray':
                    contents[key] = value.tolist()
                elif input_type == 'list' or input_type == 'vector':
                    contents[key] = list(value)
                elif input_type == 'image_array':
                    # Handle image array
                    # original version
                    name = Tub._image_file_name(self.manifest.current_index, key)
                    image_path = os.path.join(self.images_base_path, name)
                    image = Image.fromarray(np.uint8(value))
                    image.save(image_path)
                    image.close()
                    del image

                    # cv2 version
                    #cv2.imwrite(image_path,np.uint8(value))
                    
                    # zmq version
                    # key = image_path
                    # message = key.encode() + b" " + np.uint8(value).tobytes()

                    # This is sync io
                    # self.socket.send(message)

                    # Write binary
                    # with open(image_path, "wb") as binary_file:
                        # binary_file.write(value.tobytes())
                        # pass

                    # Bytesio
                    # write_byte = BytesIO(value.tobytes())
 
                    # with open(image_path, "wb") as f:
                        # f.write(write_byte.getbuffer())
                    
                    # OpenCV
                    # cv2.imwrite(image_path, value)

                    # common part
                    contents[key] = name

                elif input_type == 'gray16_array':
                    # Handle image array
                    #name = Tub._image_file_name(self.manifest.current_index, key)
                    #image_path = os.path.join(self.depths_base_path, name)
                    #np.savez_compressed(image_path, img=np.uint16(value))
                    #contents[key] = name

                    # save np.uint16 as a 16bit png
                    image = Image.fromarray(np.uint16(value))
                    name = Tub._image_file_name(self.manifest.current_index, key, extension='.png')
                    image.save(os.path.join(self.depths_base_path, name))
                    del image
                    contents[key]=name

        # Private properties
        contents['_timestamp_ms'] = int(round(time.time() * 1000))
        contents['_index'] = self.manifest.current_index
        contents['_session_id'] = self.manifest.session_id

        self.manifest.write_record(contents)

    def delete_records(self, record_indexes):
        self.manifest.delete_records(record_indexes)

    def label_records(self, record_indexes, label):
        self.manifest.label_records(record_indexes, label)

    def delete_last_n_records(self, n):
        # build ordered list of non-deleted indexes
        all_alive_indexes = sorted(set(range(self.manifest.current_index))
                                   - self.manifest.deleted_indexes)
        to_delete_indexes = all_alive_indexes[-n:]
        self.manifest.delete_records(to_delete_indexes)

    def restore_records(self, record_indexes):
        self.manifest.restore_records(record_indexes)

    def close(self):
        self.manifest.close()

    def __iter__(self):
        return ManifestIterator(self.manifest)

    def __len__(self):
        return self.manifest.__len__()

    @classmethod
    def images(cls):
        return 'images'

    @classmethod
    def _image_file_name(cls, index, key, extension='.jpg'):
        key_prefix = key.replace('/', '_')
        name = '_'.join([str(index), key_prefix, extension])
        # Return relative paths to maintain portability
        return name


class TubWriter(object):
    """
    A Donkey part, which can write records to the datastore.
    """
    def __init__(self, base_path, inputs=[], types=[], metadata=[],
                 max_catalog_len=1000):
        self.tub = Tub(base_path, inputs, types, metadata, max_catalog_len)

    def run(self, *args):
        assert len(self.tub.inputs) == len(args), \
            f'Expected {len(self.tub.inputs)} inputs but received {len(args)}'
        record = dict(zip(self.tub.inputs, args))
        self.tub.write_record(record)
        return self.tub.manifest.current_index

    def __iter__(self):
        return self.tub.__iter__()

    def close(self):
        self.tub.close()

    def shutdown(self):
        self.close()


class TubWiper:
    """
    Donkey part which deletes a bunch of records from the end of tub.
    This allows to remove bad data already during recording. As this gets called
    in the vehicle loop the deletion runs only once in each continuous
    activation. A new execution requires to release of the input trigger. The
    action could result in a multiple number of executions otherwise.
    """
    def __init__(self, tub, num_records=20):
        """
        :param tub: tub to operate on
        :param num_records: number or records to delete
        """
        self._tub = tub
        self._num_records = num_records
        self._active_loop = False

    def run(self, is_delete):
        """
        Method in the vehicle loop. Delete records when trigger switches from
        False to True only.
        :param is_delete: if deletion has been triggered by the caller
        """
        # only run if input is true and debounced
        if is_delete:
            if not self._active_loop:
                # action command
                self._tub.delete_last_n_records(self._num_records)
                # increase the loop counter
                self._active_loop = True
        else:
            # trigger released, reset active loop
            self._active_loop = False