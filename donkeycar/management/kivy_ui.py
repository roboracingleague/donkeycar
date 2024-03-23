import json
import re
import time
from copy import copy, deepcopy
from datetime import datetime
from functools import partial
from subprocess import Popen, PIPE, STDOUT
from threading import Thread
from collections import namedtuple
from kivy.logger import Logger
import platform
import traceback 

import io
import os
import sys
import shutil
import glob
import atexit
import yaml
from PIL import Image as PilImage
import pandas as pd
import numpy as np
from scipy.ndimage import label, find_objects

import plotly.express as px
from kivy.clock import Clock
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.core.image import Image as CoreImage
from kivy.properties import NumericProperty, ObjectProperty, StringProperty, \
    ListProperty, BooleanProperty
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.lang.builder import Builder
from kivy.core.window import Window
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.scrollview import ScrollView
from kivy.uix.spinner import SpinnerOption, Spinner
from kivy.graphics import Color, Rectangle, Point, GraphicException,Line, Ellipse

from donkeycar import load_config
from donkeycar.parts.tub_v2 import Tub
from donkeycar.pipeline.augmentations import ImageAugmentation
from donkeycar.pipeline.database import PilotDatabase
from donkeycar.pipeline.types import TubRecord
from donkeycar.utils import get_model_by_type
from donkeycar.pipeline.training import train
from donkeycar.pipeline.sam import Segmentation

Logger.propagate = False

Builder.load_file(os.path.join(os.path.dirname(__file__), 'ui.kv'))
Window.clearcolor = (0.2, 0.2, 0.2, 1)
LABEL_SPINNER_TEXT = 'Add/remove'

# Data struct to show tub field in the progress bar, containing the name,
# the name of the maximum value in the config file and if it is centered.
FieldProperty = namedtuple('FieldProperty',
                           ['field', 'max_value_id', 'centered'])


def get_norm_value(value, cfg, field_property, normalised=True):
    max_val_key = field_property.max_value_id
    max_value = getattr(cfg, max_val_key, 1.0)
    out_val = value / max_value if normalised else value * max_value
    return out_val


def map_range(x, X_min, X_max, Y_min, Y_max, enforce_input_in_range=False):
    '''
    Linear mapping between two ranges of values
    '''
    if enforce_input_in_range:
        x=max(min(x,X_max),X_min)

    if (X_min==X_max):
        return Y_min
    if (Y_min==Y_max):
        return Y_min

    X_range = X_max - X_min
    Y_range = Y_max - Y_min
    XY_ratio = X_range/Y_range


    y = ((x-X_min) / XY_ratio + Y_min) // 1

    return int(y)

def np_label(mask):
    labeled_mask = np.zeros_like(mask, dtype=int)
    label_count = 0

    def dfs(i, j):
        nonlocal label_count
        if 0 <= i < mask.shape[0] and 0 <= j < mask.shape[1] and mask[i, j] == 1 and labeled_mask[i, j] == 0:
            label_count += 1
            labeled_mask[i, j] = label_count

            # Explore neighbors
            dfs(i + 1, j)
            dfs(i - 1, j)
            dfs(i, j + 1)
            dfs(i, j - 1)

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] == 1 and labeled_mask[i, j] == 0:
                dfs(i, j)

    return labeled_mask, label_count

def find_widest_area(mask):
    labeled_mask, num_features = np_label(mask)

    unique_labels, label_counts = np.unique(labeled_mask, return_counts=True)
    unique_labels = unique_labels[1:]
    label_counts = label_counts[1:]

    if len(unique_labels) == 0:
        return None

    widest_area_label = unique_labels[np.argmax(label_counts)]
    widest_area_mask = (labeled_mask == widest_area_label).astype(int)

    return widest_area_mask

def has_multiple_areas(mask):
    labeled_mask, num_features = label(mask)
    
    return num_features > 1

def extract_widest_area(mask):
    # Label connected components in the binary mask
    labeled_mask, num_features = label(mask)
    counts = np.bincount(labeled_mask.flatten())
    most_frequent_value = np.argmax(counts[1:])
    widest_mask = (labeled_mask == (most_frequent_value+1)).astype(int)
    return widest_mask

# def extract_widest_area(mask):
#     # Label connected components in the binary mask
#     labeled_mask, num_features = label(mask)

#     # Find bounding boxes for each connected component
#     bounding_boxes = find_objects(labeled_mask)

#     # Find the bounding box of the widest area
#     max_width = 0
#     widest_area_bbox = None

#     for bbox in bounding_boxes:
#         width = bbox[1].stop - bbox[1].start
#         if width > max_width:
#             max_width = width
#             widest_area_bbox = bbox

#     if widest_area_bbox is not None:
#         # Extract the widest area from the original mask
#         widest_area = mask[widest_area_bbox[0], widest_area_bbox[1]]
#         padded_widest_area = np.zeros_like(mask)
#         padded_widest_area[widest_area_bbox[0], widest_area_bbox[1]] = widest_area

#         return padded_widest_area.astype(int)  # Convert to binary mask (0 and 1)
#     else:
#         return None
    
def tub_screen():
    return App.get_running_app().tub_screen if App.get_running_app() else None

def annotate_screen():
    return App.get_running_app().annotate_screen if App.get_running_app() else None


def pilot_screen():
    return App.get_running_app().pilot_screen if App.get_running_app() else None


def train_screen():
    return App.get_running_app().train_screen if App.get_running_app() else None


def car_screen():
    return App.get_running_app().car_screen if App.get_running_app() else None


def recursive_update(target, source):
    """ Recursively update dictionary """
    if isinstance(target, dict) and isinstance(source, dict):
        for k, v in source.items():
            v_t = target.get(k)
            if not recursive_update(v_t, v):
                target[k] = v
        return True
    else:
        return False


def decompose(field):
    """ Function to decompose a string vector field like 'gyroscope_1' into a
        tuple ('gyroscope', 1) """
    field_split = field.split('_')
    if len(field_split) > 1 and field_split[-1].isdigit():
        return '_'.join(field_split[:-1]), int(field_split[-1])
    return field, None


class RcFileHandler:
    """ This handles the config file which stores the data, like the field
        mapping for displaying of bars and last opened car, tub directory. """

    # These entries are expected in every tub, so they don't need to be in
    # the file
    known_entries = [
        FieldProperty('user/angle', '', centered=True),
        FieldProperty('user/throttle', '', centered=False),
        FieldProperty('pilot/angle', '', centered=True),
        FieldProperty('pilot/throttle', '', centered=False),
    ]

    def __init__(self, file_path='~/.donkeyrc'):
        self.file_path = os.path.expanduser(file_path)
        self.data = self.create_data()
        recursive_update(self.data, self.read_file())
        self.field_properties = self.create_field_properties()

        def exit_hook():
            self.write_file()
        # Automatically save config when program ends
        atexit.register(exit_hook)

    def create_field_properties(self):
        """ Merges known field properties with the ones from the file """
        field_properties = {entry.field: entry for entry in self.known_entries}
        field_list = self.data.get('field_mapping')
        if field_list is None:
            field_list = {}
        for entry in field_list:
            assert isinstance(entry, dict), \
                'Dictionary required in each entry in the field_mapping list'
            field_property = FieldProperty(**entry)
            field_properties[field_property.field] = field_property
        return field_properties

    def create_data(self):
        data = dict()
        data['user_pilot_map'] = {'user/throttle': 'pilot/throttle',
                                  'user/angle': 'pilot/angle'}
        return data

    def read_file(self):
        if os.path.exists(self.file_path):
            with open(self.file_path) as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
                Logger.info(f'Donkeyrc: Donkey file {self.file_path} loaded.')
                return data
        else:
            Logger.warn(f'Donkeyrc: Donkey file {self.file_path} does not '
                        f'exist.')
            return {}

    def write_file(self):
        if os.path.exists(self.file_path):
            Logger.info(f'Donkeyrc: Donkey file {self.file_path} updated.')
        with open(self.file_path, mode='w') as f:
            self.data['time_stamp'] = datetime.now()
            data = yaml.dump(self.data, f)
            return data


rc_handler = RcFileHandler()


class MySpinnerOption(SpinnerOption):
    """ Customization for Spinner """
    pass


class MySpinner(Spinner):
    """ Customization of Spinner drop down menu """
    def __init__(self, **kwargs):
        super().__init__(option_cls=MySpinnerOption, **kwargs)


class FileChooserPopup(Popup):
    """ File Chooser popup window"""
    load = ObjectProperty()
    root_path = StringProperty()
    filters = ListProperty()


class FileChooserBase:
    """ Base class for file chooser widgets"""
    file_path = StringProperty("No file chosen")
    popup = ObjectProperty(None)
    root_path = os.path.expanduser('~')
    title = StringProperty(None)
    filters = ListProperty()

    def open_popup(self):
        self.popup = FileChooserPopup(load=self.load, root_path=self.root_path,
                                      title=self.title, filters=self.filters)
        self.popup.open()

    def load(self, selection):
        """ Method to load the chosen file into the path and call an action"""
        self.file_path = str(selection[0])
        self.popup.dismiss()
        self.load_action()

    def load_action(self):
        """ Virtual method to run when file_path has been updated """
        pass


class ConfigManager(BoxLayout, FileChooserBase):
    """ Class to mange loading of the config file from the car directory"""
    config = ObjectProperty(None)
    file_path = StringProperty(rc_handler.data.get('car_dir', ''))

    def load_action(self):
        """ Load the config from the file path"""
        if self.file_path:
            try:
                path = os.path.join(self.file_path, 'config.py')
                self.config = load_config(path)
                # If load successful, store into app config
                rc_handler.data['car_dir'] = self.file_path
            except FileNotFoundError:
                Logger.error(f'Config: Directory {self.file_path} has no '
                             f'config.py')
            except Exception as e:
                Logger.error(f'Config: {e}')


class TubLoader(BoxLayout, FileChooserBase):
    """ Class to manage loading or reloading of the Tub from the tub directory.
        Loading triggers many actions on other widgets of the app. """
    file_path = StringProperty(rc_handler.data.get('last_tub', ''))
    tub = ObjectProperty(None)
    len = NumericProperty(1)
    len_with_deleted = NumericProperty(1)
    records = None

    def load_action(self):
        """ Update tub from the file path"""
        if self.update_tub():
            # If update successful, store into app config
            rc_handler.data['last_tub'] = self.file_path
            annotate_screen().load_action()

    def update_tub(self, event=None, syncAnnotate=True):
        if not self.file_path:
            return False
        # If config not yet loaded return
        cfg = tub_screen().ids.config_manager.config
        if not cfg:
            return False
        # At least check if there is a manifest file in the tub path
        if not os.path.exists(os.path.join(self.file_path, 'manifest.json')):
            tub_screen().status(f'Path {self.file_path} is not a valid tub.')
            if (os.path.isdir(f'{self.file_path}/raw_catalogs') and os.path.isfile(f'{self.file_path}/raw_catalogs/manifest.json')):
                tub_screen().status(f"init Tub : found {self.file_path}/raw_catalogs, trying to import ...")
                for afile in glob.glob(f'{self.file_path}/raw_catalogs/catalog_*.catalog'):
                    shutil.copy(f'{self.file_path}/raw_catalogs/{os.path.basename(afile)}', f'{self.file_path}/{os.path.basename(afile)}')
                shutil.copyfile(f'{self.file_path}/raw_catalogs/manifest.json', f'{self.file_path}/manifest.json')
            else:
                return False
        try:
            if self.tub:
                self.tub.close()
            self.tub = Tub(self.file_path)
        except Exception as e:
            tub_screen().status(f'Failed loading tub: {str(e)}')
            return False
        # Check if filter is set in tub screen
        # expression = tub_screen().ids.tub_filter.filter_expression
        train_filter = getattr(cfg, 'TRAIN_FILTER', None)

        # Use filter, this defines the function
        def select(underlying):
            if not train_filter:
                return True
            else:
                try:
                    record = TubRecord(cfg, self.tub.base_path, underlying)
                    res = train_filter(record)
                    return res
                except KeyError as err:
                    Logger.error(f'Filter: {err}')
                    return True

        self.records = [TubRecord(cfg, self.tub.base_path, record)
                        for record in self.tub if select(record)]
        self.len = len(self.records)
        self.len_with_deleted = self.tub.manifest.current_index
        if self.len > 0:
            tub_screen().index = 0
            tub_screen().ids.data_plot.update_dataframe_from_tub()
            msg = f'Loaded tub {self.file_path} with {self.len} records'
        else:
            msg = f'No records in tub {self.file_path}'
        tub_screen().status(msg)
        if syncAnnotate:
            annotate_screen().update_mask_tub()
        return True


class LabelBar(BoxLayout):
    """ Widget that combines a label with a progress bar. This is used to
        display the record fields in the data panel."""
    field = StringProperty()
    field_property = ObjectProperty()
    config = ObjectProperty()
    msg = ''

    def update(self, record):
        """ This function is called everytime the current record is updated"""
        if not record:
            return
        field, index = decompose(self.field)
        if field in record.underlying:
            val = record.underlying[field]
            if index is not None:
                val = val[index]
            # Update bar if a field property for this field is known
            if self.field_property:
                norm_value = get_norm_value(val, self.config,
                                            self.field_property)
                new_bar_val = (norm_value + 1) * 50 if \
                    self.field_property.centered else norm_value * 100
                self.ids.bar.value = new_bar_val
            self.ids.field_label.text = self.field
            if isinstance(val, float) or isinstance(val, np.float32):
                text = f'{val:+07.3f}'
            elif isinstance(val, int):
                text = f'{val:10}'
            else:
                text = str(val)
            self.ids.value_label.text = text
        else:
            Logger.error(f'Record: Bad record {record.underlying["_index"]} - '
                         f'missing field {self.field}')


class DataPanel(BoxLayout):
    """ Data panel widget that contains the label/bar widgets and the drop
        down menu to select/deselect fields."""
    record = ObjectProperty()
    # dual mode is used in the pilot arena where we only show angle and
    # throttle or speed
    dual_mode = BooleanProperty(False)
    auto_text = StringProperty(LABEL_SPINNER_TEXT)
    throttle_field = StringProperty('user/throttle')
    link = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.labels = {}
        self.screen = ObjectProperty()

    def add_remove(self):
        """ Method to add or remove a LabelBar. Depending on the value of the
            drop down menu the LabelBar is added if it is not present otherwise
            removed."""
        field = self.ids.data_spinner.text
        if field is LABEL_SPINNER_TEXT:
            return
        if field in self.labels and not self.dual_mode:
            self.remove_widget(self.labels[field])
            del(self.labels[field])
            self.screen.status(f'Removing {field}')
        else:
            # in dual mode replace the second entry with the new one
            if self.dual_mode and len(self.labels) == 2:
                k, v = list(self.labels.items())[-1]
                self.remove_widget(v)
                del(self.labels[k])
            field_property = rc_handler.field_properties.get(decompose(field)[0])
            cfg = tub_screen().ids.config_manager.config
            lb = LabelBar(field=field, field_property=field_property, config=cfg)
            self.labels[field] = lb
            self.add_widget(lb)
            lb.update(self.record)
            if len(self.labels) == 2:
                self.throttle_field = field
            self.screen.status(f'Adding {field}')
        if self.screen.name == 'tub':
            self.screen.ids.data_plot.plot_from_current_bars()
        self.ids.data_spinner.text = LABEL_SPINNER_TEXT
        self.auto_text = field

    def on_record(self, obj, record):
        """ Kivy function that is called every time self.record changes"""
        for v in self.labels.values():
            v.update(record)

    def clear(self):
        for v in self.labels.values():
            self.remove_widget(v)
        self.labels.clear()


class FullImage(Image):
    """ Widget to display an image that fills the space. """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.core_image = None

    def update(self, record):
        """ This method is called ever time a record gets updated. """
        try:
            img_arr = self.get_image(record)
            pil_image = PilImage.fromarray(img_arr)
            bytes_io = io.BytesIO()
            pil_image.save(bytes_io, format='png')
            bytes_io.seek(0)
            self.core_image = CoreImage(bytes_io, ext='png')
            self.texture = self.core_image.texture
        except KeyError as e:
            Logger.error(f'Record: Missing key: {e}')
        except Exception as e:
            Logger.error(f'Record: Bad record: {e}')

    def get_image(self, record):
        return record.image()

class FullAnnotateImage(Image):
    """ Widget to display an image that fills the space. """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.core_image = None

    def update(self, record):
        """ This method is called ever time a record gets updated. """
        try:
            img_arr = self.get_image(record)
            pil_image = PilImage.fromarray(img_arr)
            bytes_io = io.BytesIO()
            pil_image.save(bytes_io, format='png')
            bytes_io.seek(0)
            self.core_image = CoreImage(bytes_io, ext='png')
            self.texture = self.core_image.texture
        except KeyError as e:
            Logger.error(f'Record: FullAnnotateImage : Missing key: {e}')
        except Exception as e:
            Logger.error(f'Record: FullAnnotateImage : Bad record: {e}')

    def update_with_mask(self, record, mask_left, mask_right):
        try:
            color_left = np.array([255/255, 30/255, 30/255, 0.6])
            color_right = np.array([30/255, 255/255, 30/255, 0.6])
            #h, w = mask_left.shape if mask_left!=None else mask_right.shape
            h, w = mask_left.shape 
            mask_left_image = mask_left.reshape(h, w, 1) * color_left.reshape(1, 1, -1)
            mask_right_image = mask_right.reshape(h, w, 1) * color_right.reshape(1, 1, -1)
            pil_mask_left_image = PilImage.fromarray((mask_left_image * 255).astype(np.uint8))
            pil_mask_right_image = PilImage.fromarray((mask_right_image * 255).astype(np.uint8))
            img_arr = self.get_image(record)
            pil_image = PilImage.fromarray(img_arr)
            pil_image.paste (pil_mask_left_image,(0, 0), pil_mask_left_image)
            pil_image.paste (pil_mask_right_image,(0, 0), pil_mask_right_image)
            bytes_io = io.BytesIO()
            pil_image.save(bytes_io, format='png')
            bytes_io.seek(0)
            self.core_image = CoreImage(bytes_io, ext='png')
            self.texture = self.core_image.texture
        except KeyError as e:
            Logger.error(f'Record index {record.underlying["_index"]}: update_with_mask : Missing key: {e}')
        except Exception as e:
            Logger.error(f'Record index {record.underlying["_index"]}: update_with_mask : Bad record: {e}')
            traceback.print_exc() 

    def get_image(self, record):
        return record.image()


class ControlPanel(BoxLayout):
    """ Class for control panel navigation. """
    screen = ObjectProperty()
    nextThrottleFallingEdge = BooleanProperty(False)
    nextEmptyMask = BooleanProperty(False)
    speed = NumericProperty(1.0)
    record_display = StringProperty()
    labels_display = StringProperty()
    throttle_field = StringProperty('user/throttle')
    clock = None
    fwd = None

    def start(self, fwd=True, continuous=False, skip=False):
        """
        Method to cycle through records if either single <,> or continuous
        <<, >> buttons are pressed
        :param fwd:         If we go forward or backward
        :param continuous:  If we do <<, >> or <, >
        :param skip:        If skip function must be used
        :return:            None
        """
        # this widget it used in two screens, so reference the original location
        # of the config which is the config manager in the tub screen
        cfg = tub_screen().ids.config_manager.config
        hz = cfg.DRIVE_LOOP_HZ if cfg else 20
        time.sleep(0.1)
        if (skip):
            if (self.nextThrottleFallingEdge) :
                call = partial (self.skip_next_throttle_down)
            elif (self.nextEmptyMask):
                call = partial (self.skip_next_unmask)
        else:
            call = partial(self.step, fwd, continuous)
        if continuous:
            self.fwd = fwd
            s = float(self.speed) * hz
            cycle_time = 1.0 / s
        else:
            cycle_time = 0.08
        self.clock = Clock.schedule_interval(call, cycle_time)

    def step(self, fwd=True, continuous=False, *largs):
        """
        Updating a single step and cap/floor the index so we stay w/in the tub.
        :param fwd:         If we go forward or backward
        :param continuous:  If we are in continuous mode <<, >>
        :param largs:       dummy
        :return:            None
        """
        if self.screen.index is None:
            self.screen.status("No tub loaded")
            return
        new_index = self.screen.index + (1 if fwd else -1)
        if new_index >= tub_screen().ids.tub_loader.len:
            new_index = 0
        elif new_index < 0:
            new_index = tub_screen().ids.tub_loader.len - 1
        self.screen.index = new_index
        msg = f'Donkey {"run" if continuous else "step"} ' \
              f'{"forward" if fwd else "backward"}'
        if not continuous:
            msg += f' - you can also use {"<right>" if fwd else "<left>"} key'
        else:
            msg += ' - you can toggle run/stop with <space>'
        self.screen.status(msg)

    def skip_next_throttle_down(self, *largs):
        if self.screen.index is None:
            self.screen.status("No tub loaded")
            return
        starting_index = self.screen.index + 1
        if starting_index >= tub_screen().ids.tub_loader.len:
            starting_index = 0
        elif starting_index < 0:
            starting_index = tub_screen().ids.tub_loader.len - 1
        new_index = starting_index
        for index, rec in enumerate (tub_screen().ids.tub_loader.records[starting_index:]): 
            if (float(rec.underlying[self.throttle_field] < 0.05)):
                new_index = index+starting_index
                break
        self.screen.index = new_index
        msg = f'Donkey {"step"} ' \
              f'{"forward"}'
        msg += ' - Low Throttle detected'
        self.screen.status(msg)

    def skip_next_unmask(self, *largs):
        annotate_screen().skip_next_unmask(largs)

    def stop(self):
        if self.clock:
            self.clock.cancel()
            self.clock = None

    def restart(self):
        if self.clock:
            self.stop()
            self.start(self.fwd, True)

    def update_speed(self, up=True):
        """ Method to update the speed on the controller"""
        values = self.ids.control_spinner.values
        idx = values.index(self.ids.control_spinner.text)
        if up and idx < len(values) - 1:
            self.ids.control_spinner.text = values[idx + 1]
        elif not up and idx > 0:
            self.ids.control_spinner.text = values[idx - 1]

    def set_button_status(self, disabled=True):
        """ Method to disable(enable) all buttons. """
        self.ids.run_bwd.disabled = self.ids.run_fwd.disabled = \
            self.ids.step_fwd.disabled = self.ids.step_bwd.disabled = disabled

    def on_keyboard(self, key, scancode):
        """ Method to chack with keystroke has ben sent. """
        if key == ' ':
            if self.clock and self.clock.is_triggered:
                self.stop()
                self.set_button_status(disabled=False)
                self.screen.status('Donkey stopped')
            else:
                self.start(continuous=True)
                self.set_button_status(disabled=True)
        elif scancode == 79:
            self.step(fwd=True)
        elif scancode == 80:
            self.step(fwd=False)
        elif scancode == 45:
            self.update_speed(up=False)
        elif scancode == 46:
            self.update_speed(up=True)

class AnnotateLeftPanel(BoxLayout):
    """ Class for control panel navigation. """
    screen = ObjectProperty()
    record_display = StringProperty()

    def box(self,left=False):
        annotate_screen().box(left)
        pass

    def point (self, left=False, fg=True):
        annotate_screen().point(left, fg)
        pass

    def reset_poi (self):
        annotate_screen().reset_poi()

    def reset_mask(self, left=False):
        annotate_screen().reset_mask(left)
        
    def show_hide_rules(self):
        annotate_screen().show_hide_rules()

    def on_keyboard(self, key, scancode):
        """ Method to chack with keystroke has ben sent. """
        pass

class AnnotateRightPanel(BoxLayout):
    """ Class for control panel navigation. """
    screen = ObjectProperty()
    record_display = StringProperty()
    labels_display = StringProperty()

    def instanciate_sam(self, event):
        annotate_screen().instanciate_sam()

    def clean_mask(self) :
        annotate_screen().clean_mask()

    def clean_all_mask(self) :
        annotate_screen().clean_all_mask()

    def box(self,left=False):
        annotate_screen().box(left)
        pass

    def point (self, left=False, fg=True):
        annotate_screen().point(left, fg)
        pass

    def load_sam(self) :
        annotate_screen().status("Loading SAM, please wait...")
        Clock.schedule_once(self.instanciate_sam)

    def reset_mask(self, left=False):
        annotate_screen().reset_mask(left)

    def set_auto_mask(self):
        annotate_screen().set_auto_mask()

    def on_keyboard(self, key, scancode):
        """ Method to chack with keystroke has ben sent. """
        pass


class PaddedBoxLayout(BoxLayout):
    pass


class TubEditor(PaddedBoxLayout):
    """ Tub editor widget. Contains left/right index interval and the
        manipulator buttons for deleting / restoring and reloading """
    lr = ListProperty([0, 0])

    def set_lr(self, is_l=True):
        """ Sets left or right range to the current tub record index """
        if not tub_screen().current_record:
            return
        self.lr[0 if is_l else 1] = tub_screen().current_record.underlying['_index']

    def del_lr(self, is_del):
        """ Deletes or restores records in chosen range """
        tub = tub_screen().ids.tub_loader.tub
        mask_tub = annotate_screen().mask_tub
        if self.lr[1] >= self.lr[0]:
            selected = list(range(*self.lr))
        else:
            last_id = tub.manifest.current_index
            selected = list(range(self.lr[0], last_id))
            selected += list(range(self.lr[1]))
        tub.delete_records(selected) if is_del else tub.restore_records(selected)
        mask_tub.delete_records(selected) if is_del else tub.restore_records(selected)

    def restore_all(self):
        tub = tub_screen().ids.tub_loader.tub
        mask_tub = annotate_screen().mask_tub
        tub.restore_records(list(range(0, tub_screen().ids.tub_loader.len - 1)))
        mask_tub.restore_records(list(range(0, tub_screen().ids.tub_loader.len - 1)))

    def apply_label_lr(self):
        tub = tub_screen().ids.tub_loader.tub
        if self.lr[1] >= self.lr[0]:
            selected = list(range(*self.lr))
        else:
            last_id = tub.manifest.current_index
            selected = list(range(self.lr[0], last_id))
            selected += list(range(self.lr[1]))
        tub.label_records(selected, self.ids.label_spinner.text)


class TubFilter(PaddedBoxLayout):
    """ Tub filter widget. """
    filter_expression = StringProperty(None)
    record_filter = StringProperty(rc_handler.data.get('record_filter', ''))

    def update_filter(self):
        filter_text = self.ids.record_filter.text
        config = tub_screen().ids.config_manager.config
        # empty string resets the filter
        if filter_text == '':
            self.record_filter = ''
            self.filter_expression = ''
            rc_handler.data['record_filter'] = self.record_filter
            if hasattr(config, 'TRAIN_FILTER'):
                delattr(config, 'TRAIN_FILTER')
            tub_screen().status(f'Filter cleared')
            return
        filter_expression = self.create_filter_string(filter_text)
        try:
            record = tub_screen().current_record
            filter_func_text = f"""def filter_func(record): 
                                       return {filter_expression}       
                                """
            # creates the function 'filter_func'
            ldict = {}
            exec(filter_func_text, globals(), ldict)
            filter_func = ldict['filter_func']
            res = filter_func(record)
            status = f'Filter result on current record: {res}'
            if isinstance(res, bool):
                self.record_filter = filter_text
                self.filter_expression = filter_expression
                rc_handler.data['record_filter'] = self.record_filter
                setattr(config, 'TRAIN_FILTER', filter_func)
            else:
                status += ' - non bool expression can\'t be applied'
            status += ' - press <Reload tub> to see effect'
            tub_screen().status(status)
        except Exception as e:
            tub_screen().status(f'Filter error on current record: {e}')

    @staticmethod
    def create_filter_string(filter_text, record_name='record'):
        """ Converts text like 'user/angle' into 'record.underlying['user/angle']
        so that it can be used in a filter. Will replace only expressions that
        are found in the tub inputs list.

        :param filter_text: input text like 'user/throttle > 0.1'
        :param record_name: name of the record in the expression
        :return:            updated string that has all input fields wrapped
        """
        for field in tub_screen().current_record.underlying.keys():
            field_list = filter_text.split(field)
            if len(field_list) > 1:
                filter_text = f'{record_name}.underlying["{field}"]'\
                    .join(field_list)
        return filter_text


class DataPlot(PaddedBoxLayout):
    """ Data plot panel which embeds matplotlib interactive graph"""
    df = ObjectProperty(force_dispatch=True, allownone=True)

    def plot_from_current_bars(self, in_app=True):
        """ Plotting from current selected bars. The DataFrame for plotting
            should contain all bars except for strings fields and all data is
            selected if bars are empty.  """
        tub = tub_screen().ids.tub_loader.tub
        field_map = dict(zip(tub.manifest.inputs, tub.manifest.types))
        # Use selected fields or all fields if nothing is slected
        all_cols = tub_screen().ids.data_panel.labels.keys() or self.df.columns
        cols = [c for c in all_cols if decompose(c)[0] in field_map
                and field_map[decompose(c)[0]] not in ['image_array', 'str', 'callback']]

        df = self.df[cols]
        if df is None:
            return
        # Don't plot the milliseconds time stamp as this is a too big number
        df = df.drop(labels=['_timestamp_ms'], axis=1, errors='ignore')

        if in_app:
            tub_screen().ids.graph.df = df
        else:
            fig = px.line(df, x=df.index, y=df.columns, title=tub.base_path)
            fig.update_xaxes(rangeslider=dict(visible=True))
            fig.show()

    def unravel_vectors(self):
        """ Unravels vector and list entries in tub which are created
            when the DataFrame is created from a list of records"""
        manifest = tub_screen().ids.tub_loader.tub.manifest
        for k, v in zip(manifest.inputs, manifest.types):
            if v == 'vector' or v == 'list':
                dim = len(tub_screen().current_record.underlying[k])
                df_keys = [k + f'_{i}' for i in range(dim)]
                self.df[df_keys] = pd.DataFrame(self.df[k].tolist(),
                                                index=self.df.index)
                self.df.drop(k, axis=1, inplace=True)

    def update_dataframe_from_tub(self):
        """ Called from TubManager when a tub is reloaded/recreated. Fills
            the DataFrame from records, and updates the dropdown menu in the
            data panel."""
        generator = (t.underlying for t in tub_screen().ids.tub_loader.records)
        self.df = pd.DataFrame(generator).dropna()
        to_drop = {'cam/image_array', 'cam/undistorted_rgb', 'cam/depth_array'}
        self.df.drop(labels=to_drop, axis=1, errors='ignore', inplace=True)
        self.df.set_index('_index', inplace=True)
        self.unravel_vectors()
        tub_screen().ids.data_panel.ids.data_spinner.values = self.df.columns
        self.plot_from_current_bars()


class TabBar(BoxLayout):
    manager = ObjectProperty(None)

    def disable_only(self, bar_name):
        this_button_name = bar_name + '_btn'
        for button_name, button in self.ids.items():
            button.disabled = button_name == this_button_name


class TubScreen(Screen):
    """ First screen of the app managing the tub data. """
    index = NumericProperty(None, force_dispatch=True)
    current_record = ObjectProperty(None)
    keys_enabled = BooleanProperty(False)

    def initialise(self, e):
        self.ids.config_manager.load_action()
        self.ids.tub_loader.update_tub()

    def _get_label(self,index):
        labels=[]
        for aLabel in self.ids.tub_loader.tub.manifest.labeled_indexes:
            if index in self.ids.tub_loader.tub.manifest.labeled_indexes[aLabel] :
                labels.append(aLabel)
        return labels

    def change_index (self, index):
        if (index != self.index):
            self.ids.slider.value = index
         
    def on_index(self, obj, index, syncAnnotateScreen=True):
        """ Kivy method that is called if self.index changes"""
        if index >= 0:
            self.current_record = self.ids.tub_loader.records[index]
            self.ids.slider.value = index
            if syncAnnotateScreen and annotate_screen().mask_records:
                annotate_screen().change_index(index)

    def on_current_record(self, obj, record):
        """ Kivy method that is called if self.current_record changes."""
        self.ids.img.update(record)
        i = record.underlying['_index']
        t = len(tub_screen().ids.tub_loader.tub)
        self.ids.control_panel.record_display = f"Record {i:06} [{t:06} kept]"
        self.ids.control_panel.labels_display = f"{self._get_label(i)}"

    def status(self, msg):
        self.ids.status.text = msg

    def on_keyboard(self, instance, keycode, scancode, key, modifiers):
        if self.keys_enabled:
            self.ids.control_panel.on_keyboard(key, scancode)

class AnnotateScreen(Screen):
    index = NumericProperty(None, force_dispatch=True)
    current_record = ObjectProperty(None)
    current_mask_record = ObjectProperty(None)
    keys_enabled = BooleanProperty(False)
    sam_loaded = BooleanProperty(False)
    config = ObjectProperty()
    mask_tub = ObjectProperty(None)
    mask_records = None
    mask_len = NumericProperty(1)
    drawing = StringProperty()

    def initialise(self):
        self.point_size = 20
        if 'linux' in platform.system().lower():
            self.point_size = 5
        self.drawing = ""
        self.poi={}
        self.segment = None
        self.touch_down_pos = [0, 0]
        self.show_rules = False
        self.bbox_mask_left = None
        self.bbox_mask_right = None
        self.auto_mask = False
        self.auto_mask_offset = 5
        self.poi_left_foreground_points=[]                
        self.poi_left_background_points=[]                
        self.poi_right_foreground_points=[]                
        self.poi_right_background_points=[]                
        self.poi_left_box=[]
        self.poi_right_box=[]
        tub_screen().ids.config_manager.load_action()
        tub_screen().ids.tub_loader.update_tub()
        self.init_mask_tub()
        self.update_mask_tub()
        self.config = tub_screen().ids.config_manager.config
        if tub_screen().ids.tub_loader.records:
            self.index = tub_screen().index
            self.current_record = tub_screen().ids.tub_loader.records[0]
            if self.mask_records:
                self.current_mask_record=self.mask_records[0]
            self.update_image_with_mask(self.current_record, index=self.index)

    def get_empty_mask(self):
        pil_image_ref = PilImage.fromarray(tub_screen().ids.tub_loader.records[0].image())
        default_mask = np.array(PilImage.new('RGB', pil_image_ref.size, color='black'))[:, :, 0]
        return default_mask
    
    def init_mask_tub(self):
        if not tub_screen().ids.tub_loader.file_path:
            return False

        if not os.path.exists(os.path.join(tub_screen().ids.tub_loader.file_path, 'manifest.json')):
            self.status(f'Path {tub_screen().ids.tub_loader.file_path} is not a valid tub.')
            return False

        self.binary_mask_file_path = os.path.join(tub_screen().ids.tub_loader.file_path,'binary_masks')

        if not os.path.exists(self.binary_mask_file_path):
            self.status(f'Path {self.binary_mask_file_path} : No directory for binary mask found, creating it.')
            os.makedirs(self.binary_mask_file_path, exist_ok=True)

        if not os.path.exists(os.path.join(self.binary_mask_file_path, 'manifest.json')):
            self.status(f'Path {self.binary_mask_file_path} : No manifest for binary mask found.')

        try:
            if self.mask_tub:
                self.mask_tub.close()
            inputs = ['left_mask', 'right_mask', 'left_poi', 'right_poi']
            types = ['left_mask', 'right_mask','dict', 'dict']
            self.mask_tub = Tub(self.binary_mask_file_path, inputs=inputs, types=types, lr=True)
        except Exception as e:
            self.status(f'Failed loading binary mask tub: {str(e)}')
            return False

        self.mask_records = [TubRecord(self.config, self.mask_tub.base_path, record)
                        for record in self.mask_tub]
        self.mask_len = len(self.mask_records)
        if tub_screen().ids.tub_loader.records:
            default_mask = self.get_empty_mask()
            self.mask_height, self.mask_width = default_mask.shape
            to_create = tub_screen().ids.tub_loader.len_with_deleted-self.mask_tub.manifest.current_index
            if (to_create>0):
                print("Init/Complete default or imported mask tub, wait a minute ...") 
                for idx in range(to_create):
                    left_mask = default_mask
                    right_mask = default_mask
                    if os.path.isfile(f'{self.mask_tub.base_path}/left/{idx}_cam_image_array_.npy'):
                        existing_record = np.load(f'{self.mask_tub.base_path}/left/{idx}_cam_image_array_.npy')
                        left_mask = existing_record
                    if os.path.isfile(f'{self.mask_tub.base_path}/right/{idx}_cam_image_array_.npy'):
                        existing_record = np.load(f'{self.mask_tub.base_path}/right/{idx}_cam_image_array_.npy')
                        right_mask = existing_record
                    mask_record={'left_mask':left_mask, 'right_mask':right_mask, 'left_poi':[], 'right_poi':[]}
                    self.mask_tub.write_record(mask_record)
        else:
            self.status(f'Unale to create default mask, no records in tub')

    def instanciate_sam(self):
        self.segment = Segmentation(self.config)
        self.sam_loaded = True
        self.ids.annotate_left_panel.ids.left_box.disabled = False
        self.ids.annotate_right_panel.ids.right_box.disabled = False
        self.ids.annotate_right_panel.ids.auto_mask.disabled = False
        self.ids.annotate_left_panel.ids.left_foreground_point.disabled = False
        self.ids.annotate_left_panel.ids.left_background_point.disabled = False
        self.ids.annotate_right_panel.ids.right_foreground_point.disabled = False
        self.ids.annotate_right_panel.ids.right_background_point.disabled = False
        self.status("SAM Loaded")

    def load_action(self):
        """ Update tub from the file path"""
        self.init_mask_tub()
        self.update_mask_tub()

    def update_mask_tub(self):

        if self.mask_tub == None:
            return
        self.mask_records = [TubRecord(self.config, self.mask_tub.base_path, record)
                        for record in self.mask_tub]
        self.mask_len = len(self.mask_records)

        if self.mask_len > 0:
            msg = f'Loaded tub {self.binary_mask_file_path} with {self.mask_len} records'
        else:
            msg = f'No records in tub {self.binary_mask_file_path}'
        self.status(msg)
        return True

    def clean_mask(self):
        if self.index is None:
            self.status("No tub loaded")
            return
        mask_left = self.current_mask_record.image(key='left_mask', as_nparray=True, format='NPY', reload=True, image_base_path='left')
        mask_right = self.current_mask_record.image(key='right_mask', as_nparray=True, format='NPY', reload=True, image_base_path='right')
        if np.any(mask_left):
            nb_left_area = has_multiple_areas (mask_left)
            if nb_left_area:
                widest_left_area = extract_widest_area (mask_left)
                if widest_left_area is not None:
                    self.store_mask (self.current_mask_record, index=self.current_mask_record.underlying['_index'], mask=widest_left_area, left=True)
                    print(f"Left mask index {self.current_mask_record.underlying['_index']} fixed !")
                else :
                    print(f"Was not able to process Left mask index {self.current_mask_record.underlying['_index']}")
        if np.any(mask_right):
            nb_right_area = has_multiple_areas (mask_right)
            if nb_right_area:
                widest_right_area = extract_widest_area (mask_right)
                if widest_right_area is not None:
                    self.store_mask (self.current_mask_record, index=self.current_mask_record.underlying['_index'], mask=widest_right_area, left=False)
                    print(f"Left mask index {self.current_mask_record.underlying['_index']} fixed !")
                else:
                    print(f"Was not able to process Right mask index {self.current_mask_record.underlying['_index']}")

    def clean_all_mask(self):
        if self.index is None:
            self.status("No tub loaded")
            return
        starting_index = 0
        nb_fix = 0
        fixed_left=[]
        fixed_right=[]
        for index, rec in enumerate (self.mask_records[starting_index:]): 
            mask_left = rec.image(key='left_mask', as_nparray=True, format='NPY', reload=True, image_base_path='left')
            mask_right = rec.image(key='right_mask', as_nparray=True, format='NPY', reload=True, image_base_path='right')
            if np.any(mask_left):
                nb_left_area = has_multiple_areas (mask_left)
                if nb_left_area:
                    widest_left_area = extract_widest_area (mask_left)
                    if widest_left_area is not None:
                        self.store_mask (rec, index=rec.underlying['_index'], mask=widest_left_area, left=True)
                        fixed_left.append(rec.underlying['_index'])
                        nb_fix+=1
                    else :
                        print(f"Was not able to process Left mask index {index}")
            if np.any(mask_right):
                nb_right_area = has_multiple_areas (mask_right)
                if nb_right_area:
                    widest_right_area = extract_widest_area (mask_right)
                    if widest_right_area is not None:
                        self.store_mask (rec, index=rec.underlying['_index'], mask=widest_right_area, left=False)
                        fixed_right.append(rec.underlying['_index'])
                        nb_fix+=1
                    else:
                        print(f"Was not able to process Right mask index {index}")
            
        print(f"Fixed left indexes {fixed_left} ")
        print(f"Fixed right indexes {fixed_right} ")

        self.status(f'Mask tub cleanup done ({nb_fix} masks fixed)')

    def reset_mask(self, left=False):
        default_mask = self.get_empty_mask()
        if self.mask_records:
            mask_record = self.mask_records[self.index]
            if left:
                mask_record={'left_mask':default_mask, 'right_mask':mask_record.underlying['right_mask'], 'left_poi':[], 'right_poi':mask_record.underlying['right_poi']}
            else:
                mask_record={'left_mask':mask_record.underlying['left_mask'], 'right_mask':default_mask, 'left_poi':mask_record.underlying['left_poi'], 'right_poi':[]}
            self.mask_tub.write_record(mask_record, index=self.index)
            self.status(f'Mask tub record index {self.index} reinitialized')
            self.update_image_with_mask(self.current_record, index=self.index)

    def set_auto_mask(self):
        if (self.auto_mask):
            self.auto_mask=False
        else:
            self.auto_mask=True
        self.status(f'Auto Mask set to {self.auto_mask}')

    def update_image_with_mask (self, record, index=None):
        if self.mask_records:
            if (index != None): 
                input_box = np.array([])
                self.current_mask_record = self.mask_records[index]
                mask_left = self.current_mask_record.image(key='left_mask', as_nparray=True, format='NPY', reload=True, image_base_path='left')
                mask_right = self.current_mask_record.image(key='right_mask', as_nparray=True, format='NPY', reload=True, image_base_path='right')

                #np.set_printoptions(threshold=sys.maxsize)
                if not np.any(mask_left): #if no mask
                    if self.bbox_mask_left and self.auto_mask: #and previous one and auto_mask
                        input_box=np.array([max(min(self.bbox_mask_left[0],self.bbox_mask_left[1])-self.auto_mask_offset,0), 
                                    max(min(self.bbox_mask_left[2],self.bbox_mask_left[3])-self.auto_mask_offset,0),
                                    min(max(self.bbox_mask_left[0],self.bbox_mask_left[1])+self.auto_mask_offset, self.mask_width),
                                    min(max(self.bbox_mask_left[2],self.bbox_mask_left[3])+self.auto_mask_offset, self.mask_height)])
                        mask_left, scores, logits = self.segment_image(input_points=None, input_labels=None, input_box=input_box)
                        self.store_mask (self.current_mask_record, index=index, mask=mask_left, left=True)
                        mask_left = self.current_mask_record.image(key='left_mask', as_nparray=True, format='NPY', reload=True, image_base_path='left')
                if np.any(mask_left): #if mask
                    left = np.where(mask_left == 1)
                    self.bbox_mask_left = np.min(left[1]), np.max(left[1]), np.min(left[0]), np.max(left[0])
                if not np.any(mask_right): #if no mask
                    if self.bbox_mask_right and self.auto_mask: #and previous one and auto_mask
                        print ("Previous right mask found")
                        input_box=np.array([max(min(self.bbox_mask_right[0],self.bbox_mask_right[1])-self.auto_mask_offset,0), 
                                    max(min(self.bbox_mask_right[2],self.bbox_mask_right[3])-self.auto_mask_offset,0),
                                    min(max(self.bbox_mask_right[0],self.bbox_mask_right[1])+self.auto_mask_offset, self.mask_width),
                                    min(max(self.bbox_mask_right[2],self.bbox_mask_right[3])+self.auto_mask_offset, self.mask_height)])
                        mask_right, scores, logits = self.segment_image(input_points=None, input_labels=None, input_box=input_box)
                        self.store_mask (self.current_mask_record,  index=index, mask=mask_right, left=False)
                        mask_right = self.current_mask_record.image(key='right_mask', as_nparray=True, format='NPY', reload=True, image_base_path='right')
                if np.any(mask_right): #if mask
                    right = np.where(mask_right == 1)
                    self.bbox_mask_right = np.min(right[1]), np.max(right[1]), np.min(right[0]), np.max(right[0])
                self.ids.annotate_img.update_with_mask(record, mask_left, mask_right)
                return
        self.ids.annotate_img.update(record)

    def change_index (self, index):
        if (index != self.index):
            self.ids.annotate_slider.value = index

    def on_index(self, obj, index):
        """ Kivy method that is called if self.index changes. Here we update
            self.current_record and the slider value. """
        if tub_screen().ids.tub_loader.records:
            tub_screen().change_index (index)
            self.current_record = tub_screen().ids.tub_loader.records[index]
            self.ids.annotate_slider.value = index
            if self.mask_records:
                self.current_mask_record = self.mask_records[index]
#        if self.ids.mask_records:
#            current_mask_record
        self.reset_poi()
        self.update_image_with_mask(self.current_record, index=index)

    def on_current_record(self, obj, record):
        """ Kivy method that is called when self.current_index changes. Here
            we update the images and the control panel entry."""
        if not record:
            return
        i = record.underlying['_index']
        self.ids.annotate_left_panel.record_display = f"Record {i:06}"
    
    def skip_next_unmask(self, *largs):
        if self.index is None:
            self.status("No tub loaded")
            return
        starting_index = self.index + 1
        if starting_index >= tub_screen().ids.tub_loader.len:
            starting_index = 0
        elif starting_index < 0:
            starting_index = tub_screen().ids.tub_loader.len - 1

        new_index=self.index
        for index, rec in enumerate (self.mask_records[starting_index:]): 
            mask_left = rec.image(key='left_mask', as_nparray=True, format='NPY', reload=True, image_base_path='left')
            mask_right = rec.image(key='right_mask', as_nparray=True, format='NPY', reload=True, image_base_path='right')
            if (not np.any(mask_left)) and (not np.any(mask_right)):
                new_index = index+starting_index
                self.status(f"No mask set on this image")
                break
            nb_left_area = has_multiple_areas (mask_left)
            nb_right_area = has_multiple_areas (mask_right)
            if nb_left_area:
                new_index = index+starting_index
                self.status(f"Left mask has more than one area")
                break
            if nb_right_area:
                new_index = index+starting_index
                self.status(f"Right mask has more than one area")
                break

        self.index = new_index

    def status(self, msg):
        self.ids.annotate_status.text = msg

    def clear_left_markers(self):
        self.canvas.remove_group(u"left_box")
        self.canvas.remove_group(u"left_point_foreground")
        self.canvas.remove_group(u"left_point_background")
        if 'left_point_foreground' in self.poi:
            del self.poi['left_point_foreground']
        if 'left_point_background' in self.poi:
            del self.poi['left_point_background']


    def clear_right_markers(self):
        self.canvas.remove_group(u"right_box")
        self.canvas.remove_group(u"right_point_foreground")
        self.canvas.remove_group(u"right_point_background")
        if 'right_point_foreground' in self.poi:
            del self.poi['right_point_foreground']
        if 'right_point_background' in self.poi:
            del self.poi['right_point_background']

    def clear_box_markers(self):
        self.canvas.remove_group(u"right_box")
        self.canvas.remove_group(u"left_box")
        if 'box' in self.poi:
            del self.poi['box']

    def clear_points_markers(self):
        self.canvas.remove_group(u"right_point_foreground")
        self.canvas.remove_group(u"right_point_background")
        self.canvas.remove_group(u"left_point_foreground")
        self.canvas.remove_group(u"left_point_background")
        if 'left_point_foreground' in self.poi:
            del self.poi['left_point_foreground']
        if 'left_point_background' in self.poi:
            del self.poi['left_point_background']
        if 'right_point_foreground' in self.poi:
            del self.poi['right_point_foreground']
        if 'right_point_background' in self.poi:
            del self.poi['right_point_background']

    def clear_markers(self):
        self.clear_left_markers()
        self.clear_right_markers()

    def raz_poi_record(self):
        self.poi = {}

    def box(self, left=False):
        self.status(f"Draw {'right' if left==False else 'left'} box")
        if left and 'left' not in self.drawing:
            self.clear_right_markers()
            self.clear_box_markers()
        if not left and 'right' not in self.drawing:
            self.clear_left_markers()
            self.clear_box_markers()
        self.drawing = f"{'left' if left==True else 'right'}_box"

    def point(self, left=False, fg=True):
        self.status(f"Draw {'right' if left==False else 'left'} {'background' if fg==False else 'foreground'} point")
        if left and "left_point" not in self.drawing:
            self.clear_right_markers()
            self.clear_box_markers()
        if not left and "right_point" not in self.drawing:
            self.clear_left_markers()
            self.clear_box_markers()
        self.drawing = f"{'left' if left==True else 'right'}_point_{'background' if fg==False else 'foreground'}"

    def reset_poi(self):
        self.clear_markers()
        self.raz_poi_record()

    def show_hide_rules(self):
        #Toggle
        if self.show_rules:
            self.show_rules = False
        else:
            self.show_rules = True

        if self.show_rules:
            with self.canvas:
                Color(0.7,0.7,0.5,0.5,mode='rgba')
                for h in[50,55,60,65,70,75,80,85,90]:
                    x0, y0 = self.map_original_coordinates (0, self.mask_height-h)
                    x1, y1 = self.map_original_coordinates (self.mask_width, self.mask_height-h)
                    Line(points=[x0,y0,x1,y1], width=2, group=u'rules')
        else:
            self.canvas.remove_group(u"rules")

    def on_keyboard(self, instance, keycode, scancode, key, modifiers):
        if self.keys_enabled:
            self.ids.control_panel.on_keyboard(key, scancode)
            self.ids.annotate_left_panel.on_keyboard(key, scancode)
            self.ids.annotate_right_panel.on_keyboard(key, scancode)

    def get_original_coordinates(self,x,y):
        ox=map_range(x,self.ids.annotate_img.x,self.ids.annotate_img.x+self.ids.annotate_img.size[0],0,self.ids.annotate_img.norm_image_size[0], True)
        oy=map_range(y,self.ids.annotate_img.y,self.ids.annotate_img.y+self.ids.annotate_img.size[1],0,self.ids.annotate_img.norm_image_size[1], True)
        return ox, oy

    def map_original_coordinates(self,x,y):
        ox=map_range(x,0,self.ids.annotate_img.norm_image_size[0], self.ids.annotate_img.x,self.ids.annotate_img.x+self.ids.annotate_img.size[0], True)
        oy=map_range(y,0,self.ids.annotate_img.norm_image_size[1], self.ids.annotate_img.y,self.ids.annotate_img.y+self.ids.annotate_img.size[1], True)
        return ox, oy

    def update_poi_record(self):
        self.poi_left_foreground_points=[]                
        self.poi_left_background_points=[]                
        self.poi_right_foreground_points=[]                
        self.poi_right_background_points=[]                
        self.poi_left_box=[]
        self.poi_right_box=[]
        for key in self.poi:
            if 'point_foreground' in key:
                for point in self.poi[key]:
                    x,y = self.get_original_coordinates (point.pos[0],point.pos[1] )
                    if 'left' in key:
                        self.poi_left_foreground_points.append({'x':x, 'y':y})
                    else:
                        self.poi_right_foreground_points.append({'x':x, 'y':y})

            if 'point_background' in key:
                for point in self.poi[key]:
                    x,y = self.get_original_coordinates (point.pos[0],point.pos[1] )
                    if 'left' in key:
                        self.poi_left_background_points.append({'x':x, 'y':y})
                    else:
                        self.poi_right_background_points.append({'x':x, 'y':y})

            if 'box' in key:
                x0,y0 = self.get_original_coordinates (self.poi[key].pos[0],self.poi[key].pos[1] )
                x1,y1 = self.get_original_coordinates (self.poi[key].pos[0]+self.poi[key].size[0],self.poi[key].pos[1]+self.poi[key].size[1] )
                if 'left' in self.drawing:
                    self.poi_left_box = {'x0':x0, 'y0':y0,'x1':x1, 'y1':y1}
                else:
                    self.poi_right_box = {'x0':x0, 'y0':y0,'x1':x1, 'y1':y1}
               
    def store_mask(self, record, index, mask, left=True):
            mask_record={}
            mask_record['left_poi']=record.underlying['left_poi']
            mask_record['right_poi']=record.underlying['right_poi']
            mask_record['left_mask']=record.underlying['left_mask']
            mask_record['right_mask']=record.underlying['right_mask']
            if left: #'left_mask', 'right_mask', 'left_poi', 'right_poi'
                mask_record['left_mask'] = mask
                if (self.poi_left_foreground_points and self.poi_left_background_points and self.poi_left_box):
                    mask_record['left_poi']={'fg_pts':self.poi_left_foreground_points, 'bg_pts':self.poi_left_background_points, 'box':self.poi_left_box}
            else:        
                mask_record['right_mask'] = mask
                if (self.poi_right_foreground_points and self.poi_right_background_points and self.poi_right_box):
                    mask_record['right_poi']={'fg_pts':self.poi_right_foreground_points, 'bg_pts':self.poi_right_background_points, 'box':self.poi_right_box}
            self.mask_tub.write_record(mask_record, index=index)
            self.status(f'Mask tub record index {index} written')
        
    def segment_image(self, input_points=None, input_labels=None, input_box=None):
            if self.segment:
                self.status(f'Processing image with SAM ....')
                self.segment.set_image(self.current_record.image())
                masks, scores, logits = self.segment.predict(
                    input_points=input_points if input_points is not None and input_points.size>0 else None,
                    input_labels=input_labels if input_labels is not None and input_labels.size>0 else None,
                    input_box=input_box)
                widest_area = extract_widest_area (masks[0])
                if widest_area is not None:
                    mask=widest_area
                else: mask = masks[0]
                return mask, scores, logits
            else:
                return None, None, None

    def segment_image_from_ui(self):
        self.update_poi_record()
        left = True if 'left' in self.drawing else False
        input_points = np.empty((0,2), int)
        input_labels = np.empty((0,2), int)
        input_box = np.array([])
        if left :
            for pts in self.poi_left_foreground_points:
                input_points=np.append(input_points, np.array([[pts['x'],self.mask_height-pts['y']]]), axis=0)
                input_labels=np.append(input_labels, np.array([[1]]))
            for pts in self.poi_left_background_points:
                input_points=np.append(input_points, np.array([[pts['x'], self.mask_height-pts['y']]]), axis=0)
                input_labels=np.append(input_labels, np.array([[0]]))
            if len(self.poi_left_box)>0:
                input_box=np.array([min(self.poi_left_box['x0'],self.poi_left_box['x1']), 
                                    min(self.mask_height-self.poi_left_box['y0'],self.mask_height-self.poi_left_box['y1']),
                                    max(self.poi_left_box['x0'],self.poi_left_box['x1']),
                                    max(self.mask_height-self.poi_left_box['y0'],self.mask_height-self.poi_left_box['y1'])])
        else :
            for pts in self.poi_right_foreground_points:
                input_points=np.append(input_points, np.array([[pts['x'], self.mask_height-pts['y']]]), axis=0)
                input_labels=np.append(input_labels, np.array([[1]]))
            for pts in self.poi_right_background_points:
                input_points=np.append(input_points, np.array([[pts['x'], self.mask_height-pts['y']]]), axis=0)
                input_labels=np.append(input_labels, np.array([[0]]))
            if len(self.poi_right_box)>0:
                input_box=np.array([min(self.poi_right_box['x0'],  self.poi_right_box['x1']),
                                    min(self.mask_height-self.poi_right_box['y0'], self.mask_height-self.poi_right_box['y1']),
                                    max(self.poi_right_box['x0'],  self.poi_right_box['x1']),
                                    max(self.mask_height-self.poi_right_box['y0'], self.mask_height-self.poi_right_box['y1'])])

        print (f"NP Array input_points : {input_points}")
        print (f"NP Array input_labels : {input_labels}")
        print (f"NP Array input_box : {input_box}")
        if self.segment and input_box.size >0:
            mask, scores, logits = self.segment_image (input_points, input_labels, input_box)
            idx = self.current_record.underlying['_index']

            self.store_mask (self.current_mask_record, index=idx,  mask=mask, left=left)
            self.update_image_with_mask(self.current_record, index=idx)
            self.reset_poi()
        else:
            self.status(f'SAM not loaded')

        # if 'right_point' in self.drawing:
        #     print (self.canvas.get_group(u"right_point_foreground"))
        #     print (self.canvas.get_group(u"right_point_backgroun"))
        # if 'left_point' in self.drawing:
        #     print (self.canvas.get_group(u"left_point_foreground"))
        #     print (self.canvas.get_group(u"left_point_background"))
        # if 'right_box' in self.drawing:
        #     print (self.canvas.get_group(u"right_box"))
        # if 'left_box' in self.drawing:
        #     print (self.canvas.get_group(u"left_box"))
                #self.mask_tub.write_record(self.current_record)

    def on_touch_down(self, touch):
        if not self.ids.annotate_img.collide_point(*touch.pos):
            return super(Screen, self).on_touch_down(touch)
        touch.grab(self)
        px, py = self.get_original_coordinates(touch.x, touch.y)
        if len(self.drawing)>0:
            with self.canvas:
                opacity=1.0
                if 'box' in self.drawing:
                    opacity = 0.2
                if 'left' in self.drawing :
                    if 'background' in  self.drawing:
                        Color(0.5,0,0,opacity,mode='rgba')
                    else:
                        Color(1,0,0,opacity,mode='rgba')
                else:
                    if 'background' in  self.drawing:
                        Color(0,0.5,0,opacity,mode='rgba')
                    else:
                        Color(0,1,0,opacity,mode='rgba')
                if "point" in self.drawing:
                    if self.drawing not in self.poi:
                        self.poi[self.drawing]=[]
                    self.poi[self.drawing].append(Ellipse(pos=(touch.x-self.point_size/2, touch.y-self.point_size/2),size=(self.point_size,self.point_size), group = self.drawing))
                if "box" in self.drawing:
                    self.touch_down_pos = (touch.x, touch.y)
                    #ud['box']=Line(rectangle=(touch.x, touch.y,5,5), width=2, group = self.drawing)
                    self.poi['box']=Rectangle(pos=(touch.x, touch.y),size=(5,5),group = self.drawing)
        return True
    
    def on_touch_up(self, touch):
        if touch.grab_current is self:
            # I receive my grabbed touch, I must ungrab it!
            touch.ungrab(self)
            self.segment_image_from_ui()
            # if 'box' in self.drawing:
            #     self.clear_box_markers()
            #     self.drawing = ''
            return True
        else:
            return super(Screen, self).on_touch_up(touch)

    
    def on_touch_move(self, touch):
        if touch.grab_current is self:
            if len(self.drawing)>0:
                with self.canvas:
                    ud = touch.ud
                    if "box" in self.drawing:
                        self.poi['box'].size = (touch.x - self.touch_down_pos[0], touch.y - self.touch_down_pos[1])
            return True
        else:
            return super(Screen, self).on_touch_move(touch)

    
class PilotLoader(BoxLayout, FileChooserBase):
    """ Class to mange loading of the config file from the car directory"""
    num = StringProperty()
    model_type = StringProperty()
    pilot = ObjectProperty(None)
    filters = ['*.h5', '*.tflite', '*.savedmodel', '*.trt', '*.onnx']

    def load_action(self):
        if self.file_path and self.pilot:
            try:
                self.pilot.load(os.path.join(self.file_path))
                rc_handler.data['pilot_' + self.num] = self.file_path
                rc_handler.data['model_type_' + self.num] = self.model_type
                self.ids.pilot_spinner.text = self.model_type
                Logger.info(f'Pilot: Successfully loaded {self.file_path}')
            except FileNotFoundError:
                Logger.error(f'Pilot: Model {self.file_path} not found')
            except Exception as e:
                Logger.error(f'Failed loading {self.file_path}: {e}')

    def on_model_type(self, obj, model_type):
        """ Kivy method that is called if self.model_type changes. """
        if self.model_type and self.model_type != 'Model type':
            cfg = tub_screen().ids.config_manager.config
            if cfg:
                self.pilot = get_model_by_type(self.model_type, cfg)
                self.ids.pilot_button.disabled = False
                if 'tflite' in self.model_type:
                    self.filters = ['*.tflite']
                elif 'tensorrt' in self.model_type:
                    self.filters = ['*.trt']
                elif 'onnx' in self.model_type:
                    self.filters = ['*.onnx']
                else:
                    self.filters = ['*.h5', '*.savedmodel']

    def on_num(self, e, num):
        """ Kivy method that is called if self.num changes. """
        self.file_path = rc_handler.data.get('pilot_' + self.num, '')
        self.model_type = rc_handler.data.get('model_type_' + self.num, '')


class OverlayImage(FullImage):
    """ Widget to display the image and the user/pilot data for the tub. """
    index = NumericProperty(None, force_dispatch=True)
    pilot = ObjectProperty()
    pilot_record = ObjectProperty()
    throttle_field = StringProperty('user/throttle')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_left = True

    def on_index(self, obj, index):
        """ Kivy method that is called if self.index changes. Here we update
            self.current_record and the slider value. """
        if tub_screen().ids.tub_loader.records:
            self.current_record = tub_screen().ids.tub_loader.records[index]
            self.ids.slider.value = index

    def augment(self, img_arr):
        if pilot_screen().trans_list:
            img_arr = pilot_screen().transformation.run(img_arr)
        if pilot_screen().aug_list:
            img_arr = pilot_screen().augmentation.run(img_arr)
        return img_arr

    def get_image(self, record):
        from donkeycar.management.makemovie import MakeMovie
        config = tub_screen().ids.config_manager.config
        orig_img_arr = super().get_image(record)
        aug_img_arr = self.augment(orig_img_arr)
        img_arr = copy(aug_img_arr)
        angle = record.underlying['user/angle']
        throttle = get_norm_value(
            record.underlying[self.throttle_field], config,
            rc_handler.field_properties[self.throttle_field])
        rgb = (0, 255, 0)
        MakeMovie.draw_line_into_image(angle, throttle, False, img_arr, rgb)
        if not self.pilot:
            return img_arr

        output = (0, 0)
        try:
            # Not each model is supported in each interpreter
            if len(self.pilot.get_input_shapes()) > 1:
                output = self.pilot.run(aug_img_arr, np.array([0,1,0]))
            else:    
                output = self.pilot.run(aug_img_arr)
        except Exception as e:
            Logger.error(e)

        rgb = (0, 0, 255)
        MakeMovie.draw_line_into_image(output[0], output[1], True, img_arr, rgb)
        out_record = copy(record)
        out_record.underlying['pilot/angle'] = output[0]
        # rename and denormalise the throttle output
        pilot_throttle_field \
            = rc_handler.data['user_pilot_map'][self.throttle_field]
        out_record.underlying[pilot_throttle_field] \
            = get_norm_value(output[1], tub_screen().ids.config_manager.config,
                             rc_handler.field_properties[self.throttle_field],
                             normalised=False)
        self.pilot_record = out_record
        return img_arr        

class PilotScreen(Screen):
    """ Screen to do the pilot vs pilot comparison ."""
    index = NumericProperty(None, force_dispatch=True)
    current_record = ObjectProperty(None)
    keys_enabled = BooleanProperty(False)
    aug_list = ListProperty(force_dispatch=True)
    augmentation = ObjectProperty()
    trans_list = ListProperty(force_dispatch=True)
    transformation = ObjectProperty()
    config = ObjectProperty()

    def on_index(self, obj, index):
        """ Kivy method that is called if self.index changes. Here we update
            self.current_record and the slider value. """
        if tub_screen().ids.tub_loader.records:
            self.current_record = tub_screen().ids.tub_loader.records[index]
            self.ids.slider.value = index

    def on_current_record(self, obj, record):
        """ Kivy method that is called when self.current_index changes. Here
            we update the images and the control panel entry."""
        if not record:
            return
        i = record.underlying['_index']
        self.ids.pilot_control.record_display = f"Record {i:06}"
        self.ids.img_1.update(record)
        self.ids.img_2.update(record)

    def initialise(self, e):
        self.ids.pilot_loader_1.on_model_type(None, None)
        self.ids.pilot_loader_1.load_action()
        self.ids.pilot_loader_2.on_model_type(None, None)
        self.ids.pilot_loader_2.load_action()
        mapping = copy(rc_handler.data['user_pilot_map'])
        del(mapping['user/angle'])
        self.ids.data_in.ids.data_spinner.values = mapping.keys()
        self.ids.data_in.ids.data_spinner.text = 'user/angle'
        self.ids.data_panel_1.ids.data_spinner.disabled = True
        self.ids.data_panel_2.ids.data_spinner.disabled = True

    def map_pilot_field(self, text):
        """ Method to return user -> pilot mapped fields except for the
            initial value called Add/remove. """
        if text == LABEL_SPINNER_TEXT:
            return text
        return rc_handler.data['user_pilot_map'][text]

    def set_brightness(self, val=None):
        if not self.config:
            return
        if self.ids.button_bright.state == 'down':
            self.config.AUG_MULTIPLY_RANGE = (val, val)
            if 'MULTIPLY' not in self.aug_list:
                self.aug_list.append('MULTIPLY')
        elif 'MULTIPLY' in self.aug_list:
            self.aug_list.remove('MULTIPLY')
        # update dependency
        self.on_aug_list(None, None)

    def set_blur(self, val=None):
        if not self.config:
            return
        if self.ids.button_blur.state == 'down':
            self.config.AUG_BLUR_RANGE = (val, val)
            if 'BLUR' not in self.aug_list:
                self.aug_list.append('BLUR')
        elif 'BLUR' in self.aug_list:
            self.aug_list.remove('BLUR')
        # update dependency
        self.on_aug_list(None, None)

    def on_aug_list(self, obj, aug_list):
        if not self.config:
            return
        self.config.AUGMENTATIONS = self.aug_list
        self.augmentation = ImageAugmentation(self.config, 'AUGMENTATIONS')
        self.on_current_record(None, self.current_record)

    def on_trans_list(self, obj, trans_list):
        if not self.config:
            return
        self.config.TRANSFORMATIONS = self.trans_list
        self.transformation = ImageAugmentation(self.config, 'TRANSFORMATIONS')
        self.on_current_record(None, self.current_record)

    def set_mask(self, state):
        if state == 'down':
            self.ids.status.text = 'Trapezoidal mask on'
            self.trans_list.append('TRAPEZE')
        else:
            self.ids.status.text = 'Trapezoidal mask off'
            if 'TRAPEZE' in self.trans_list:
                self.trans_list.remove('TRAPEZE')

    def set_crop(self, state):
        if state == 'down':
            self.ids.status.text = 'Crop on'
            self.trans_list.append('CROP')
        else:
            self.ids.status.text = 'Crop off'
            if 'CROP' in self.trans_list:
                self.trans_list.remove('CROP')

    def status(self, msg):
        self.ids.status.text = msg

    def on_keyboard(self, instance, keycode, scancode, key, modifiers):
        if self.keys_enabled:
            self.ids.pilot_control.on_keyboard(key, scancode)


class ScrollableLabel(ScrollView):
    pass


class DataFrameLabel(Label):
    pass


class TransferSelector(BoxLayout, FileChooserBase):
    """ Class to select transfer model"""
    filters = ['*.h5']


class TrainScreen(Screen):
    """ Class showing the training screen. """
    config = ObjectProperty(force_dispatch=True, allownone=True)
    database = ObjectProperty()
    pilot_df = ObjectProperty(force_dispatch=True)
    tub_df = ObjectProperty(force_dispatch=True)

    def train_call(self, model_type, *args):
        # remove car directory from path
        tub_path = tub_screen().ids.tub_loader.tub.base_path
        transfer = self.ids.transfer_spinner.text
        if transfer != 'Choose transfer model':
            transfer = os.path.join(self.config.MODELS_PATH, transfer + '.h5')
        else:
            transfer = None
        try:
            history = train(self.config, tub_paths=tub_path,
                            model_type=model_type,
                            transfer=transfer,
                            comment=self.ids.comment.text)
            self.ids.status.text = f'Training completed.'
            self.ids.comment.text = 'Comment'
            self.ids.transfer_spinner.text = 'Choose transfer model'
            self.reload_database()
        except Exception as e:
            Logger.error(e)
            self.ids.status.text = f'Train failed see console'
        finally:
            self.ids.train_button.state = 'normal'

    def train(self, model_type):
        self.config.SHOW_PLOT = False
        Thread(target=self.train_call, args=(model_type,)).start()
        self.ids.status.text = f'Training started.'

    def set_config_attribute(self, input):
        try:
            val = json.loads(input)
        except ValueError:
            val = input

        att = self.ids.cfg_spinner.text.split(':')[0]
        setattr(self.config, att, val)
        self.ids.cfg_spinner.values = self.value_list()
        self.ids.status.text = f'Setting {att} to {val} of type ' \
                               f'{type(val).__name__}'

    def value_list(self):
        if self.config:
            return [f'{k}: {v}' for k, v in self.config.__dict__.items()]
        else:
            return ['select']

    def on_config(self, obj, config):
        if self.config and self.ids:
            self.ids.cfg_spinner.values = self.value_list()
            self.reload_database()

    def reload_database(self):
        if self.config:
            self.database = PilotDatabase(self.config)

    def on_database(self, obj, database):
        group_tubs = self.ids.check.state == 'down'
        pilot_txt, tub_txt, pilot_names = self.database.pretty_print(group_tubs)
        self.ids.scroll_tubs.text = tub_txt
        self.ids.scroll_pilots.text = pilot_txt
        self.ids.transfer_spinner.values \
            = ['Choose transfer model'] + pilot_names
        self.ids.delete_spinner.values \
            = ['Pilot'] + pilot_names


class CarScreen(Screen):
    """ Screen for interacting with the car. """
    config = ObjectProperty(force_dispatch=True, allownone=True)
    files = ListProperty()
    car_dir = StringProperty(rc_handler.data.get('robot_car_dir', '~/mycar'))
    event = ObjectProperty(None, allownone=True)
    connection = ObjectProperty(None, allownone=True)
    pid = NumericProperty(None, allownone=True)
    pilots = ListProperty()
    is_connected = BooleanProperty(False)

    def initialise(self):
        self.event = Clock.schedule_interval(self.connected, 3)

    def list_remote_dir(self, dir):
        if self.is_connected:
            cmd = f'ssh {self.config.PI_USERNAME}@{self.config.PI_HOSTNAME}' + \
                  f' "ls {dir}"'
            listing = os.popen(cmd).read()
            adjusted_listing = listing.split('\n')[1:-1]
            return adjusted_listing
        else:
            return []

    def list_car_dir(self, dir):
        self.car_dir = dir
        self.files = self.list_remote_dir(dir)
        # non-empty director found
        if self.files:
            rc_handler.data['robot_car_dir'] = dir

    def update_pilots(self):
        model_dir = os.path.join(self.car_dir, 'models')
        self.pilots = self.list_remote_dir(model_dir)

    def pull(self, tub_dir):
        target = f'{self.config.PI_USERNAME}@{self.config.PI_HOSTNAME}' + \
               f':{os.path.join(self.car_dir, tub_dir)}'
        dest = self.config.DATA_PATH
        if self.ids.create_dir.state == 'normal':
            target += '/'
        cmd = ['rsync', '-rv', '--progress', '--partial', target, dest]
        Logger.info('car pull: ' + str(cmd))
        proc = Popen(cmd, shell=False, stdout=PIPE, text=True,
                     encoding='utf-8', universal_newlines=True)
        repeats = 100
        call = partial(self.show_progress, proc, repeats, True)
        event = Clock.schedule_interval(call, 0.0001)

    def send_pilot(self):
        # add trailing '/'
        src = os.path.join(self.config.MODELS_PATH,'')
        # check if any sync buttons are pressed and update path accordingly
        buttons = ['h5', 'savedmodel', 'tflite', 'trt', 'onnx']
        select = [btn for btn in buttons if self.ids[f'btn_{btn}'].state
                  == 'down']
        # build filter: for example this rsyncs all .tfilte and .trt models
        # --include=*.trt/*** --include=*.tflite --exclude=*
        filter = ['--include=database.json']
        for ext in select:
            if ext in ['savedmodel', 'trt']:
                ext += '/***'
            filter.append(f'--include=*.{ext}')
        # if nothing selected, sync all
        if not select:
            filter.append('--include=*')
        else:
            filter.append('--exclude=*')
        dest = f'{self.config.PI_USERNAME}@{self.config.PI_HOSTNAME}:' + \
               f'{os.path.join(self.car_dir, "models")}'
        cmd = ['rsync', '-rv', '--progress', '--partial', *filter, src, dest]
        Logger.info('car push: ' + ' '.join(cmd))
        proc = Popen(cmd, shell=False, stdout=PIPE,
                     encoding='utf-8', universal_newlines=True)
        repeats = 0
        call = partial(self.show_progress, proc, repeats, False)
        event = Clock.schedule_interval(call, 0.0001)

    def show_progress(self, proc, repeats, is_pull, e):
        # find 'to-check=33/4551)' in OSX or 'to-chk=33/4551)' in
        # Linux which is end of line
        pattern = 'to-(check|chk)=(.*)\)'

        def end():
            # call ended this stops the schedule
            if is_pull:
                button = self.ids.pull_tub
                self.ids.pull_bar.value = 0
                # merge in previous deleted indexes which now might have been
                # overwritten
                old_tub = tub_screen().ids.tub_loader.tub
                if old_tub:
                    deleted_indexes = old_tub.manifest.deleted_indexes
                    tub_screen().ids.tub_loader.update_tub()
                    if deleted_indexes:
                        new_tub = tub_screen().ids.tub_loader.tub
                        new_tub.manifest.add_deleted_indexes(deleted_indexes)
            else:
                button = self.ids.send_pilots
                self.ids.push_bar.value = 0
                self.update_pilots()
            button.disabled = False

        if proc.poll() is not None:
            end()
            return False
        # find the next repeats lines with update info
        count = 0
        while True:
            stdout_data = proc.stdout.readline()
            if stdout_data:
                res = re.search(pattern, stdout_data)
                if res:
                    if count < repeats:
                        count += 1
                    else:
                        remain, total = tuple(res.group(2).split('/'))
                        bar = 100 * (1. - float(remain) / float(total))
                        if is_pull:
                            self.ids.pull_bar.value = bar
                        else:
                            self.ids.push_bar.value = bar
                        return True
            else:
                # end of stream command completed
                end()
                return False

    def connected(self, event):
        if not self.config:
            return
        if self.connection is None:
            if not hasattr(self.config, 'PI_USERNAME') or \
                    not hasattr(self.config, 'PI_HOSTNAME'):
                self.ids.connected.text = 'Requires PI_USERNAME, PI_HOSTNAME'
                return
            # run new command to check connection status
            cmd = ['ssh',
                   '-o ConnectTimeout=3',
                   f'{self.config.PI_USERNAME}@{self.config.PI_HOSTNAME}',
                   'date']
            self.connection = Popen(cmd, shell=False, stdout=PIPE,
                                    stderr=STDOUT, text=True,
                                    encoding='utf-8', universal_newlines=True)
        else:
            # ssh is already running, check where we are
            return_val = self.connection.poll()
            self.is_connected = False
            if return_val is None:
                # command still running, do nothing and check next time again
                status = 'Awaiting connection...'
                self.ids.connected.color = 0.8, 0.8, 0.0, 1
            else:
                # command finished, check if successful and reset connection
                if return_val == 0:
                    status = 'Connected'
                    self.ids.connected.color = 0, 0.9, 0, 1
                    self.is_connected = True
                else:
                    status = 'Disconnected'
                    self.ids.connected.color = 0.9, 0, 0, 1
                self.connection = None
            self.ids.connected.text = status

    def drive(self):
        model_args = ''
        if self.ids.pilot_spinner.text != 'No pilot':
            model_path = os.path.join(self.car_dir, "models",
                                      self.ids.pilot_spinner.text)
            model_args = f'--type {self.ids.type_spinner.text} ' + \
                         f'--model {model_path}'
        cmd = ['ssh',
               f'{self.config.PI_USERNAME}@{self.config.PI_HOSTNAME}',
               f'source env/bin/activate; cd {self.car_dir}; ./manage.py '
               f'drive {model_args} 2>&1']
        Logger.info(f'car connect: {cmd}')
        proc = Popen(cmd, shell=False, stdout=PIPE, text=True,
                     encoding='utf-8', universal_newlines=True)
        while True:
            stdout_data = proc.stdout.readline()
            if stdout_data:
                # find 'PID: 12345'
                pattern = 'PID: .*'
                res = re.search(pattern, stdout_data)
                if res:
                    try:
                        self.pid = int(res.group(0).split('PID: ')[1])
                        Logger.info(f'car connect: manage.py drive PID: '
                                    f'{self.pid}')
                    except Exception as e:
                        Logger.error(f'car connect: {e}')
                    return
                Logger.info(f'car connect: {stdout_data}')
            else:
                return

    def stop(self):
        if self.pid:
            cmd = f'ssh {self.config.PI_USERNAME}@{self.config.PI_HOSTNAME} '\
                  + f'kill {self.pid}'
            out = os.popen(cmd).read()
            Logger.info(f"car connect: Kill PID {self.pid} + {out}")
            self.pid = None


class StartScreen(Screen):
    img_path = os.path.realpath(os.path.join(
        os.path.dirname(__file__),
        '../parts/web_controller/templates/static/donkeycar-logo-sideways.png'))
    pass


class DonkeyApp(App):
    start_screen = None
    tub_screen = None
    annotate_screen = None
    train_screen = None
    pilot_screen = None
    car_screen = None
    title = 'Donkey Manager'

    def initialise(self, event):
        self.tub_screen.ids.config_manager.load_action()
        self.annotate_screen.initialise()
        self.pilot_screen.initialise(event)
        self.car_screen.initialise()
        # This builds the graph which can only happen after everything else
        # has run, therefore delay until the next round.
        Clock.schedule_once(self.tub_screen.ids.tub_loader.update_tub)

    def build(self):
        Window.bind(on_request_close=self.on_request_close)
        self.start_screen = StartScreen(name='donkey')
        self.tub_screen = TubScreen(name='tub')
        self.annotate_screen = AnnotateScreen(name='annotate')
        self.train_screen = TrainScreen(name='train')
        self.pilot_screen = PilotScreen(name='pilot')
        self.car_screen = CarScreen(name='car')
        Window.bind(on_keyboard=self.tub_screen.on_keyboard)
        Window.bind(on_keyboard=self.annotate_screen.on_keyboard)
        Window.bind(on_keyboard=self.pilot_screen.on_keyboard)
        Clock.schedule_once(self.initialise)
        sm = ScreenManager()
        sm.add_widget(self.start_screen)
        sm.add_widget(self.tub_screen)
        sm.add_widget(self.annotate_screen)
        sm.add_widget(self.train_screen)
        sm.add_widget(self.pilot_screen)
        sm.add_widget(self.car_screen)
        return sm

    def on_request_close(self, *args):
        tub = self.tub_screen.ids.tub_loader.tub
        if tub:
            tub.close()
        Logger.info("Good bye Donkey")
        return False


def main():
    tub_app = DonkeyApp()
    tub_app.run()


if __name__ == '__main__':
    main()
