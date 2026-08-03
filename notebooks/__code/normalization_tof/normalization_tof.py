import glob
import h5py
import logging
import logging as notebook_logging
import os
from pathlib import Path
import numpy as np
import pandas as pd

import ipywidgets as widgets
import plotly.express as px

from IPython.display import HTML, clear_output, display
from ipywidgets import interactive
from PIL import Image

from __code.normalization_tof import Roi
from __code._utilities.list import extract_list_of_runs_from_string
from __code._utilities.nexus import extract_file_path_from_nexus
from __code.normalization_tof import DataType
from __code._utilities.json import load_json

from __code.ipywe.fileselector import FileSelectorPanel as MyFileSelectorPanel
from __code.normalization_tof import (
    DetectorType,
    RebinCustomBasis,
    RebinCustomScale,
    RebinMode,
    autoreduce_dir,
    distance_source_detector_m,
    raw_dir,
)
from __code.normalization_tof.config import DEBUG_DATA, timepix1_config, timepix3_config
from __code.normalization_tof.jump_folder_selector import FileSelectorPanelWithJumpFolders as MyFileSelectorPanelWithJumpFolders
from __code.normalization_tof.normalization_for_timepix1_timepix3 import (
    load_data_using_multithreading,
    # normalization,
    normalization_with_list_of_full_path,
    retrieve_list_of_tif,
)
from __code.normalization_tof.utilities import (
    DEFAULT_BLACK_FILTER_BACKGROUND_SHAPE_FILE,
    build_rebin_bin_groups,
    calculate_time_lambda_energy_arrays,
    get_detector_offset_from_nexus,
)


class NormalizationTof:
    sample_folder = None
    sample_run_numbers = None
    sample_run_numbers_selected = None
    
    # if the spectra file is missing, the program will create the spectra array on the fly
    spectra_array = None 
    spectra_file_found = True
    list_spectra_file_found = []

    integrated_data = None

    check_nbr_tiff = {DataType.sample: [], DataType.ob: [], DataType.dc: []}

    ob_folder = None
    ob_run_numbers = None
    ob_run_numbers_selected = None

    dc_folder = None
    dc_run_numbers = None
    dc_run_numbers_selected = None

    output_folder = None
    
    # {'full_path_data': {'data': None, 'nexus': None}}
    dict_sample = {}
    dict_ob = {}
    dict_dc = {}
    dict_bragg_edge_cd_sample_background = {}
    dict_bragg_edge_cd_ob_background = {}
    dict_closed_slits_sample_background = {}
    dict_closed_slits_ob_background = {}

    # {'short_name': 'full_path_data'}
    dict_short_name_full_path = {
        "sample": {},
        "ob": {},
        "dc": {},
        "bragg_edge_cd_sample_background": {},
        "bragg_edge_cd_ob_background": {},
        "closed_slits_sample_background": {},
        "closed_slits_ob_background": {},
    }

    dict_ob_runs = None
    dict_ob_data = None
    dict_dc_data = None
    bragg_edge_cd_sample_background_run_numbers_selected = None
    bragg_edge_cd_ob_background_run_numbers_selected = None
    bragg_edge_cd_sample_background_check_nbr_tiff = []
    bragg_edge_cd_ob_background_check_nbr_tiff = []
    closed_slits_sample_background_run_numbers_selected = None
    closed_slits_ob_background_run_numbers_selected = None
    closed_slits_sample_background_check_nbr_tiff = []
    closed_slits_ob_background_check_nbr_tiff = []

    roi = None  # full spectrum ROI
    
    # container
    container_roi = None # container only ROI
    default_roi = Roi(left=156, top=156, width=200, height=200)
    default_container_roi = Roi(left=150, top=150, width=40, height=40)
    we_need_to_automatically_save_the_container_roi = False
    rect_container = None
    container_roi_file = None
    
    def initialize(self):
        LOG_PATH = "/SNS/VENUS/shared/log/"
        file_name, ext = os.path.splitext(os.path.basename(__file__))
        user_name = os.getlogin()  # add user name to the log file name
        log_file_name = os.path.join(LOG_PATH, f"{file_name}_{user_name}.log")
        notebook_logging.basicConfig(
            filename=log_file_name,
            filemode="w",
            format="[%(levelname)s] - %(asctime)s - %(message)s",
            level=notebook_logging.INFO,
        )
        notebook_logging.info(f"*** Starting a new script {file_name} ***")

    # def __new__(cls, *args, **kwargs):
    #     logging.info(f"Creating instance of {cls.__name__}")

    def __init__(self, working_dir=None, debug=False):

        self.initialize()

        if debug:
            self.working_dir = DEBUG_DATA.working_dir
            self.shared_dir = self.working_dir + "/shared"
            self.output_dir = DEBUG_DATA.output_folder
            self.default_roi = Roi(left=DEBUG_DATA.roi[0], top=DEBUG_DATA.roi[1], 
                          width=DEBUG_DATA.roi[2], height=DEBUG_DATA.roi[3])
            self.default_container_roi = Roi(left=DEBUG_DATA.container_roi[0], 
                                             top=DEBUG_DATA.container_roi[1],
                                             width=DEBUG_DATA.container_roi[2], 
                                             height=DEBUG_DATA.container_roi[3])
            self.detector_type = DEBUG_DATA.detector_type
        
        else:
            self.working_dir = working_dir
            self.shared_dir = os.path.join(self.working_dir, "shared")
            self.output_dir = os.path.join(self.working_dir, "shared")
            self.detector_type = DetectorType.tpx1
        
        self.nexus_folder = os.path.join(self.working_dir, "nexus")
        self.debug = debug
        _, _facility, _beamline, ipts, _ = self.shared_dir.split("/")

        self.ipts = ipts
        self.ipts_folder = self.working_dir
        self.instrument = _beamline.upper()

        # self.autoreduce_dir = autoreduce_dir[_beamline][0] + str(ipts) + autoreduce_dir[_beamline][1]
        # self.shared_dir = str(Path(shared_dir[self.instrument][0]) / str(ipts) / shared_dir[self.instrument][1])
        self.shared_dir = Path("/") / _facility / self.instrument / str(ipts) / "shared"

        notebook_logging.info(f"Instrument: {self.instrument}")
        notebook_logging.info(f"Working dir: {self.working_dir}")
        notebook_logging.info(f"IPTS: {self.ipts}")
        notebook_logging.info(f"IPTS folder: {self.ipts_folder}")
        notebook_logging.info(f"facility: {_facility}")
        notebook_logging.info(f"nexus folder: {self.nexus_folder}")
        notebook_logging.info(f"Shared dir: {self.shared_dir}")

        display(HTML("<span style='color:blue; font-size:16px'>Select detector type</span>"))
        self.detector_type_widget = widgets.Dropdown(
            options=[DetectorType.tpx1_legacy, DetectorType.tpx1, DetectorType.tpx3],
            value=self.detector_type,
            layout=widgets.Layout(width="400px"),
            disabled=False,
        )
        display(self.detector_type_widget)
        
    def reset_sample_dicts(self):
        self.dict_sample = {}
        self.dict_short_name_full_path["sample"] = {}
        self.check_nbr_tiff[DataType.sample] = []

    def reset_ob_dicts(self):
        self.dict_short_name_full_path["ob"] = {}
        self.dict_ob = {}
        self.dict_ob_runs = None
        self.dict_ob_data = None
        self.check_nbr_tiff[DataType.ob] = []

    def reset_dc_dicts(self):
        self.dict_short_name_full_path["dc"] = {}
        self.dict_dc = {}
        self.dict_dc_runs = None
        self.dict_dc_data = None
        self.check_nbr_tiff[DataType.dc] = []

    def reset_measured_background_dicts(self, background_key: str, role: str):
        short_key = f"{background_key}_{role}_background"
        self.dict_short_name_full_path[short_key] = {}
        setattr(self, f"dict_{short_key}", {})
        setattr(self, f"{short_key}_check_nbr_tiff", [])

    def reset_bragg_edge_cd_sample_background_dicts(self):
        self.reset_measured_background_dicts("bragg_edge_cd", "sample")

    def reset_bragg_edge_cd_ob_background_dicts(self):
        self.reset_measured_background_dicts("bragg_edge_cd", "ob")

    def reset_closed_slits_sample_background_dicts(self):
        self.reset_measured_background_dicts("closed_slits", "sample")

    def reset_closed_slits_ob_background_dicts(self):
        self.reset_measured_background_dicts("closed_slits", "ob")

    def setup_default_paths(self):
        notebook_logging.info("Setting up default paths...")
        self.detector_type = self.detector_type_widget.value
        self.raw_dir = Path(raw_dir[self.instrument][self.detector_type][0]) / str(self.ipts)
        self.autoreduce_dir = (
            Path(autoreduce_dir[self.instrument][self.detector_type][0])
            / str(self.ipts)
            / Path(autoreduce_dir[self.instrument][self.detector_type][1])
        )
        self.sample_dir = Path(self.autoreduce_dir) / "raw" / "radiography"
        self.ob_dir = Path(self.autoreduce_dir) / "ob"

        notebook_logging.info(f"\tAutoreduce dir: {self.autoreduce_dir}")
        notebook_logging.info(f"\tDetector type: {self.detector_type}")
        notebook_logging.info(f"\tRaw dir: {self.raw_dir}")
        # self.select_folder(instruction="Select sample top folder", next_function=self.sample_folder_selected)

    def select_sample_run_numbers(self):
        self.setup_default_paths()

        if self.debug:
            sample_runs = DEBUG_DATA.sample_runs_selected
            sample_run_numbers_list = []
            for _run in sample_runs:
                _, number = _run.split("_")
                sample_run_numbers_list.append(number)
            str_sample_run_numbers = ", ".join(sample_run_numbers_list)
        else:
            str_sample_run_numbers = ""

        sample_label = widgets.HTML(
            value="<b><font color='green'>List of sample run numbers (ex: 8702, 8704-8706)</font></b>"
        )

        self.sample_run_numbers_widget = widgets.Textarea(
            value=str_sample_run_numbers, placeholder="", layout=widgets.Layout(width="400px")
        )
        vertical_layout = widgets.VBox(
            [
                sample_label,
                self.sample_run_numbers_widget,
            ]
        )
        display(vertical_layout)

        display(HTML("<span style='font-size: 16px; color:red'>OR</span>"))
        # give focus to the widgets self.sample_run_numbers_widget
        # self.sample_run_numbers_widget.focus()

        self.select_folder(
            instruction="Browse sample runs to normalize",
            next_function=self.save_sample_run_numbers_selected,
            multiple=True,
            start_dir=self.sample_dir,
            newdir_toolbar_button=False,
        )

    def save_sample_run_numbers_selected(self, runs_selected):
        self.sample_run_numbers_selected = runs_selected

    def retrieve_file_path_from_nexus(self, run_number):
        """
        Retrieve the full path to the NeXus file for the given run number.
        This function should be implemented to read the NeXus file and extract the path.
        """
        notebook_logging.info(f"Retrieving file path from NeXus for run number: {run_number}")
        # Placeholder implementation, replace with actual logic to read NeXus file
        nexus_file_path = Path(self.nexus_folder) / f"{self.instrument.upper()}_{run_number}.nxs.h5"
        notebook_logging.info(f"\tNeXus file path: {nexus_file_path}")
        if nexus_file_path.exists():
            return extract_file_path_from_nexus(nexus_file_path)
        else:
            return None

    def extract_full_path(self, run_number=None):
        """
        Extract the full path to the run number based on the detector type.
        """
        notebook_logging.info(f"Extracting full path for run number: {run_number} with detector type: {self.detector_type}")

        if run_number is None:
            raise ValueError("Run number must be provided")

        if self.detector_type == DetectorType.tpx1_legacy:
            return Path(self.autoreduce_dir) / f"Run_{run_number}"

        elif self.detector_type in [DetectorType.tpx1, DetectorType.tpx3]:
            # retrieve the path from the NeXus file
            file_path = self.retrieve_file_path_from_nexus(run_number)
            if self.detector_type == DetectorType.tpx1:
                logging.info(f"{self.autoreduce_dir}")
                file_path = Path(self.autoreduce_dir).parent.parent / file_path
            elif self.detector_type == DetectorType.tpx3:
                file_path = Path(self.raw_dir) / file_path
            if file_path is None:
                raise ValueError(f"No full path file found for run number {run_number}")
            
            return file_path

        else:
            raise ValueError(f"Unknown detector type: {self.detector_type}")

    def display_infos(self, input_full_path=None, spectra_file_found=True, correct_chips_alignment_flag=None):
        if input_full_path is None:
            return

        # retrieve the list of tiff files
        list_tiff = retrieve_list_of_tif(input_full_path)
        nbr_tiff = len(list_tiff)
        
        if correct_chips_alignment_flag is None:
            correct_chips_alignment_flag = False

        # load the first tiff file to get the shape and dtype
        data = Image.open(list_tiff[0])
        shape = data.size  # (width, height)
        dtype = np.array(data).dtype  # e.g. 'I;16' for 16-bit unsigned integer

        spectra_cell_color = "green" if spectra_file_found else "red"

        # present result in a table
        display(HTML(f"""
                        <h3>Information for run: {os.path.basename(input_full_path)}</h3>
                    <table border="3px solid black" style="border-collapse:collapse;">
                        <tr><th>Nbr TIFF</th><th>Images height</th><th>Images width</th><th>Data Type</th><th>Spectra File Found</th><th>Chips Alignment Correction</th></tr>
                        <tr><td>{nbr_tiff}</td><td>{shape[0]}</td><td>{shape[1]}</td><td>{dtype}</td><td style="color:{spectra_cell_color}">{spectra_file_found}</td><td>{correct_chips_alignment_flag}</td></tr>
                    </table>
        """))

    @staticmethod
    def _is_spectra_file_found_and_list(full_path):
        list_files = glob.glob(os.path.join(full_path, "*_Spectra.txt"))
        if len(list_files) == 0:
            return False, None
        
        return os.path.exists(list_files[0]), list_files[0]

    @staticmethod
    def _is_summary_json_file_found(full_path):
        summary_json_file = os.path.join(full_path, "summary.json")
        if not os.path.exists(summary_json_file):
            return False, None
        
        # load json file summary_json_file
        json_dict = load_json(summary_json_file)
        return True, json_dict

    def check_sample(self):
        """
        Check if the sample folder and runs are valid.
        """
        self.reset_sample_dicts()

        notebook_logging.info("Checking sample inputs...")
        display(HTML("Sample run numbers selected:"))

        if self.sample_run_numbers_widget.value.strip() != "":
            list_of_runs = extract_list_of_runs_from_string(self.sample_run_numbers_widget.value)
            notebook_logging.info(f"\t{list_of_runs = }")

            list_of_sample_full_path = []
            for _run in list_of_runs:
                try:
                    _full_path = self.extract_full_path(run_number=_run)
                    list_of_sample_full_path.append(_full_path)
                except TypeError as e:
                    notebook_logging.error(f"Error extracting full path for run number {_run}: {e}")
                    display(HTML(f"<span style='color:red'>Error extracting full path for run number {_run}: File not found!</span>"))
                    continue

            logging.info(f"\t{list_of_sample_full_path = }")
            for _file_full_path in list_of_sample_full_path:
               
                if os.path.exists(_file_full_path):
                    notebook_logging.info(f"\tSample run number {_file_full_path} - FOUND")
                    is_valid_run, report_dict = self.check_folder_is_valid(_file_full_path)
                    if is_valid_run:
                        nbr_tiff = report_dict["nbr_tiff"]
                        self.check_nbr_tiff[DataType.sample].append(nbr_tiff)
                        notebook_logging.info(f"\tSample run number {_file_full_path} - FOUND with {nbr_tiff} tif* files")
                        display(HTML(f"<span style='color:green'>{_file_full_path}</span> - OK"))
                        self.dict_sample[_file_full_path] = {}
                        self.dict_short_name_full_path["sample"][os.path.basename(_file_full_path)] = _file_full_path
                       
                        _is_spectra_file_found, spectra_file_name = NormalizationTof._is_spectra_file_found_and_list(_file_full_path)
                        if not _is_spectra_file_found:
                            self.spectra_file_found = False
                        else:
                            self.list_spectra_file_found.append(spectra_file_name)
 
                        _is_summary_json_file_found, _summary_dict = NormalizationTof._is_summary_json_file_found(_file_full_path)
                        notebook_logging.info(f"\tSummary JSON file found: {_is_summary_json_file_found}")
                        notebook_logging.info(f"\tSummary JSON content: {_summary_dict}")
                        correct_chips_alignment_flag = _summary_dict.get("chips_alignment_correction", None) if _is_summary_json_file_found else None
                        
                        self.display_infos(input_full_path=_file_full_path,
                                           spectra_file_found=self.spectra_file_found, 
                                           correct_chips_alignment_flag=correct_chips_alignment_flag)

                    else:
                        display(HTML(f"<span style='color:red'>{_file_full_path} - EMPTY!</span>"))

                else:
                    notebook_logging.info(f"\tSample run number {_file_full_path} - NOT FOUND")
                    display(HTML(f"<span style='color:red'>{_file_full_path} - NOT FOUND!</span>"))

        else:
            notebook_logging.info(f"Sample run numbers selected: {self.sample_run_numbers_selected}")
            if self.sample_run_numbers_selected is None:
                display(HTML(f"<span style='color:red'>No sample runs selected!</span>"))
                return
            
            for _run in self.sample_run_numbers_selected:
                _run = os.path.abspath(_run)
                if os.path.exists(_run):
                    notebook_logging.info(f"\tSample run number {_run} - FOUND")
                    # check here that the folder is not empty (contains tiff)
                    is_valid_run, report_dict = self.check_folder_is_valid(_run)
                    if is_valid_run:
                        nbr_tiff = report_dict["nbr_tiff"]
                        self.check_nbr_tiff[DataType.sample].append(nbr_tiff)
                        display(HTML(f"<span style='color:green'>{_run}</span> - OK"))
                        notebook_logging.info(f"\tfolder seems to be a valid folder containing {nbr_tiff} tif* files")
                        self.dict_sample[_run] = {}
                        self.dict_short_name_full_path["sample"][os.path.basename(_run)] = _run

                        _is_spectra_file_found, spectra_file_name = NormalizationTof._is_spectra_file_found_and_list(_run)
                        if not _is_spectra_file_found:
                            self.spectra_file_found = False
                        else:
                            self.list_spectra_file_found.append(spectra_file_name)
                        
                        _is_summary_json_file_found, _summary_dict = NormalizationTof._is_summary_json_file_found(_run)
                        notebook_logging.info(f"\tSummary JSON file found: {_is_summary_json_file_found}")
                        notebook_logging.info(f"\tSummary JSON content: {_summary_dict}")
                        correct_chips_alignment_flag = _summary_dict.get("chips_alignment_correction", None) if _is_summary_json_file_found else None
                       
                        self.display_infos(input_full_path=_run,
                                           spectra_file_found=self.spectra_file_found,
                                           correct_chips_alignment_flag=correct_chips_alignment_flag)
                    else:
                        display(HTML(f"<span style='color:red'>{_run} - EMPTY!</span>"))
                else:
                    display(HTML(f"<span style='color:red'>{_run} - NOT FOUND!</span> - ERROR!"))
                    notebook_logging.info(f"\tSample run number {_run} - NOT FOUND!")
            
            self.sample_run_numbers_selected = None

        if len(set(self.check_nbr_tiff[DataType.sample])) > 1:
            display(HTML(f"<span style='color:red'>Warning: Different number of TIFF files found in selected sample runs: {self.check_nbr_tiff[DataType.sample]}</span>"))
            notebook_logging.info(f"WARNING:Different number of TIFF files found in selected sample runs: {self.check_nbr_tiff[DataType.sample]}")

    def select_ob_folder(self):
        self.select_folder(instruction="Browse ob top folder", next_function=self.ob_folder_selected)

    def select_ob_run_numbers(self):

        if self.debug:
            ob_runs = DEBUG_DATA.ob_runs_selected
            ob_run_numbers_list = []
            for _run in ob_runs:
                _, number = _run.split("_")
                ob_run_numbers_list.append(number)
            str_ob_run_numbers = ", ".join(ob_run_numbers_list)

            output_folder = DEBUG_DATA.output_folder

        else:
            str_ob_run_numbers = ""

        ob_label = widgets.HTML(value="<b><font color='green'>List of ob run numbers (ex: 8705, 8707)</font></b>")

        self.ob_run_numbers_widget = widgets.Textarea(
            value=str_ob_run_numbers, placeholder="", layout=widgets.Layout(width="400px")
        )
        vertical_layout = widgets.VBox(
            [
                ob_label,
                self.ob_run_numbers_widget,
            ]
        )
        display(vertical_layout)

        display(HTML("<span style='font-size: 16px; color:red'>OR</span>"))

        self.select_folder(
            instruction="Browse ob run number folders",
            next_function=self.save_ob_run_numbers_selected,
            start_dir=self.ob_dir,
            multiple=True,
        )

    def check_ob(self):
        """
        Check if the ob folder and runs are valid.
        """
        notebook_logging.info("Checking ob inputs...")
        display(HTML("OB run numbers selected:"))

        self.reset_ob_dicts()

        if self.ob_run_numbers_widget.value.strip() != "":
            list_of_runs = extract_list_of_runs_from_string(self.ob_run_numbers_widget.value)
            notebook_logging.info(f"\t{list_of_runs = }")

            list_of_ob_full_path = []
            for _run in list_of_runs:
                try:
                    _full_path = self.extract_full_path(run_number=_run)
                    list_of_ob_full_path.append(_full_path)
                except TypeError as e:
                    notebook_logging.error(f"Error extracting full path for run number {_run}: {e}")
                    display(HTML(f"<span style='color:red'>Error extracting full path for run number {_run}: File not found!</span>"))
                    continue

            for _file_full_path in list_of_ob_full_path:
                if os.path.exists(_file_full_path):
                    notebook_logging.info(f"\tOB run number {_file_full_path} - FOUND")
                    is_valid_run, report_dict = self.check_folder_is_valid(_file_full_path)
                    if is_valid_run:
                        nbr_tiff = report_dict["nbr_tiff"]
                        self.check_nbr_tiff[DataType.ob].append(nbr_tiff)
                        notebook_logging.info(f"\tOB run number {_file_full_path} - FOUND with {nbr_tiff} tif* files")
                        display(HTML(f"<span style='color:green'>{_file_full_path}</span> - OK"))
                        self.dict_ob[_file_full_path] = {}
                        self.dict_short_name_full_path["ob"][os.path.basename(_file_full_path)] = _file_full_path
                    
                        _is_spectra_file_found, spectra_file_name = NormalizationTof._is_spectra_file_found_and_list(_file_full_path)
                        if not _is_spectra_file_found:
                            self.spectra_file_found = False
                        else:
                            self.list_spectra_file_found.append(spectra_file_name)
                      
                        _is_summary_json_file_found, _summary_dict = NormalizationTof._is_summary_json_file_found(_file_full_path)
                        notebook_logging.info(f"\tSummary JSON file found: {_is_summary_json_file_found}")
                        notebook_logging.info(f"\tSummary JSON content: {_summary_dict}")
                        correct_chips_alignment_flag = _summary_dict.get("chips_alignment_correction", None) if _is_summary_json_file_found else None
                      
                        self.display_infos(input_full_path=_file_full_path,
                                           spectra_file_found=self.spectra_file_found,
                                           correct_chips_alignment_flag=correct_chips_alignment_flag)

                    else:
                        display(HTML(f"<span style='color:red'>{_file_full_path} - EMPTY!</span>"))
                else:
                    notebook_logging.info(f"\tOB run number {_file_full_path} - NOT FOUND")
                    display(HTML(f"<span style='color:red'>{_file_full_path} - NOT FOUND!</span>"))

        else:
            notebook_logging.info(f"OB run numbers selected: {self.ob_run_numbers_selected}")
            if self.ob_run_numbers_selected is None:
                display(HTML(f"<span style='color:red'>No OB runs selected!</span>"))
                return
            
            for _run in self.ob_run_numbers_selected:
                _run = os.path.abspath(_run)
                if os.path.exists(_run):
                    notebook_logging.info(f"\tOB run number {_run} - FOUND")
                    # check here that the folder is not empty (contains tiff)
                    is_valid_run, report_dict = self.check_folder_is_valid(_run)
                    if is_valid_run:
                        nbr_tiff = report_dict["nbr_tiff"]
                        self.check_nbr_tiff[DataType.ob].append(nbr_tiff)
                        display(HTML(f"<span style='color:green'>{_run}</span> - OK"))
                        notebook_logging.info(f"\tfolder seems to be a valid folder containing {nbr_tiff} tif* files")
                        self.dict_short_name_full_path["ob"][os.path.basename(_run)] = _run
                        self.dict_ob[_run] = {}

                        _is_spectra_file_found, spectra_file_name = self._is_spectra_file_found_and_list(_run)
                        if not _is_spectra_file_found:
                            self.spectra_file_found = False
                        else:
                            self.list_spectra_file_found.append(spectra_file_name)
                        
                        _is_summary_json_file_found, _summary_dict = NormalizationTof._is_summary_json_file_found(_run)
                        notebook_logging.info(f"\tSummary JSON file found: {_is_summary_json_file_found}")
                        notebook_logging.info(f"\tSummary JSON content: {_summary_dict}")
                        correct_chips_alignment_flag = _summary_dict.get("chips_alignment_correction", None) if _is_summary_json_file_found else None
                                                
                        self.display_infos(input_full_path=_run,
                                           spectra_file_found=self.spectra_file_found,
                                           correct_chips_alignment_flag=correct_chips_alignment_flag)

                    else:
                        display(HTML(f"<span style='color:red'>{_run} - EMPTY!</span>"))
                else:
                    display(HTML(f"<span style='color:red'>{_run} - NOT FOUND!</span> - ERROR!"))
                    notebook_logging.info(f"\tOB run number {_run} - NOT FOUND!")
            self.ob_run_numbers_selected = None

        if len(set(self.check_nbr_tiff[DataType.ob])) > 1:
            display(HTML(f"<span style='color:red'>Warning: Different number of TIFF files found in selected OB runs: {self.check_nbr_tiff[DataType.ob]}</span>"))
            notebook_logging.info(f"WARNING: Different number of TIFF files found in selected OB runs: {self.check_nbr_tiff[DataType.ob]}")

        elif len(self.check_nbr_tiff[DataType.ob]) == 0:  # check_nbr_tiff[DataType.ob] is not empty
            display(HTML(f"<span style='color:red'>Empty OB run selected!</span>"))
            notebook_logging.info("WARNING: Not valid OB runs found!")
                        
        else:
            
            if len(self.check_nbr_tiff[DataType.sample]) > 0 and len(self.check_nbr_tiff[DataType.ob]) > 0:
                if self.check_nbr_tiff[DataType.ob][0] != self.check_nbr_tiff[DataType.sample][0]:
                    display(HTML(f"<span style='color:red'>Not valid OB runs found (different number of OB and sample TIFF files)!</span>"))
                    notebook_logging.info("WARNING: Not valid OB runs found!")

    def select_dc_run_numbers(self):
        self.select_folder(instruction="Browse dc top folder", next_function=self.dc_folder_selected)

    def select_dc_run_numbers(self):

        if self.debug:
            dc_runs = DEBUG_DATA.dc_runs_selected
            if dc_runs:
                dc_run_numbers_list = []
                for _run in dc_runs:
                    _, number = _run.split("_")
                    dc_run_numbers_list.append(number)
                str_dc_run_numbers = ", ".join(dc_run_numbers_list)
            else:
                str_dc_run_numbers = ""

        else:
            str_dc_run_numbers = ""

        dc_label = widgets.HTML(value="<b><font color='green'>List of dc run numbers (ex: 8705, 8707)</font></b>")

        self.dc_run_numbers_widget = widgets.Textarea(
            value=str_dc_run_numbers, placeholder="", layout=widgets.Layout(width="400px")
        )
        vertical_layout = widgets.VBox(
            [
                dc_label,
                self.dc_run_numbers_widget,
            ]
        )
        display(vertical_layout)

        display(HTML("<span style='font-size: 16px; color:red'>OR</span>"))

        self.select_folder(
            instruction="Browse dc run number folders",
            next_function=self.save_dc_run_numbers_selected,
            start_dir=self.dc_folder,
            multiple=True,
        )

    def check_dc(self):
        """
        Check if the dc folder and runs are valid.
        """
        notebook_logging.info("Checking dc inputs...")
        display(HTML("DC run numbers selected:"))

        self.reset_dc_dicts()

        if self.dc_run_numbers_widget.value.strip() != "":
            list_of_runs = extract_list_of_runs_from_string(self.dc_run_numbers_widget.value)
            notebook_logging.info(f"\t{list_of_runs = }")

            list_of_dc_full_path = []
            for _run in list_of_runs:
                _full_path = self.extract_full_path(run_number=_run)
                list_of_dc_full_path.append(_full_path)

            for _file_full_path in list_of_dc_full_path:
                if os.path.exists(_file_full_path):
                    notebook_logging.info(f"\tDC run number {_file_full_path} - FOUND")
                    is_valid_run, report_dict = self.check_folder_is_valid(_file_full_path)
                    if is_valid_run:
                        nbr_tiff = report_dict["nbr_tiff"]
                        self.check_nbr_tiff[DataType.dc].append(nbr_tiff)
                        notebook_logging.info(f"\tDC run number {_file_full_path} - FOUND with {nbr_tiff} tif* files")
                        display(HTML(f"<span style='color:green'>{_file_full_path}</span> - OK"))
                        self.dict_dc[_file_full_path] = {}
                        self.dict_short_name_full_path["dc"][os.path.basename(_file_full_path)] = _file_full_path

                        _is_summary_json_file_found, _summary_dict = NormalizationTof._is_summary_json_file_found(_file_full_path)
                        notebook_logging.info(f"\tSummary JSON file found: {_is_summary_json_file_found}")
                        notebook_logging.info(f"\tSummary JSON content: {_summary_dict}")
                        correct_chips_alignment_flag = _summary_dict.get("chips_alignment_correction", None) if _is_summary_json_file_found else None

                        self.display_infos(input_full_path=_file_full_path,
                                           correct_chips_alignment_flag=correct_chips_alignment_flag)

                    else:
                        display(HTML(f"<span style='color:red'>{_file_full_path} - EMPTY!</span>"))
                else:
                    notebook_logging.info(f"\tDC run number {_file_full_path} - NOT FOUND")
                    display(HTML(f"<span style='color:red'>{_file_full_path} - NOT FOUND!</span>"))

        else:
            notebook_logging.info(f"DC run numbers selected: {self.dc_run_numbers_selected}")
            if self.dc_run_numbers_selected is None:
                display(HTML(f"<span style='color:red'>No DC runs selected!</span>"))
                return

            for _run in self.dc_run_numbers_selected:
                _run = os.path.abspath(_run)
                if os.path.exists(_run):
                    notebook_logging.info(f"\tDC run number {_run} - FOUND")
                    # check here that the folder is not empty (contains tiff)
                    is_valid_run, report_dict = self.check_folder_is_valid(_run)
                    if is_valid_run:
                        nbr_tiff = report_dict["nbr_tiff"]
                        self.check_nbr_tiff[DataType.dc].append(nbr_tiff)
                        display(HTML(f"<span style='color:green'>{_run}</span> - OK"))
                        notebook_logging.info(f"\tfolder seems to be a valid folder containing {nbr_tiff} tif* files")
                        self.dict_short_name_full_path["dc"][os.path.basename(_run)] = _run
                        self.dict_dc[_run] = {}
                        
                        _is_summary_json_file_found, _summary_dict = NormalizationTof._is_summary_json_file_found(_run)
                        notebook_logging.info(f"\tSummary JSON file found: {_is_summary_json_file_found}")
                        notebook_logging.info(f"\tSummary JSON content: {_summary_dict}")
                        correct_chips_alignment_flag = _summary_dict.get("chips_alignment_correction", None) if _is_summary_json_file_found else None
                                               
                        self.display_infos(input_full_path=_run,
                                           correct_chips_alignment_flag=correct_chips_alignment_flag)
                    else:
                        display(HTML(f"<span style='color:red'>{_run} - EMPTY!</span>"))
                else:
                    display(HTML(f"<span style='color:red'>{_run} - NOT FOUND!</span> - ERROR!"))
                    notebook_logging.info(f"\tDC run number {_run} - NOT FOUND!")
            self.dc_run_numbers_selected = None

        if len(set(self.check_nbr_tiff[DataType.dc])) > 1:
            display(HTML(f"<span style='color:red'>Warning: Different number of TIFF files found in selected DC runs: {self.check_nbr_tiff[DataType.dc]}</span>"))
            notebook_logging.info(f"WARNING: Different number of TIFF files found in selected DC runs: {self.check_nbr_tiff[DataType.dc]}")

        else:
            if self.check_nbr_tiff[DataType.dc][0] != self.check_nbr_tiff[DataType.sample][0]:
                display(HTML(f"<span style='color:red'>No valid DC runs found (different number of DC and sample TIFF files)</span>"))
                notebook_logging.info("WARNING: No valid DC runs found!")

    def _select_measured_background_run_numbers(
        self,
        background_key: str,
        role: str,
        background_label: str,
    ):
        label = "sample" if role == "sample" else "OB"
        short_key = f"{background_key}_{role}_background"
        widget_name = f"{short_key}_run_numbers_widget"
        selected_attr = f"{short_key}_run_numbers_selected"
        setattr(
            self,
            widget_name,
            widgets.Textarea(value="", placeholder="", layout=widgets.Layout(width="400px")),
        )
        display(HTML(
            f"<b><font color='green'>List of {label} {background_label} background run numbers "
            "for Bragg edge / background-correction mode</font></b>"
        ))
        display(getattr(self, widget_name))
        display(HTML("<span style='font-size: 16px; color:red'>OR</span>"))
        self.select_folder(
            instruction=f"Browse {label} {background_label} background run folders",
            next_function=lambda folder_selected, attr=selected_attr: setattr(self, attr, folder_selected),
            start_dir=self.sample_folder if role == "sample" else self.ob_folder,
            multiple=True,
        )
        setattr(self, selected_attr, getattr(self, selected_attr, None))

    def _select_bragg_edge_cd_background_run_numbers(self, role: str):
        self._select_measured_background_run_numbers("bragg_edge_cd", role, "Cd-filter")

    def select_bragg_edge_cd_sample_background_run_numbers(self):
        self._select_bragg_edge_cd_background_run_numbers("sample")

    def select_bragg_edge_cd_ob_background_run_numbers(self):
        self._select_bragg_edge_cd_background_run_numbers("ob")

    def select_closed_slits_sample_background_run_numbers(self):
        self._select_measured_background_run_numbers("closed_slits", "sample", "closed-slits")

    def select_closed_slits_ob_background_run_numbers(self):
        self._select_measured_background_run_numbers("closed_slits", "ob", "closed-slits")

    def _check_measured_background(self, background_key: str, role: str, background_label: str):
        label = "sample" if role == "sample" else "OB"
        short_key = f"{background_key}_{role}_background"
        widget = getattr(self, f"{short_key}_run_numbers_widget", None)
        selected_attr = f"{short_key}_run_numbers_selected"
        dict_attr = f"dict_{short_key}"
        check_attr = f"{short_key}_check_nbr_tiff"
        self.reset_measured_background_dicts(background_key, role)
        display(HTML(f"{label} {background_label} background runs selected:"))

        full_paths = []
        if widget is not None and widget.value.strip() != "":
            for run_number in extract_list_of_runs_from_string(widget.value):
                full_paths.append(self.extract_full_path(run_number=run_number))
        else:
            selected = getattr(self, selected_attr, None)
            if selected is None:
                display(HTML(f"<span style='color:red'>No {label} {background_label} background runs selected!</span>"))
                return
            full_paths = [os.path.abspath(_run) for _run in selected]

        for full_path in full_paths:
            if not os.path.exists(full_path):
                display(HTML(f"<span style='color:red'>{full_path} - NOT FOUND!</span>"))
                continue
            is_valid_run, report_dict = self.check_folder_is_valid(full_path)
            if not is_valid_run:
                display(HTML(f"<span style='color:red'>{full_path} - EMPTY!</span>"))
                continue

            nbr_tiff = report_dict["nbr_tiff"]
            getattr(self, check_attr).append(nbr_tiff)
            getattr(self, dict_attr)[full_path] = {}
            self.dict_short_name_full_path[short_key][os.path.basename(full_path)] = full_path
            display(HTML(f"<span style='color:green'>{full_path}</span> - OK"))

            _is_summary_json_file_found, _summary_dict = NormalizationTof._is_summary_json_file_found(full_path)
            correct_chips_alignment_flag = (
                _summary_dict.get("chips_alignment_correction", None) if _is_summary_json_file_found else None
            )
            self.display_infos(
                input_full_path=full_path,
                correct_chips_alignment_flag=correct_chips_alignment_flag,
            )

        if len(getattr(self, check_attr)) == 0:
            display(HTML(f"<span style='color:red'>No valid {label} {background_label} background runs found!</span>"))
        elif len(set(getattr(self, check_attr))) > 1:
            display(HTML(
                f"<span style='color:red'>Warning: Different number of TIFF files found in selected "
                    f"{label} {background_label} background runs: {getattr(self, check_attr)}</span>"
            ))
        elif len(self.check_nbr_tiff[DataType.sample]) > 0 and getattr(self, check_attr)[0] != self.check_nbr_tiff[DataType.sample][0]:
            display(HTML(
                f"<span style='color:red'>Warning: {label} {background_label} background runs have a different "
                f"number of TIFF files than the sample run.</span>"
            ))
        setattr(self, selected_attr, None)

    def _check_bragg_edge_cd_background(self, role: str):
        self._check_measured_background("bragg_edge_cd", role, "Cd-filter")

    def check_bragg_edge_cd_sample_background(self):
        self._check_bragg_edge_cd_background("sample")

    def check_bragg_edge_cd_ob_background(self):
        self._check_bragg_edge_cd_background("ob")

    def check_closed_slits_sample_background(self):
        self._check_measured_background("closed_slits", "sample", "closed-slits")

    def check_closed_slits_ob_background(self):
        self._check_measured_background("closed_slits", "ob", "closed-slits")

    def _load_and_get_integrated_ob(self, full_path):
        """
        Load the integrated open beam data from the given OB run path.
        This function is a placeholder and should be implemented to load the actual data.
        """
        notebook_logging.info(f"Loading integrated OB data for {full_path}")
        display(HTML(f"<span style='color:blue'>Loading integrated OB data for {os.path.basename(full_path)} ... </span>"))
        # Here you would load the integrated OB data, for example using a specific library
        # For now, we will just return a dummy value
        if self.dict_ob[full_path].get("data") is None:
            notebook_logging.info("No data found for this OB run, loading it now...")
            # load the data from the OB run
            notebook_logging.info(f"\tFull path to OB run: {os.path.basename(full_path)}")
            list_tiff = retrieve_list_of_tif(full_path)
            notebook_logging.info(f"\tNumber of TIFF files found: {len(list_tiff)}")
            if len(list_tiff) == 0:
                display(HTML(f"<span style='color:red'>No TIFF files found in {full_path}!</span>"))
                notebook_logging.error(f"No TIFF files found in {full_path}!")
                return None
            data = load_data_using_multithreading(list_tiff, combine_tof=True)
            self.dict_ob[full_path]["data"] = data
        display(HTML(f"<span style='color:green'>Integrated OB data loaded for {os.path.basename(full_path)}!</span>"))

        return self.dict_ob[full_path]["data"]

    def preview_ob_runs(self):
        if self.dict_ob is None:
            display(HTML("<span style='color:red'>No OB runs selected!</span>"))
            return

        notebook_logging.info("Previewing OB runs")

        # list_ob_short_runs = [os.path.basename(_run) for _run in self.ob_run_numbers]
        # self.dict_ob_runs = {_short_name: _full_name for _short_name, _full_name in zip(list_ob_short_runs, self.ob_run_numbers)}
        # self.dict_ob_data = {_short_name: None for _short_name in list_ob_short_runs}

        list_ob_key = list(self.dict_ob.keys())
        list_ob_short_runs = [os.path.basename(_run) for _run in list_ob_key]
        if len(list_ob_key) == 1:
            notebook_logging.info("Only one OB run")
            full_path = list_ob_key[0]
            notebook_logging.info(f"\tFull path to OB run: {full_path}")
            integrated_ob = self._load_and_get_integrated_ob(full_path)
            if integrated_ob is None:
                display(
                    HTML(
                        f"<span style='color:red'>Failed to load integrated OB data for {list_ob_short_runs[0]}!</span>"
                    )
                )
                return
            # Calculate 2-98% percentile range to remove outliers
            vmin, vmax = np.percentile(integrated_ob, [2, 98])
            fig = px.imshow(integrated_ob, 
                           color_continuous_scale="viridis", 
                           aspect="equal",
                           title=f"Integrated OB run: {list_ob_short_runs[0]}",
                           labels={"color": "Intensity"},
                           zmin=vmin,
                           zmax=vmax)
            fig.update_layout(width=800, height=800)
            fig.show()

        else:
            notebook_logging.info(f"Multiple OB runs to display: {len(list_ob_key)}")

            def display_ob_run(short_name):
                """
                Display the integrated OB run data.
                """
                full_path = self.dict_short_name_full_path["ob"][short_name]
                integrated_ob = self._load_and_get_integrated_ob(full_path)
                if integrated_ob is None:
                    display(
                        HTML(
                            f"<span style='color:red'>Failed to load integrated OB data for {os.path.basename(full_path)}!</span>"
                        )
                    )
                    return
                # Calculate 2-98% percentile range to remove outliers
                vmin, vmax = np.percentile(integrated_ob, [2, 98])
                fig = px.imshow(integrated_ob, 
                               color_continuous_scale="viridis", 
                               aspect="auto",
                               title=f"Integrated OB run: {os.path.basename(full_path)}",
                               labels={"color": "Intensity"},
                               zmin=vmin,
                               zmax=vmax)
                fig.update_layout(width=800, height=500)
                fig.show()

            list_ob_key = list(self.dict_short_name_full_path["ob"].keys())
            _display = interactive(
                display_ob_run,
                short_name=widgets.Dropdown(
                    options=list_ob_key,
                    description="OB run:",
                    layout=widgets.Layout(width="100%"),
                    disabled=False,
                ),
            )
            display(_display)

    def checking_spectra_files(self):
        if not self.spectra_file_found:

            if len(self.list_spectra_file_found) > 0:
                display(HTML("<span style='color:orange; font-size:16px'>Some spectra files were NOT found but on the good side, some spectra files were found. The first one of those spectra file will be used for all!</span>"))
                notebook_logging.info("Some spectra files were NOT found but on the good side, some spectra files were found. The first one of those spectra file will be used for all!")

                spectra_file_to_use = self.list_spectra_file_found[0]
                pd_spectra = pd.read_csv(spectra_file_to_use, sep=",", header=0)
                self.spectra_array = np.array(pd_spectra["shutter_time"].values)

                notebook_logging.info(f"Using spectra file: {spectra_file_to_use} with {len(self.spectra_array)} TOF channels.")

            else:
                display(HTML("<span style='color:red; font-size:16px'>Error: No spectra files were found in the selected runs! We need to create the spectra file.</span>"))
                notebook_logging.error("Error: No spectra files were found in the selected runs! Manually creating the spectra file needed!")
                
                self.manually_create_spectra_array()
       
        else:
            display(HTML("<span style='color:green; font-size:16px'>All selected runs have the spectra file.</span>"))
            notebook_logging.info("All selected runs have the spectra file.")

    def manually_create_spectra_array(self):

        label = widgets.Label(f"Enter the TOF bins size (in nS)")
        self.tof_bin_size_widget = widgets.IntText(
            value=700,
            min=100,
            max=10000,
            layout=widgets.Layout(width="300px"),
        )
        display(label)
        display(self.tof_bin_size_widget)

        self.create_spectra_button = widgets.Button(
            description="Create spectra arrays",
            button_style="success",
            tooltip="Click to create spectra arrays",
            layout=widgets.Layout(width="300px"),
        )
        self.create_spectra_button.on_click(self.create_spectra_arrays_clicked)
        display(self.create_spectra_button)

    def create_spectra_arrays_clicked(self, b):
        tof_bin_size = self.tof_bin_size_widget.value
        notebook_logging.info(f"Creating spectra arrays with TOF bin size: {tof_bin_size} nS")

        # get the number of TOF channels from the first sample run
        first_sample_run = list(self.dict_sample.keys())[0]
        list_tiff = retrieve_list_of_tif(first_sample_run)
        if len(list_tiff) == 0: 
            display(HTML(f"<span style='color:red'>No TIFF files found in {first_sample_run}!</span>"))
            notebook_logging.error(f"No TIFF files found in {first_sample_run}!")
            return
        nbr_files = len(list_tiff)
        tof_bin_size_in_s = tof_bin_size * 1e-9  # convert nS to seconds

        # Manual fallback spectra represent the center of each uniform TOF bin.
        # This keeps lambda/energy conversion finite while avoiding the "near-zero"
        # first bin that produced pathological TPX3 x-axis values.
        spectra_array = (np.arange(nbr_files, dtype=np.float64) + 0.5) * tof_bin_size_in_s
        self.spectra_array = spectra_array

        display(HTML(f"<span style='color:blue; font-size:16px'>Created spectra arrays with TOF bin size: {tof_bin_size} nS!</span>"))
        notebook_logging.info(f"Created spectra arrays with TOF bin size: {tof_bin_size} nS and {len(spectra_array) = } ... Done!")

    def select_output_folder(self):

        if (self.spectra_file_found is False) and (self.spectra_array is None):
            display(HTML("<span style='color:red; font-size:16px'>You need to create the spectra arrays before selecting the output folder!</span>"))
            notebook_logging.error("You need to create the spectra arrays before selecting the output folder!")
            return

        if self.debug:
            self.output_folder_selected(DEBUG_DATA.output_folder)
        else:
            self.select_folder(
                ipts_folder=self.ipts_folder,
                instruction="Select output folder", 
                start_dir=self.output_dir, 
                next_function=self.output_folder_selected,
                newdir_toolbar_button=True,
            )

    def retrieve_nexus_file_path(self):
        """
        Retrieve the NeXus file paths for sample, OB and DC.
        
        This function assumes that the NeXus files are named in a specific format"""

        all_nexus_files_found = True
        notebook_logging.info("Retrieving NeXus file paths for sample, OB and DC runs...")

        notebook_logging.info("\tworking with sample runs:")
        for full_path in self.dict_sample.keys():
            if self.detector_type == DetectorType.tpx1_legacy:
                run_number = os.path.basename(full_path).split("_")[1]
            elif self.detector_type in [DetectorType.tpx1, DetectorType.tpx3]:
                file_name_split = os.path.basename(full_path).split("_")
                run_number = file_name_split[2]

            nexus_full_path = os.path.join(
                self.nexus_folder, f"{self.instrument.upper()}_{run_number}.nxs.h5"
            )
            if os.path.exists(nexus_full_path):
                notebook_logging.info(f"\tNeXus file found: {nexus_full_path}")
                self.dict_sample[full_path]["nexus"] = nexus_full_path
            else:
                notebook_logging.warning(f"\tNeXus file NOT found: {nexus_full_path}")
                all_nexus_files_found = False
                self.dict_sample[full_path]["nexus"] = None

        notebook_logging.info("\tworking with ob runs:")
        for full_path in self.dict_ob.keys():
            if self.detector_type == DetectorType.tpx1_legacy:
                run_number = os.path.basename(full_path).split("_")[1]
            elif self.detector_type in [DetectorType.tpx1, DetectorType.tpx3]:
                file_name_split = os.path.basename(full_path).split("_")
                run_number = file_name_split[2]

            nexus_full_path = os.path.join(
                self.nexus_folder, f"{self.instrument.upper()}_{run_number}.nxs.h5"
            )
            if os.path.exists(nexus_full_path):
                notebook_logging.info(f"\tNeXus file found: {nexus_full_path}")
                self.dict_ob[full_path]["nexus"] = nexus_full_path
            else:
                notebook_logging.warning(f"\tNeXus file NOT found: {nexus_full_path}")
                all_nexus_files_found = False
                self.dict_ob[full_path]["nexus"] = None

        notebook_logging.info("\tworking with dc runs:")
        for full_path in self.dict_dc.keys():
            if self.detector_type == DetectorType.tpx1_legacy:
                run_number = os.path.basename(full_path).split("_")[1]
            elif self.detector_type in [DetectorType.tpx1, DetectorType.tpx3]:
                file_name_split = os.path.basename(full_path).split("_")
                run_number = file_name_split[2]

            nexus_full_path = os.path.join(
                self.nexus_folder, f"{self.instrument.upper()}_{run_number}.nxs.h5"
            )
            if os.path.exists(nexus_full_path):
                notebook_logging.info(f"\tNeXus file found: {nexus_full_path}")
                self.dict_dc[full_path]["nexus"] = nexus_full_path
            else:
                notebook_logging.warning(f"\tNeXus file NOT found: {nexus_full_path}")
                all_nexus_files_found = False
                self.dict_dc[full_path]["nexus"] = None

        for label, background_dict in [
            ("sample Cd-filter measured background", self.dict_bragg_edge_cd_sample_background),
            ("OB Cd-filter measured background", self.dict_bragg_edge_cd_ob_background),
            ("sample closed-slits measured background", self.dict_closed_slits_sample_background),
            ("OB closed-slits measured background", self.dict_closed_slits_ob_background),
        ]:
            notebook_logging.info(f"\tworking with {label} runs:")
            for full_path in background_dict.keys():
                if self.detector_type == DetectorType.tpx1_legacy:
                    run_number = os.path.basename(full_path).split("_")[1]
                elif self.detector_type in [DetectorType.tpx1, DetectorType.tpx3]:
                    file_name_split = os.path.basename(full_path).split("_")
                    run_number = file_name_split[2]

                nexus_full_path = os.path.join(
                    self.nexus_folder, f"{self.instrument.upper()}_{run_number}.nxs.h5"
                )
                if os.path.exists(nexus_full_path):
                    notebook_logging.info(f"\tNeXus file found: {nexus_full_path}")
                    background_dict[full_path]["nexus"] = nexus_full_path
                else:
                    notebook_logging.warning(f"\tNeXus file NOT found: {nexus_full_path}")
                    all_nexus_files_found = False
                    background_dict[full_path]["nexus"] = None

        notebook_logging.info("Done retrieving NeXus file paths.")

        return all_nexus_files_found

    def _on_remove_container_flag_change(self, change):
        if change['new']:
           # enable the widgets
            disable_widgets = False
        else:
            # disable the widgets
            disable_widgets = True
        self.remove_container_options_flag.disabled = disable_widgets

    def _set_measured_background_inputs_enabled(self, background_key: str, enabled: bool):
        for role in ["sample", "ob"]:
            widget = getattr(self, f"{background_key}_{role}_background_run_numbers_widget", None)
            if widget is not None:
                widget.disabled = not enabled

        input_box = getattr(self, f"{background_key}_background_input_box", None)
        if input_box is not None:
            input_box.layout.display = None if enabled else "none"

    def _on_measured_background_flag_change(self, background_key: str, change):
        self._set_measured_background_inputs_enabled(background_key, bool(change["new"]))

    def _create_measured_background_run_input_box(self, background_key: str, background_label: str):
        sample_widget = widgets.Textarea(
            value="",
            placeholder="e.g. 19537, 19538",
            description="sample bg:",
            disabled=True,
            layout=widgets.Layout(width="520px", height="55px"),
        )
        ob_widget = widgets.Textarea(
            value="",
            placeholder="e.g. 19536",
            description="OB bg:",
            disabled=True,
            layout=widgets.Layout(width="520px", height="55px"),
        )
        setattr(self, f"{background_key}_sample_background_run_numbers_widget", sample_widget)
        setattr(self, f"{background_key}_ob_background_run_numbers_widget", ob_widget)

        input_box = widgets.VBox(
            [
                widgets.HTML(
                    f"<span style='font-size: 12px;'>Enter {background_label} background run numbers. "
                    "Use the same run-number syntax as sample/OB selection, e.g. 19537 or 19537-19538.</span>"
                ),
                sample_widget,
                ob_widget,
            ],
            layout=widgets.Layout(
                display="none",
                margin="0 0 8px 28px",
                border="1px solid #ddd",
                padding="6px",
                width="590px",
            ),
        )
        setattr(self, f"{background_key}_background_input_box", input_box)
        return input_box

    def _collect_enabled_measured_background_run_numbers(self):
        background_specs = [
            (
                "bragg_edge_cd",
                "Cd-filter",
                getattr(self, "bragg_edge_cd_background_flag", None),
            ),
            (
                "closed_slits",
                "closed-slits",
                getattr(self, "closed_slits_background_flag", None),
            ),
        ]
        for background_key, background_label, flag_widget in background_specs:
            if flag_widget is None or not flag_widget.value:
                continue
            for role in ["sample", "ob"]:
                short_key = f"{background_key}_{role}_background"
                dict_attr = f"dict_{short_key}"
                widget = getattr(self, f"{short_key}_run_numbers_widget", None)
                widget_has_value = widget is not None and widget.value.strip() != ""
                if widget_has_value or not getattr(self, dict_attr):
                    self._check_measured_background(background_key, role, background_label)

    def _on_rebin_mode_change(self, change):
        selected_mode = change["new"]
        self.rebin_delta_tof_us_ui.disabled = selected_mode != RebinMode.linear_tof
        self.rebin_delta_lambda_a_ui.disabled = selected_mode != RebinMode.linear_lambda
        self.rebin_delta_tof_over_tof_ui.disabled = selected_mode != RebinMode.log_tof
        self.rebin_delta_lambda_over_lambda_ui.disabled = selected_mode != RebinMode.log_lambda
        self.rebin_delta_lambda_squared_a2_ui.disabled = selected_mode != RebinMode.inverse_log_lambda
        self.rebin_full_bins_only_ui.disabled = selected_mode == RebinMode.none
        custom_schedule_disabled = selected_mode != RebinMode.custom_schedule
        self.rebin_custom_basis_ui.disabled = custom_schedule_disabled
        self.rebin_custom_scale_ui.disabled = custom_schedule_disabled
        self.rebin_custom_schedule_ui.disabled = custom_schedule_disabled
        if hasattr(self, "preview_rebin_boundaries_button"):
            self.preview_rebin_boundaries_button.disabled = selected_mode == RebinMode.none
        self._update_rebin_snap_to_native_grid_state()
        self._update_custom_schedule_help()

    def _fixed_width_rebin_mode_supports_native_snap(self):
        selected_mode = self.rebin_mode_ui.value
        if selected_mode in [RebinMode.linear_tof, RebinMode.linear_lambda]:
            return True
        if not hasattr(self, "rebin_custom_scale_ui") or not hasattr(self, "rebin_custom_basis_ui"):
            return False
        return (
            selected_mode == RebinMode.custom_schedule
            and self.rebin_custom_scale_ui.value == RebinCustomScale.linear
            and self.rebin_custom_basis_ui.value in [
                RebinCustomBasis.energy_tof,
                RebinCustomBasis.tof,
                RebinCustomBasis.lambda_,
            ]
        )

    def _update_rebin_snap_to_native_grid_state(self):
        if not hasattr(self, "rebin_snap_to_native_grid_ui"):
            return
        self.rebin_snap_to_native_grid_ui.disabled = not self._fixed_width_rebin_mode_supports_native_snap()

    def _get_custom_schedule_scale_options(self):
        if self.rebin_custom_basis_ui.value in [
            RebinCustomBasis.energy_tof,
            RebinCustomBasis.lambda_squared,
        ]:
            return [RebinCustomScale.linear]
        return [RebinCustomScale.linear, RebinCustomScale.log, RebinCustomScale.reverse_log]

    def _update_custom_schedule_scale_options(self):
        current_value = getattr(self.rebin_custom_scale_ui, "value", None)
        options = self._get_custom_schedule_scale_options()
        self.rebin_custom_scale_ui.options = options
        if current_value not in options:
            self.rebin_custom_scale_ui.value = options[0]

    def _update_custom_schedule_help(self):
        if not hasattr(self, "rebin_custom_schedule_help_ui"):
            return

        basis = self.rebin_custom_basis_ui.value
        scale = self.rebin_custom_scale_ui.value

        if basis == RebinCustomBasis.energy_tof:
            end_label = "upper_energy_edge_eV"
            step_label = "delta_tof_us"
            boundary_help = (
                "Energy boundaries must be listed in strictly increasing order. "
                "Each TOF width applies from the previous energy edge through the listed edge."
            )
        elif basis == RebinCustomBasis.tof:
            end_label = "end_tof_us"
            step_label = "delta_us" if scale == RebinCustomScale.linear else "dt/t"
            boundary_help = "TOF boundaries must be listed in strictly increasing order, from the minimum TOF upward."
        elif basis == RebinCustomBasis.lambda_:
            end_label = "end_lambda_A"
            step_label = "delta_A" if scale == RebinCustomScale.linear else "dl/l"
            boundary_help = "Lambda boundaries must be listed in strictly increasing order, from the minimum lambda upward."
        else:
            end_label = "end_lambda2_A2"
            step_label = "delta_(A^2)"
            boundary_help = "Lambda^2 boundaries must be listed in strictly increasing order, from the minimum lambda^2 upward."

        self.rebin_custom_schedule_help_ui.value = (
            "<span style='font-size: 12px;'>"
            "Custom schedule format: one segment per line as "
            f"<code>{end_label}, {step_label}</code>. "
            f"{boundary_help} "
            "Leave the final boundary blank to continue to the maximum value in the data."
            "</span>"
        )

    def _on_custom_schedule_basis_change(self, change):
        self._update_custom_schedule_scale_options()
        self._update_rebin_snap_to_native_grid_state()
        self._update_custom_schedule_help()

    def _on_custom_schedule_scale_change(self, change):
        self._update_rebin_snap_to_native_grid_state()
        self._update_custom_schedule_help()

    def _on_black_filter_background_flag_change(self, change):
        disabled = not change["new"]
        self.black_filter_background_shape_file_ui.disabled = disabled
        self.black_filter_background_anchor_energy_ui.disabled = disabled

    def _parse_custom_rebin_schedule(self) -> list[dict]:
        schedule_text = self.rebin_custom_schedule_ui.value
        if schedule_text is None:
            raise ValueError("Custom schedule text is empty.")

        parsed_schedule = []
        for line_index, raw_line in enumerate(schedule_text.splitlines(), start=1):
            line_without_comment = raw_line.split("#", 1)[0].strip()
            if not line_without_comment:
                continue

            parts = [part.strip() for part in line_without_comment.split(",")]
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid custom schedule line {line_index}: expected 'end_value, step'."
                )

            end_value_raw, step_raw = parts
            try:
                end_value = None if end_value_raw == "" else float(end_value_raw)
                step_value = float(step_raw)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid numeric value in custom schedule line {line_index}: {line_without_comment}"
                ) from exc

            parsed_schedule.append(
                {
                    "end_value": end_value,
                    "step": step_value,
                }
            )

        if not parsed_schedule:
            raise ValueError("Custom schedule requires at least one non-empty segment line.")

        return parsed_schedule

    @staticmethod
    def _load_spectra_file_for_preview(spectra_file: str = None) -> tuple[np.ndarray, np.ndarray]:
        if spectra_file is None:
            return None, None

        pd_spectra = pd.read_csv(spectra_file, sep=",", header=0)
        if "shutter_time" in pd_spectra.columns:
            tof_array = np.asarray(pd_spectra["shutter_time"].values, dtype=np.float64)
        else:
            tof_array = np.asarray(pd_spectra.iloc[:, 0].values, dtype=np.float64)

        counts_array = None
        if "counts" in pd_spectra.columns:
            counts_array = np.asarray(pd_spectra["counts"].values, dtype=np.float64)
        elif pd_spectra.shape[1] >= 2:
            counts_array = np.asarray(pd_spectra.iloc[:, 1].values, dtype=np.float64)

        return tof_array, counts_array

    @staticmethod
    def _load_tof_array_only_for_preview(spectra_file: str = None) -> np.ndarray:
        tof_array, _ = NormalizationTof._load_spectra_file_for_preview(spectra_file)
        return tof_array

    @staticmethod
    def _compute_total_counts_from_tiff_stack(full_path: str = None) -> np.ndarray:
        list_tiff = retrieve_list_of_tif(full_path)
        if len(list_tiff) == 0:
            return None

        total_counts = np.empty(len(list_tiff), dtype=np.float64)
        for index, tiff_file in enumerate(list_tiff):
            total_counts[index] = np.sum(np.asarray(Image.open(tiff_file), dtype=np.float64))
        return total_counts

    @staticmethod
    def _get_proton_charge_from_nexus_for_preview(nexus_full_path: str = None) -> float:
        if nexus_full_path is None or not os.path.exists(nexus_full_path):
            return None

        try:
            with h5py.File(nexus_full_path, "r") as hdf5_data:
                return float(hdf5_data["entry"]["proton_charge"][0] / 1e12)
        except (KeyError, OSError, TypeError, ValueError):
            return None

    def _get_preview_run_axis(self, dict_runs=None) -> np.ndarray:
        if not dict_runs:
            raise ValueError("No run selected for preview.")

        first_run = list(dict_runs.keys())[0]
        spectra_file_found, spectra_file_name = NormalizationTof._is_spectra_file_found_and_list(first_run)
        if spectra_file_found:
            tof_array = self._load_tof_array_only_for_preview(spectra_file_name)
        else:
            if self.spectra_array is None:
                raise ValueError(
                    "No spectra axis is available yet. Load a spectra file or create the manual spectra array first."
                )
            tof_array = np.asarray(self.spectra_array, dtype=np.float64)

        if tof_array is None or len(tof_array) == 0:
            raise ValueError("Preview TOF axis is empty.")
        return np.asarray(tof_array, dtype=np.float64)

    def _get_preview_signal_for_run(self, dict_runs=None, run_role: str = "sample") -> tuple[np.ndarray, np.ndarray, float, str]:
        if not dict_runs:
            raise ValueError(f"Select at least one {run_role} run before previewing rebin boundaries.")

        first_run = list(dict_runs.keys())[0]
        spectra_file_found, spectra_file_name = NormalizationTof._is_spectra_file_found_and_list(first_run)

        if spectra_file_found:
            tof_array, counts_array = self._load_spectra_file_for_preview(spectra_file_name)
            source_label = f"first {run_role} spectra counts: {os.path.basename(first_run)}"
        else:
            if self.spectra_array is None:
                raise ValueError(
                    "No spectra axis is available yet. Load a spectra file or create the manual spectra array first."
                )
            tof_array = np.asarray(self.spectra_array, dtype=np.float64)
            counts_array = self._compute_total_counts_from_tiff_stack(first_run)
            source_label = (
                f"first {run_role} full-image counts from TIFF stack: {os.path.basename(first_run)}"
            )

        if tof_array is None or counts_array is None:
            raise ValueError(f"Unable to build a preview signal from the first {run_role} run.")

        min_length = min(len(tof_array), len(counts_array))
        if min_length == 0:
            raise ValueError("Preview signal is empty.")

        proton_charge = self._get_proton_charge_from_nexus_for_preview(
            dict_runs.get(first_run, {}).get("nexus")
        )

        return (
            np.asarray(tof_array[:min_length], dtype=np.float64),
            np.asarray(counts_array[:min_length], dtype=np.float64),
            proton_charge,
            source_label,
        )

    def _get_preview_detector_delay_us(self) -> float:
        if self.instrument == "SNAP":
            return self.detector_offset_us.value

        if not self.dict_sample:
            return 0.0

        first_sample_run = list(self.dict_sample.keys())[0]
        nexus_full_path = self.dict_sample.get(first_sample_run, {}).get("nexus")
        if nexus_full_path and os.path.exists(nexus_full_path):
            detector_delay_us = get_detector_offset_from_nexus(nexus_full_path)
            if detector_delay_us is not None:
                return float(detector_delay_us)

        return 0.0

    def _build_current_rebin_groups_for_preview(self, max_frames: int = None):
        rebin_mode = self.rebin_mode_ui.value
        if rebin_mode == RebinMode.none:
            return None, None, None, None, None

        tof_array = self._get_preview_run_axis(self.dict_sample)
        if max_frames is not None:
            tof_array = np.asarray(tof_array[:max_frames], dtype=np.float64)
        detector_delay_us = self._get_preview_detector_delay_us()
        _, lambda_array, energy_array = calculate_time_lambda_energy_arrays(
            time_spectra=tof_array,
            distance_source_detector_m=self.distance_source_detector.value,
            detector_delay_us=detector_delay_us,
        )

        rebin_custom_schedule = None
        if rebin_mode == RebinMode.custom_schedule:
            rebin_custom_schedule = self._parse_custom_rebin_schedule()

        bin_groups, bin_edges = build_rebin_bin_groups(
            rebin_mode=rebin_mode,
            tof_array=tof_array,
            lambda_array=lambda_array,
            energy_array=energy_array,
            rebin_delta_tof_us=self.rebin_delta_tof_us_ui.value if rebin_mode == RebinMode.linear_tof else None,
            rebin_delta_lambda_a=self.rebin_delta_lambda_a_ui.value if rebin_mode == RebinMode.linear_lambda else None,
            rebin_delta_tof_over_tof=(
                self.rebin_delta_tof_over_tof_ui.value if rebin_mode == RebinMode.log_tof else None
            ),
            rebin_delta_lambda_over_lambda=(
                self.rebin_delta_lambda_over_lambda_ui.value if rebin_mode == RebinMode.log_lambda else None
            ),
            rebin_delta_lambda_squared_a2=(
                self.rebin_delta_lambda_squared_a2_ui.value if rebin_mode == RebinMode.inverse_log_lambda else None
            ),
            rebin_custom_basis=self.rebin_custom_basis_ui.value if rebin_mode == RebinMode.custom_schedule else None,
            rebin_custom_scale=self.rebin_custom_scale_ui.value if rebin_mode == RebinMode.custom_schedule else None,
            rebin_custom_schedule=rebin_custom_schedule,
            rebin_full_bins_only=self.rebin_full_bins_only_ui.value,
            rebin_snap_to_native_grid=(
                self.rebin_snap_to_native_grid_ui.value
                if (
                    hasattr(self, "rebin_snap_to_native_grid_ui")
                    and self._fixed_width_rebin_mode_supports_native_snap()
                )
                else False
            ),
        )
        return tof_array, lambda_array, energy_array, bin_groups, bin_edges

    def _update_rebin_bin_count_display(self, _change=None):
        if not hasattr(self, "rebin_bin_count_ui"):
            return

        try:
            rebin_mode = self.rebin_mode_ui.value
            if rebin_mode == RebinMode.none:
                original_tof_array = self._get_preview_run_axis(self.dict_sample)
                self.rebin_bin_count_ui.value = (
                    f"<span style='font-size: 12px; color: #2b6;'>"
                    f"Active bins with current settings: {len(original_tof_array)} "
                    f"(no rebin, original frame count)</span>"
                )
                return

            _, _, _, bin_groups, _ = self._build_current_rebin_groups_for_preview()
            active_bin_count = np.sum([len(_group) > 0 for _group in bin_groups])
            source_frame_counts = np.asarray([len(_group) for _group in bin_groups if len(_group) > 0], dtype=int)
            if len(source_frame_counts) > 0:
                unique_counts, count_counts = np.unique(source_frame_counts, return_counts=True)
                source_frame_summary = ", ".join(
                    f"{int(_count)} frames: {int(_n)} bins"
                    for _count, _n in zip(unique_counts, count_counts)
                )
            else:
                source_frame_summary = "none"
            self.rebin_bin_count_ui.value = (
                f"<span style='font-size: 12px; color: #2b6;'>"
                f"Active bins with current settings: {int(active_bin_count)} "
                f"({source_frame_summary})</span>"
            )
        except Exception as exc:
            self.rebin_bin_count_ui.value = (
                f"<span style='font-size: 12px; color: #b33;'>"
                f"Active bin count unavailable: {exc}</span>"
            )

    def _observe_rebin_preview_controls(self):
        widgets_to_observe = [
            self.rebin_mode_ui,
            self.rebin_delta_tof_us_ui,
            self.rebin_delta_lambda_a_ui,
            self.rebin_delta_tof_over_tof_ui,
            self.rebin_delta_lambda_over_lambda_ui,
            self.rebin_delta_lambda_squared_a2_ui,
            self.rebin_full_bins_only_ui,
            self.rebin_snap_to_native_grid_ui,
            self.rebin_custom_basis_ui,
            self.rebin_custom_scale_ui,
            self.rebin_custom_schedule_ui,
        ]
        if hasattr(self, "distance_source_detector"):
            widgets_to_observe.append(self.distance_source_detector)
        for _widget in widgets_to_observe:
            _widget.observe(self._update_rebin_bin_count_display, names="value")

    def _get_rebin_preview_axis(self, tof_array=None, lambda_array=None, rebin_mode: str = None):
        if rebin_mode in [RebinMode.linear_tof, RebinMode.log_tof]:
            return tof_array * 1e6, "TOF (micros)"
        if rebin_mode in [RebinMode.linear_lambda, RebinMode.log_lambda]:
            return lambda_array, "Lambda (Angstroms)"
        if rebin_mode == RebinMode.inverse_log_lambda:
            return np.square(lambda_array), "Lambda^2 (Angstroms^2)"
        if rebin_mode == RebinMode.custom_schedule:
            if self.rebin_custom_basis_ui.value in [
                RebinCustomBasis.energy_tof,
                RebinCustomBasis.tof,
            ]:
                return tof_array * 1e6, "TOF (micros)"
            if self.rebin_custom_basis_ui.value == RebinCustomBasis.lambda_:
                return lambda_array, "Lambda (Angstroms)"
            return np.square(lambda_array), "Lambda^2 (Angstroms^2)"
        return tof_array * 1e6, "TOF (micros)"

    def _get_rebin_preview_edge_display_array(self, bin_edges=None, rebin_mode: str = None) -> np.ndarray:
        if bin_edges is None:
            return None
        if rebin_mode in [RebinMode.linear_tof, RebinMode.log_tof]:
            return np.asarray(bin_edges, dtype=np.float64) * 1e6
        if (
            rebin_mode == RebinMode.custom_schedule
            and self.rebin_custom_basis_ui.value in [
                RebinCustomBasis.energy_tof,
                RebinCustomBasis.tof,
            ]
        ):
            return np.asarray(bin_edges, dtype=np.float64) * 1e6
        return np.asarray(bin_edges, dtype=np.float64)

    @staticmethod
    def _format_preview_energy_tick_label(energy_value_eV: float) -> str:
        if not np.isfinite(energy_value_eV):
            return ""
        if energy_value_eV >= 1:
            return f"{energy_value_eV:.3g}"
        if energy_value_eV >= 0.1:
            return f"{energy_value_eV:.3f}"
        if energy_value_eV >= 0.01:
            return f"{energy_value_eV:.4f}"
        return f"{energy_value_eV:.2e}"

    def _get_preview_energy_axis_ticks(
        self,
        display_x_array: np.ndarray = None,
        energy_array: np.ndarray = None,
        max_ticks: int = 7,
    ) -> tuple[np.ndarray, list[str]]:
        if display_x_array is None or energy_array is None:
            return None, None

        display_x_array = np.asarray(display_x_array, dtype=np.float64)
        energy_array = np.asarray(energy_array, dtype=np.float64)
        finite_mask = np.isfinite(display_x_array) & np.isfinite(energy_array)
        if not np.any(finite_mask):
            return None, None

        valid_indices = np.flatnonzero(finite_mask)
        n_ticks = min(max_ticks, len(valid_indices))
        tick_indices = np.unique(np.linspace(0, len(valid_indices) - 1, num=n_ticks, dtype=int))
        selected_indices = valid_indices[tick_indices]
        tickvals = display_x_array[selected_indices]
        ticktext = [
            self._format_preview_energy_tick_label(_energy_value)
            for _energy_value in energy_array[selected_indices]
        ]
        return tickvals, ticktext

    def preview_rebin_boundaries_clicked(self, _button):
        with self.rebin_preview_output_ui:
            clear_output(wait=True)
            try:
                rebin_mode = self.rebin_mode_ui.value
                if rebin_mode == RebinMode.none:
                    display(HTML("<span style='color:blue'>Select a rebin mode to preview its boundaries.</span>"))
                    return

                sample_tof_array, sample_counts_array, sample_proton_charge, sample_source_label = (
                    self._get_preview_signal_for_run(self.dict_sample, run_role="sample")
                )
                ob_tof_array, ob_counts_array, ob_proton_charge, ob_source_label = (
                    self._get_preview_signal_for_run(self.dict_ob, run_role="OB")
                )
                min_length = min(len(sample_tof_array), len(ob_tof_array), len(sample_counts_array), len(ob_counts_array))
                if min_length == 0:
                    raise ValueError("Unable to build a preview transmission signal.")

                tof_array = np.asarray(sample_tof_array[:min_length], dtype=np.float64)
                sample_counts_array = np.asarray(sample_counts_array[:min_length], dtype=np.float64)
                ob_counts_array = np.asarray(ob_counts_array[:min_length], dtype=np.float64)

                if self.proton_charge_flag.value and (sample_proton_charge is not None) and (ob_proton_charge is not None):
                    sample_preview_signal = sample_counts_array / sample_proton_charge
                    ob_preview_signal = ob_counts_array / ob_proton_charge
                    preview_scaling_label = "proton-charge scaled sample/OB transmission preview"
                else:
                    sample_preview_signal = sample_counts_array
                    ob_preview_signal = ob_counts_array
                    preview_scaling_label = "raw sample/OB transmission preview"

                with np.errstate(divide="ignore", invalid="ignore"):
                    preview_transmission = np.divide(
                        sample_preview_signal,
                        ob_preview_signal,
                        out=np.full_like(sample_preview_signal, np.nan, dtype=np.float64),
                        where=np.abs(ob_preview_signal) > 0,
                    )

                _, lambda_array, energy_array, bin_groups, bin_edges = self._build_current_rebin_groups_for_preview(
                    max_frames=min_length
                )

                display_x_array, x_axis_label = self._get_rebin_preview_axis(
                    tof_array=tof_array,
                    lambda_array=lambda_array,
                    rebin_mode=rebin_mode,
                )
                energy_tick_values, energy_tick_labels = self._get_preview_energy_axis_ticks(
                    display_x_array=display_x_array,
                    energy_array=energy_array,
                )
                display_edge_array = self._get_rebin_preview_edge_display_array(
                    bin_edges=bin_edges,
                    rebin_mode=rebin_mode,
                )

                active_bin_indices = [index for index, group in enumerate(bin_groups) if len(group) > 0]
                if not active_bin_indices:
                    raise ValueError("No active bins were created with the current rebin settings.")

                import plotly.graph_objects as go

                figure = go.Figure()
                figure.add_trace(
                    go.Scatter(
                        x=display_x_array,
                        y=np.asarray(preview_transmission, dtype=np.float64),
                        mode="lines",
                        line=dict(color="black", width=1.2),
                        name="original transmission",
                        customdata=np.asarray(energy_array, dtype=np.float64).reshape(-1, 1),
                        hovertemplate=(
                            f"{x_axis_label}: %{{x}}<br>"
                            "Energy (eV): %{customdata[0]}<br>"
                            "transmission: %{y}<extra></extra>"
                        ),
                    )
                )

                rebinned_x_array = []
                rebinned_energy_array = []
                rebinned_transmission_array = []
                rebinned_sample_total_array = []
                rebinned_ob_total_array = []
                rebinned_n_frames_array = []
                for active_bin_index in active_bin_indices:
                    frame_group = np.asarray(bin_groups[active_bin_index], dtype=int)
                    rebinned_x_array.append(np.mean(display_x_array[frame_group]))
                    rebinned_energy_array.append(np.mean(energy_array[frame_group]))
                    rebinned_sample_total = np.sum(sample_preview_signal[frame_group], dtype=np.float64)
                    rebinned_ob_total = np.sum(ob_preview_signal[frame_group], dtype=np.float64)
                    if np.abs(rebinned_ob_total) > 0:
                        rebinned_transmission = rebinned_sample_total / rebinned_ob_total
                    else:
                        rebinned_transmission = np.nan
                    rebinned_transmission_array.append(rebinned_transmission)
                    rebinned_sample_total_array.append(rebinned_sample_total)
                    rebinned_ob_total_array.append(rebinned_ob_total)
                    rebinned_n_frames_array.append(len(frame_group))

                figure.add_trace(
                    go.Scatter(
                        x=np.asarray(rebinned_x_array, dtype=np.float64),
                        y=np.asarray(rebinned_transmission_array, dtype=np.float64),
                        mode="lines+markers",
                        line=dict(color="darkorange", width=2.0),
                        marker=dict(color="darkorange", size=5),
                        name="rebinned transmission",
                        customdata=np.column_stack(
                            [
                                np.asarray(rebinned_energy_array, dtype=np.float64),
                                np.asarray(rebinned_sample_total_array, dtype=np.float64),
                                np.asarray(rebinned_ob_total_array, dtype=np.float64),
                                np.asarray(rebinned_n_frames_array, dtype=np.int64),
                            ]
                        ),
                        hovertemplate=(
                            f"{x_axis_label}: %{{x}}<br>"
                            "Energy (eV): %{customdata[0]}<br>"
                            "rebinned transmission: %{y}<br>"
                            "sample sum in bin: %{customdata[1]}<br>"
                            "OB sum in bin: %{customdata[2]}<br>"
                            "frames in bin: %{customdata[3]}<extra></extra>"
                        ),
                    )
                )

                if energy_tick_values is not None and energy_tick_labels is not None:
                    figure.add_trace(
                        go.Scatter(
                            x=np.asarray(energy_tick_values, dtype=np.float64),
                            y=np.zeros(len(energy_tick_values), dtype=np.float64),
                            mode="markers",
                            marker=dict(opacity=0, size=1),
                            hoverinfo="skip",
                            showlegend=False,
                            xaxis="x2",
                            yaxis="y",
                        )
                    )

                shapes = []
                if len(active_bin_indices) <= 80:
                    for display_bin_index, active_bin_index in enumerate(active_bin_indices):
                        x0 = float(display_edge_array[active_bin_index])
                        x1 = float(display_edge_array[active_bin_index + 1])
                        if display_bin_index % 2 == 0:
                            shapes.append(
                                dict(
                                    type="rect",
                                    xref="x",
                                    yref="paper",
                                    x0=x0,
                                    x1=x1,
                                    y0=0,
                                    y1=1,
                                    fillcolor="rgba(65, 105, 225, 0.08)",
                                    line=dict(width=0),
                                    layer="below",
                                )
                            )

                boundary_values = []
                for active_bin_index in active_bin_indices:
                    boundary_values.append(display_edge_array[active_bin_index])
                    boundary_values.append(display_edge_array[active_bin_index + 1])
                boundary_values = np.unique(np.asarray(boundary_values, dtype=np.float64))

                for boundary_value in boundary_values:
                    shapes.append(
                        dict(
                            type="line",
                            xref="x",
                            yref="paper",
                            x0=float(boundary_value),
                            x1=float(boundary_value),
                            y0=0,
                            y1=1,
                            line=dict(color="crimson", width=1.2),
                            layer="above",
                        )
                    )

                figure.update_layout(
                    title=dict(
                        text=(
                            f"Preview of {rebin_mode} boundaries over simplified transmission<br>"
                            f"<sup>{sample_source_label} vs {ob_source_label} | {preview_scaling_label} | "
                            f"active bins: {len(active_bin_indices)}</sup>"
                        ),
                        x=0.02,
                        xanchor="left",
                        y=0.98,
                        yanchor="top",
                        pad=dict(t=0, b=18),
                    ),
                    xaxis_title=x_axis_label,
                    yaxis_title="Transmission (a.u.)",
                    yaxis_type="linear",
                    width=1000,
                    height=500,
                    margin=dict(t=205),
                    hovermode="x unified",
                    showlegend=True,
                    shapes=shapes,
                    plot_bgcolor="white",
                    annotations=[
                        dict(
                            text="Energy (eV)",
                            x=0.5,
                            xref="paper",
                            y=1.22,
                            yref="paper",
                            showarrow=False,
                            font=dict(size=14, color="black"),
                        )
                    ],
                    xaxis2=dict(
                        title="",
                        overlaying="x",
                        side="top",
                        anchor="y",
                        tickmode="array",
                        tickvals=energy_tick_values,
                        ticktext=energy_tick_labels,
                        showgrid=False,
                        tickangle=0,
                        tickfont=dict(size=11),
                        showline=True,
                        linecolor="black",
                        linewidth=1,
                        ticks="outside",
                        ticklen=6,
                        zeroline=False,
                    ),
                )
                figure.update_xaxes(showgrid=False)
                figure.update_yaxes(showgrid=True, gridcolor="rgba(0, 0, 0, 0.10)")
                figure.show()

            except Exception as exc:
                display(HTML(f"<span style='color:red'>Unable to preview rebin boundaries: {exc}</span>"))

    def settings(self):

        # check here that the user selected a folder for output
        if self.output_folder is None:
            display(HTML("<span style='color:red; font-size: 16px;'>You forgot to select an output folder!</span>"))
            return

        all_nexus_found = self.retrieve_nexus_file_path()
        notebook_logging.info(f"All NeXus files found: {all_nexus_found}")

        tpx3_disabled_flag = True if self.detector_type == DetectorType.tpx3 else False

        # regular normalization
        display(HTML("<span style='font-size: 16px; color:red'>Normalization pixel by pixel</span>"))
        display(widgets.Checkbox(description="Normalization of images pixel by pixel", 
                                 value=True, 
                                 disabled=True,
                                 layout=widgets.Layout(width="600px")))
        display(HTML("<hr>"))

        # normalization of full spectrum of ROI
        display(HTML("<span style='font-size: 16px; color:red'>Normalization of full spectrum of ROI</span>"))
        display(HTML("<span style='font-size: 12px;'>If checked, normalization will be done as follows. After selecting a region of interest (ROI), for each image, the total counts of that region of the sample will be divided by the total" \
        " counts of the same region of the OB. This will produce a profile of this normalization value for each image.</span>"))
        self.full_spectrum_roi_flag = widgets.Checkbox(description="Work on full spectrum of ROI", value=True)
        display(self.full_spectrum_roi_flag)
        display(HTML("<hr>"))

        # how to combine sample runs if more than 1 sample provided
        self.combine_sample_runs_flag = widgets.Checkbox(
            description="Combine sample runs (all sample will produce one normalization output)", 
            value=False, 
            disabled=False,
            layout=widgets.Layout(width="600px"),
        )
        if len(self.dict_sample) > 1:
            display(HTML("<span style='font-size: 16px; color:red'>How to treat the sample runs</span>"))
            display(self.combine_sample_runs_flag)
            display(HTML("<hr>"))

        # remove container option
        display(HTML("<span style='font-size: 16px; color:red'>Remove container</span>"))
        self.remove_container_flag = widgets.Checkbox(description="Do you want to remove container signal?", 
                                                      value=False,
                                                      layout=widgets.Layout(width="600px"))
        self.remove_container_flag.observe(self._on_remove_container_flag_change, names='value')
        display(self.remove_container_flag)

        white_space = widgets.Label("\t\t",
                                    layout=widgets.Layout(width="150px"))
        self.remove_container_options_flag = widgets.RadioButtons(
            options=[
                "Select a ROI of the sample containing only the container signal",
                "Use previously saved ROI containing only the container signal",
            ],
            description="",
            disabled=True,
            layout=widgets.Layout(width="500px"),
        )
        
        hori_layout = widgets.HBox([white_space, 
                                    self.remove_container_options_flag],
                                  layout=widgets.Layout(align_items="center",
                                                         width="100%"))
        display(hori_layout)

        display(HTML("<hr>"))

        display(HTML("<span style='font-size: 16px; color:red'>Rebin TOF axis before normalization</span>"))
        display(HTML("<span style='font-size: 12px;'>Use this only when you want the normalization to be performed on rebinned sample and OB counts rather than on the original frame-by-frame stack.</span>"))

        self.rebin_mode_ui = widgets.Dropdown(
            options=[
                RebinMode.none,
                RebinMode.linear_tof,
                RebinMode.linear_lambda,
                RebinMode.log_tof,
                RebinMode.log_lambda,
                RebinMode.inverse_log_lambda,
                RebinMode.custom_schedule,
            ],
            value=RebinMode.none,
            description="Mode:",
            layout=widgets.Layout(width="420px"),
        )
        self.rebin_mode_ui.observe(self._on_rebin_mode_change, names="value")
        display(self.rebin_mode_ui)

        self.rebin_delta_tof_us_ui = widgets.BoundedFloatText(
            value=30.0,
            min=0.001,
            max=1_000_000.0,
            step=1.0,
            description="delta_us:",
            disabled=True,
            layout=widgets.Layout(width="220px"),
        )
        self.rebin_delta_lambda_a_ui = widgets.BoundedFloatText(
            value=0.01,
            min=1e-6,
            max=100.0,
            step=0.001,
            description="delta_A:",
            disabled=True,
            layout=widgets.Layout(width="220px"),
        )
        self.rebin_delta_tof_over_tof_ui = widgets.BoundedFloatText(
            value=0.01,
            min=1e-6,
            max=10.0,
            step=0.001,
            description="dt/t:",
            disabled=True,
            layout=widgets.Layout(width="220px"),
        )
        self.rebin_delta_lambda_over_lambda_ui = widgets.BoundedFloatText(
            value=0.01,
            min=1e-6,
            max=10.0,
            step=0.001,
            description="dl/l:",
            disabled=True,
            layout=widgets.Layout(width="220px"),
        )
        self.rebin_delta_lambda_squared_a2_ui = widgets.BoundedFloatText(
            value=0.01,
            min=1e-8,
            max=1_000.0,
            step=0.001,
            description="d(l^2):",
            disabled=True,
            layout=widgets.Layout(width="220px"),
        )
        display(
            widgets.HBox(
                [
                    self.rebin_delta_tof_us_ui,
                    self.rebin_delta_lambda_a_ui,
                    self.rebin_delta_tof_over_tof_ui,
                    self.rebin_delta_lambda_over_lambda_ui,
                    self.rebin_delta_lambda_squared_a2_ui,
                ],
                layout=widgets.Layout(align_items="center", width="100%"),
            )
        )
        self.rebin_full_bins_only_ui = widgets.Checkbox(
            description="Full bins only",
            value=True,
            disabled=True,
            layout=widgets.Layout(width="220px"),
        )
        display(
            HTML(
                "<span style='font-size: 12px;'>"
                "When enabled, partial bins at segment or range boundaries are dropped instead of being kept as "
                "odd-sized transition bins."
                "</span>"
            )
        )
        display(self.rebin_full_bins_only_ui)

        self.rebin_snap_to_native_grid_ui = widgets.Checkbox(
            description="Snap fixed-width bins to native TOF grid",
            value=True,
            disabled=True,
            layout=widgets.Layout(width="420px"),
        )
        display(
            HTML(
                "<span style='font-size: 12px;'>"
                "For fixed-width TOF/lambda bins, round the requested bin width to an integer number of native "
                "source frames. This avoids alternating 5-frame/6-frame bins when, for example, 30 us is not an "
                "integer multiple of the native TPX1 TOF spacing."
                "</span>"
            )
        )
        display(self.rebin_snap_to_native_grid_ui)

        self.rebin_custom_basis_ui = widgets.Dropdown(
            options=[
                ("Energy edges / TOF widths", RebinCustomBasis.energy_tof),
                ("TOF edges / TOF widths (legacy)", RebinCustomBasis.tof),
                ("Lambda edges / lambda widths (legacy)", RebinCustomBasis.lambda_),
                ("Lambda^2 edges / lambda^2 widths (legacy)", RebinCustomBasis.lambda_squared),
            ],
            value=RebinCustomBasis.energy_tof,
            description="basis:",
            disabled=True,
            layout=widgets.Layout(width="420px"),
        )
        self.rebin_custom_basis_ui.observe(self._on_custom_schedule_basis_change, names="value")

        self.rebin_custom_scale_ui = widgets.Dropdown(
            options=[RebinCustomScale.linear, RebinCustomScale.log, RebinCustomScale.reverse_log],
            value=RebinCustomScale.linear,
            description="scale:",
            disabled=True,
            layout=widgets.Layout(width="260px"),
        )
        self.rebin_custom_scale_ui.observe(self._on_custom_schedule_scale_change, names="value")

        self.rebin_custom_schedule_ui = widgets.Textarea(
            value="0.108, 100\n0.199, 30\n, 60",
            description="segments:",
            disabled=True,
            layout=widgets.Layout(width="520px", height="110px"),
        )
        self.rebin_custom_schedule_help_ui = widgets.HTML()
        self._update_custom_schedule_scale_options()
        self._update_custom_schedule_help()

        display(
            widgets.VBox(
                [
                    widgets.HBox(
                        [self.rebin_custom_basis_ui, self.rebin_custom_scale_ui],
                        layout=widgets.Layout(align_items="center"),
                    ),
                    self.rebin_custom_schedule_ui,
                    self.rebin_custom_schedule_help_ui,
                ]
            )
        )

        display(
            HTML(
                "<span style='font-size: 12px;'>"
                "Preview uses the first selected sample run together with the first selected OB run and shows a simplified "
                "transmission estimate with the proposed bin boundaries. If no spectra file exists, it falls back to "
                "full-image counts from the TIFF stack."
                "</span>"
            )
        )
        self.preview_rebin_boundaries_button = widgets.Button(
            description="Preview rebin boundaries",
            button_style="info",
            disabled=True,
            layout=widgets.Layout(width="260px"),
            tooltip="Overlay the current rebin boundaries on the first sample run data",
        )
        self.preview_rebin_boundaries_button.on_click(self.preview_rebin_boundaries_clicked)
        self.rebin_bin_count_ui = widgets.HTML()
        self.rebin_preview_output_ui = widgets.Output()
        self._update_rebin_snap_to_native_grid_state()
        self._observe_rebin_preview_controls()
        self._update_rebin_bin_count_display()
        display(self.rebin_bin_count_ui)
        display(self.preview_rebin_boundaries_button)
        display(self.rebin_preview_output_ui)

        display(HTML("<hr>"))

        # normalization options
        display(HTML("<span style='font-size: 16px; color:red'>What to take into account for the normalization</span>"))

        if all_nexus_found:
            _value = True
            _disabled=False
        else:
            _value = False
            _disabled = True
        self.proton_charge_flag = widgets.Checkbox(description="Proton charge", 
                                                   value=_value,
                                                   disabled=_disabled)
        
        self.monitor_counts_flag = widgets.Checkbox(description="Monitor counts", 
                                                   value=False,
                                                   disabled=_disabled)

        
        self.experimental_uncertainties_flag = widgets.Checkbox(
            description="Experimental uncertainties (TPX1 detector model)",
            value=not tpx3_disabled_flag,
            disabled=tpx3_disabled_flag,
        )
        self.correct_chips_alignment_flag = widgets.Checkbox(
            description="Correct chips alignment", 
            disabled=True,         
            value=False             
        )

        vertical_layout = widgets.VBox(
            [
                self.proton_charge_flag,
                # self.monitor_counts_flag,
                self.experimental_uncertainties_flag,
                self.correct_chips_alignment_flag,
            ]
        )
        display(vertical_layout)

        display(HTML("<hr>"))
        display(HTML("<span style='font-size: 16px; color:red'>Black-filter background correction for ROI spectrum</span>"))
        display(HTML(
            "<span style='font-size: 12px;'>"
            "Optional. Enable only for production data that include the same Ag black notch used to scale the "
            "background shape. The correction subtracts scaled sample and OB background ROI profiles before "
            "TOF rebinning, then computes corrected transmission."
            "</span>"
        ))
        self.black_filter_background_flag = widgets.Checkbox(
            description="Enable black-filter background correction",
            value=False,
            layout=widgets.Layout(width="450px"),
        )
        self.black_filter_background_shape_file_ui = widgets.Text(
            description="shape CSV:",
            value=DEFAULT_BLACK_FILTER_BACKGROUND_SHAPE_FILE,
            disabled=True,
            layout=widgets.Layout(width="900px"),
        )
        self.black_filter_background_anchor_energy_ui = widgets.BoundedFloatText(
            description="Ag anchor eV:",
            value=5.1044,
            min=0.0,
            max=1e6,
            step=0.0001,
            disabled=True,
            layout=widgets.Layout(width="260px"),
        )
        self.black_filter_background_flag.observe(self._on_black_filter_background_flag_change, names="value")
        display(
            widgets.VBox(
                [
                    self.black_filter_background_flag,
                    self.black_filter_background_shape_file_ui,
                    self.black_filter_background_anchor_energy_ui,
                ]
            )
        )

        display(HTML("<hr>"))
        display(HTML("<span style='font-size: 16px; color:red'>Measured background correction for Bragg edge mode</span>"))
        display(HTML(
            "<span style='font-size: 12px;'>"
            "Optional. Enable only when separate background runs were measured for both sample "
            "and OB. The selected mode controls how the background runs are labeled in the "
            "export; the correction itself subtracts proton-charge-normalized sample and OB "
            "background estimates on the native frame grid before TOF rebinning and normalization."
            "</span>"
        ))
        self.bragg_edge_cd_background_flag = widgets.Checkbox(
            description="Enable Cd-filter background correction for Bragg edge mode",
            value=False,
            layout=widgets.Layout(width="650px"),
        )
        self.bragg_edge_cd_background_flag.observe(
            lambda change: self._on_measured_background_flag_change("bragg_edge_cd", change),
            names="value",
        )
        bragg_edge_cd_background_input_box = self._create_measured_background_run_input_box(
            "bragg_edge_cd",
            "Cd-filter",
        )
        self.closed_slits_background_flag = widgets.Checkbox(
            description="Enable closed-slits background correction",
            value=False,
            layout=widgets.Layout(width="650px"),
        )
        self.closed_slits_background_flag.observe(
            lambda change: self._on_measured_background_flag_change("closed_slits", change),
            names="value",
        )
        closed_slits_background_input_box = self._create_measured_background_run_input_box(
            "closed_slits",
            "closed-slits",
        )
        display(
            widgets.VBox(
                [
                    self.bragg_edge_cd_background_flag,
                    bragg_edge_cd_background_input_box,
                    self.closed_slits_background_flag,
                    closed_slits_background_input_box,
                ]
            )
        )

        display(HTML("<hr>"))

        display(HTML("<span style='font-size: 16px; color:red'>How to handle OB zeros - <i>May take much more time!</i></span>"))
        
        display(widgets.Checkbox(description="Ignore zeros in OB during normalization", 
                                 value=True,
                                 disabled=True,
                                 layout=widgets.Layout(width="600px")))

        self.replace_ob_zeros_by_local_median_flag = widgets.Checkbox(description="Replace zeros by local median", 
                                                                      value=False,
                                                                      layout=widgets.Layout(width="500px"))
        self.replace_ob_zeros_by_local_median_flag.observe(self._on_replace_ob_zeros_by_local_median_flag_change, 
                                                           names='value')
       
        display(self.replace_ob_zeros_by_local_median_flag)

        kernel_size_label = widgets.Label(value="Kernel size for local median (odd number):", 
                                          layout=widgets.Layout(width="300"))
        self.kernel_size_for_local_median_y = widgets.BoundedIntText(description="y axis:",
            value=3, min=1, max=99, step=2, layout=widgets.Layout(width="150px")
        )
        self.kernel_size_for_local_median_x = widgets.BoundedIntText(description="x axis:",
            value=3, min=1, max=99, step=2, layout=widgets.Layout(width="150px")
        )
        self.kernel_size_for_local_median_tof = widgets.BoundedIntText(description="tof axis:",
            value=1, min=1, max=99, step=2, layout=widgets.Layout(width="150px")
        )
        hori_layout = widgets.HBox([kernel_size_label, 
                                    self.kernel_size_for_local_median_y, 
                                    self.kernel_size_for_local_median_x,
                                    self.kernel_size_for_local_median_tof],
                                    hori_layout=widgets.Layout(align_items="center",
                                                               width="100%"))
        display(hori_layout)

        _label = widgets.Label(value="Maximum number of iterations:", layout=widgets.Layout(width="300px")) 
        self.maximum_iterations_ui = widgets.BoundedIntText(
            value=2,
            min=1,
            max=10,
            step=1,
            layout=widgets.Layout(width="50px"),
        )
        hori_layout = widgets.HBox([_label, self.maximum_iterations_ui],
                                   hori_layout=widgets.Layout(align_items="center",
                                                              width="100%"))
        display(hori_layout)

        display(HTML("<hr>"))

        label = widgets.Label(value="Distance source detector (m)", layout=widgets.Layout(width="150px"))
        self.distance_source_detector = widgets.FloatText(
            value=distance_source_detector_m[self.instrument], disabled=False, layout=widgets.Layout(width="150px")
        )
        self.distance_source_detector.observe(self._update_rebin_bin_count_display, names="value")
        hori_layout = widgets.HBox([label, self.distance_source_detector])
        display(hori_layout)

        if self.instrument == "SNAP":
            label = widgets.Label(value="Detector offset (us)", layout=widgets.Layout(width="150px"))
            self.detector_offset_us = widgets.FloatText(value=0.0, disabled=False, layout=widgets.Layout(width="150px"))
            hori_layout = widgets.HBox([label, self.detector_offset_us])
            display(hori_layout)

    def get_integrated_data(self, dict_sample):
        first_sample_run = list(dict_sample.keys())[0]
        notebook_logging.info(f"Loading first sample run: {first_sample_run}")
        list_tiff = retrieve_list_of_tif(first_sample_run)
        notebook_logging.info(f"\tNumber of TIFF files found: {len(list_tiff)}")
        # load the data
        integrated_data = load_data_using_multithreading(list_tiff, combine_tof=True)
        return integrated_data

    def select_container_from_file(self):
        # select container ROI from a previously saved file
        self.roi_container_file = MyFileSelectorPanel(
            instruction="Select ROI container file",
            start_dir=self.output_dir,
            filters={"ROI container files": ["*_container_roi.tiff"]},  # scitiff file 
            type='file',
            multiple=False,
            next=self.load_container_roi_from_file,
        )
        self.roi_container_file.show()

    def load_container_roi_from_file(self, file_path):
        self.container_roi_file = file_path
        notebook_logging.info(f"Loading container ROI from file: {file_path} ...")
        display(HTML(f"<span style='color:blue; font-size:16px'>Will use the container ROI file: {file_path}!</span>"))
       
        master_dict = load_json(file_path)
        self.container_integrated_image = master_dict["integrated_image"]
        self.container_roi = Roi(left=master_dict["container_roi"]["left"], 
                                 top=master_dict["container_roi"]["top"], 
                                 width=master_dict["container_roi"]["width"], 
                                 height=master_dict["container_roi"]["height"])
           
    def select_container(self):

       # load first sample and display integrated image to select ROI
        if not self.dict_sample:
            display(HTML("<span style='color:red'>No sample runs selected!</span>"))
            return

        display(HTML("<span style='font-size: 16px; color:blue'>Select ROI of ONLY the container!</span>"))
        logging.info(f"Selecting ROI of ONLY the container ...")

        if self.integrated_data is None:
            self.integrated_data = self.get_integrated_data(self.dict_sample)

        integrated_data = self.integrated_data

        default_left: int = self.default_roi.left
        default_top: int = self.default_roi.top
        default_width: int = self.default_roi.width
        default_height: int = self.default_roi.height

        vmin = 0
        vmax = int(np.max(integrated_data))
        self.vrange_container = [vmin, vmax]
    
        def container_roi_selection(vrange, left_right, top_bottom):
            
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Create plotly figure with rectangle overlay
            fig = go.Figure()
            fig.add_trace(go.Heatmap(
                z=integrated_data,
                colorscale="viridis",
                zmin=vrange[0],
                zmax=vrange[1],
                colorbar=dict(title="Intensity")
            ))
            
            # Add rectangle overlay
            fig.add_shape(
                type="rect",
                x0=left_right[0], y0=top_bottom[0],
                x1=left_right[1], y1=top_bottom[1],
                line=dict(color="red", width=2),
                fillcolor="rgba(0,0,0,0)"
            )
            
            fig.update_layout(
                title="Select ROI containing only the container",
                width=800, height=800,
                yaxis=dict(autorange="reversed")  # Match imshow y-axis orientation
            )
            fig.show()
            
            logging.info("Updating rectangle ...")
            self.container_roi = Roi(left=left_right[0], top=top_bottom[0], width=left_right[1]-left_right[0], height=top_bottom[1]-top_bottom[0])

        widgets_width = "800px"
        self.interactive_plot = interactive(
            container_roi_selection,
            vrange = widgets.IntRangeSlider(min=0, 
                                            max=int(np.max(integrated_data)), 
                                            step=1, 
                                            value=[0, int(np.max(integrated_data))], 
                                            description="vrange", 
                                            layout=widgets.Layout(width=widgets_width)),
            left_right = widgets.IntRangeSlider(min=0, 
                                max=integrated_data.shape[1]-1, 
                                step=1, 
                                value=[default_left, default_left+default_width],
                                description="left_right",
                                layout=widgets.Layout(width=widgets_width)),
            top_bottom = widgets.IntRangeSlider(min=0,
                                               max=integrated_data.shape[0]-1,
                                               step=1,
                                               value=[default_top, default_top+default_height],
                                               description="top_bottom",
                                               layout=widgets.Layout(width=widgets_width)),
        )
        display(self.interactive_plot)

    def select_roi(self):

        logging.info(f"Selecting ROI for full spectrum normalization...")

        # load first sample and display integrated image to select ROI
        if not self.dict_sample:
            display(HTML("<span style='color:red'>No sample runs selected!</span>"))
            return

        display(HTML("<span style='font-size: 16px; color:blue'>Select ROI for full spectrum normalization!</span>"))

        if self.integrated_data is None:
            self.integrated_data = self.get_integrated_data(self.dict_sample)
        
        integrated_data = self.integrated_data

        default_left = self.default_roi.left
        default_top = self.default_roi.top
        default_width = self.default_roi.width
        default_height = self.default_roi.height

        self.roi = Roi(left=default_left, top=default_top,
                       width=default_width, height=default_height)

        
        def roi_selection(vrange, left_right, top_bottom):
            
            left, right = left_right
            width = right - left
            
            top, bottom = top_bottom
            height = bottom - top
            
            vmin, vmax = vrange
            
            import plotly.graph_objects as go
            
            # Create plotly figure with rectangle overlay
            fig = go.Figure()
            fig.add_trace(go.Heatmap(
                z=integrated_data,
                colorscale="viridis",
                zmin=vmin,
                zmax=vmax,
                colorbar=dict(title="Intensity")
            ))
            
            # Add rectangle overlay
            fig.add_shape(
                type="rect",
                x0=left, y0=top,
                x1=right, y1=bottom,
                line=dict(color="red", width=2),
                fillcolor="rgba(0,0,0,0)"
            )
            
            fig.update_layout(
                title="Select ROI for full spectrum normalization",
                width=800, height=800,
                yaxis=dict(autorange="reversed")  # Match imshow y-axis orientation
            )
            fig.show()
            
            logging.info(f"Selected ROI - left: {left}, top: {top}, width: {width}, height: {height}")
            self.roi = Roi(left=left, top=top, width=width, height=height)

        widgets_width = "800px"
        interactive_plot = interactive(
            roi_selection,
            vrange = widgets.IntRangeSlider(min=0, 
                                            max=int(np.max(integrated_data)), 
                                            step=1, 
                                            value=[0, int(np.max(integrated_data))], 
                                            description="vrange", 
                                            layout=widgets.Layout(width=widgets_width)),
            left_right=widgets.IntRangeSlider(min=0, 
                                              max=integrated_data.shape[1]-1, 
                                              step=1, 
                                              value=[default_left, default_left+default_width],
                                              description="left_right",
                                              layout=widgets.Layout(width=widgets_width)),
            top_bottom=widgets.IntRangeSlider(min=0,
                                              max=integrated_data.shape[0]-1,
                                              step=1,
                                              value=[default_top, default_top+default_height],
                                              description="top_bottom",
                                              layout=widgets.Layout(width=widgets_width)),
        )
            
        display(interactive_plot)

    def post_settings(self):
        
        at_least_one_option = False
        if self.full_spectrum_roi_flag.value:
            self.select_roi()
            at_least_one_option = True

        if self.remove_container_flag.value:

            if self.full_spectrum_roi_flag.value:
                display(HTML("<hr>")) # to improve readability

            if self.remove_container_options_flag.value == "Use previously saved ROI containing only the container signal":
                self.select_container_from_file()
                self.container_roi_from_file = True
            
            else:
                self.select_container()
                self.container_roi_from_file = False
                self.we_need_to_automatically_save_the_container_roi = True
            
            at_least_one_option = True

        if not at_least_one_option:
            self.roi = None
            self.container_roi = None
            display(HTML("<span style='color:blue'>Info: You are good to go, nothing to do here!</span>"))

    def preview_roi_selection_container_imported(self):
         # preview of the roi selected
        if self.container_roi is not None:
            
            if self.container_roi_from_file:
                
                display(HTML("<span style='font-size: 16px; color:blue'>Preview of the loaded ROI from file ...</span>"))

                integrated_data = self.container_integrated_image

                import plotly.graph_objects as go
                
                # Create plotly figure with rectangle overlay
                fig = go.Figure()
                # Calculate 2-98% percentile range to remove outliers
                vmin, vmax = np.percentile(integrated_data, [2, 98])
                fig.add_trace(go.Heatmap(
                    z=integrated_data,
                    colorscale="viridis",
                    zmin=vmin,
                    zmax=vmax,
                    colorbar=dict(title="Intensity")
                ))
                
                # Add rectangle overlay for ROI
                fig.add_shape(
                    type="rect",
                    x0=self.container_roi.left, 
                    y0=self.container_roi.top,
                    x1=self.container_roi.left + self.container_roi.width, 
                    y1=self.container_roi.top + self.container_roi.height,
                    line=dict(color="red", width=2),
                    fillcolor="rgba(0,0,0,0)"
                )
                
                fig.update_layout(
                    title="Loaded ROI containing only the container from file",
                    width=400, height=400,
                    yaxis=dict(autorange="reversed")  # Match imshow y-axis orientation
                )
                fig.show()
                
            else:
                display(HTML("<span style='font-size: 14px; color:blue'>ROI container selected within that notebook (no need to preview again)!</span>"))
            
        else:
            display(HTML("<span style='color:blue'>No container ROI selected!</span>"))

    def _on_replace_ob_zeros_by_local_median_flag_change(self, change):
        if change['new']:
            self.kernel_size_for_local_median_y.disabled = False
            self.kernel_size_for_local_median_x.disabled = False
            self.kernel_size_for_local_median_tof.disabled = False
            self.maximum_iterations_ui.disabled = False
        else:
            self.kernel_size_for_local_median_y.disabled = True
            self.kernel_size_for_local_median_x.disabled = True
            self.kernel_size_for_local_median_tof.disabled = True
            self.maximum_iterations_ui.disabled = True

    def what_to_export(self):
        
        if self.combine_sample_runs_flag.value:
            combined_flag = True
        else:
            combined_flag = False
        
        display(HTML("<span style='font-size: 16px; color:red'>Stack of images</span>"))
        self.export_corrected_stack_of_sample_data = widgets.Checkbox(
            description="Export corrected stack of sample data", layout=widgets.Layout(width="100%"), value=False
        )
        self.export_corrected_stack_of_ob_data = widgets.Checkbox(
            description="Export corrected stack of ob data", 
            layout=widgets.Layout(width="100%"), 
            value=False,
            disabled=True,
        )

        self.export_corrected_stack_of_normalized_data = widgets.Checkbox(
            description="Export corrected stack of each sample run normalized data",
            layout=widgets.Layout(width="100%"),
            value=True,
            disabled=not combined_flag,
        )

        # self.export_corrected_stack_of_combined_normalized_data = widgets.Checkbox(
        #     description="Export corrected stack of combined normalized data (integrated sample divided by integrated ob)",
        #     layout=widgets.Layout(width="100%"),
        #     value=False,
        #     disabled=True,
        # )

        list_widget_to_display =  [
                self.export_corrected_stack_of_sample_data,
                # self.export_corrected_stack_of_ob_data,
                self.export_corrected_stack_of_normalized_data,
        ]
        # if self.combine_sample_runs_flag.value:
        #     list_widget_to_display.append(self.export_corrected_stack_of_combined_normalized_data)
        label = widgets.Label(value="Note: Any of the stacks exported will also contain the original spectra file")
        list_widget_to_display.append(label)

        vertical_layout = widgets.VBox(
            list_widget_to_display
        )

        display(vertical_layout)
        display(HTML("<span style='font-size: 16px; color:red'>Integrated images</span>"))
        self.export_corrected_integrated_sample_data = widgets.Checkbox(
            description="Export corrected integrated sample data", 
            layout=widgets.Layout(width="100%"), 
            value=False
        )
        self.export_corrected_integrated_ob_data = widgets.Checkbox(
            description="Export corrected integrated ob data", 
            layout=widgets.Layout(width="100%"), 
            value=False,
            disabled=True,
        )
        self.export_corrected_integrated_normalized_data = widgets.Checkbox(
            description=(
                "Export integrated normalized data, preview plot, and preview data "
                "(integrated sample divide by integrated ob)"
            ),
            layout=widgets.Layout(width="100%"), 
            value=False
        )

        self.export_corrected_integrated_combined_normalized_data = widgets.Checkbox(
            description="Export corrected integrated combined normalized data",
            layout=widgets.Layout(width="100%"),
            value=False,
            disabled=False,
        )

        list_widget_to_display =  [
                self.export_corrected_integrated_sample_data,
                # self.export_corrected_integrated_ob_data,
                self.export_corrected_integrated_normalized_data,
        ]
        if self.combine_sample_runs_flag.value:
            list_widget_to_display.append(self.export_corrected_integrated_combined_normalized_data)

        vertical_layout = widgets.VBox(
            list_widget_to_display,
        )

        display(vertical_layout)

        if not (self.spectra_array is None) or (self.we_need_to_automatically_save_the_container_roi):
            display(HTML("<span style='font-size: 16px; color:red'>Others</span>"))
            
        if self.spectra_array is None:
            self.export_spectra_file = widgets.Checkbox(
                description="Export spectra file used for normalization", 
                layout=widgets.Layout(width="100%"), 
                value=True)
            display(self.export_spectra_file)
            
        if self.we_need_to_automatically_save_the_container_roi:
            self.export_container_roi = widgets.Checkbox(
                description="Export container ROI file used for normalization", 
                layout=widgets.Layout(width="100%"), 
                value=True)
            display(self.export_container_roi)
            
    def check_folder_is_valid(self, full_path):
        list_tiff = glob.glob(os.path.join(full_path, "*.tif*"))
        if list_tiff:
            return True, {"nbr_tiff": len(list_tiff)}
        else:
            return False, {"nbr_tiff": 0}

    def sample_folder_selected(self, folder_selected):
        self.sample_folder = folder_selected
        display(HTML(f"Sample folder selected: <span style='color:blue'>{folder_selected}</span>"))

    def ob_folder_selected(self, folder_selected):
        self.ob_folder = folder_selected
        display(HTML(f"Open beam folder selected: <span style='color:blue'>{folder_selected}</span>"))

    def dc_folder_selected(self, folder_selected):
        self.dc_folder = folder_selected
        display(HTML(f"Dark current folder selected: <span style='color:blue'>{folder_selected}</span>"))

    def save_ob_run_numbers_selected(self, folder_selected):
        self.ob_run_numbers_selected = folder_selected
        self.ob_dir = os.path.dirname(folder_selected[0])
     
    def save_dc_run_numbers_selected(self, folder_selected):
        self.dc_run_numbers_selected = folder_selected

    def save_bragg_edge_cd_sample_background_run_numbers_selected(self, folder_selected):
        self.bragg_edge_cd_sample_background_run_numbers_selected = folder_selected

    def save_bragg_edge_cd_ob_background_run_numbers_selected(self, folder_selected):
        self.bragg_edge_cd_ob_background_run_numbers_selected = folder_selected
        
    def output_folder_selected(self, folder_selected):

        # close shared and home buttons
        self.list_input_folders_ui.shortcut_buttons.close()

        with self.list_input_folders_ui.out:
            self.output_folder = folder_selected
            display(HTML("Output folder selected:"))
            if os.path.exists(folder_selected):
                display(HTML(f"<span style='color:green'>{folder_selected} - FOUND!</span>"))
                notebook_logging.info(f"Output folder selected: {folder_selected} - FOUND")
            else:
                display(HTML(f"<span style='color:blue'>{folder_selected} - DOES NOT EXIST and will be CREATED!</span>"))
                notebook_logging.info(f"Output folder selected: {folder_selected} - NOT FOUND and will be CREATED!")

    def select_folder(self, instruction="Select a folder",
                       next_function=None, 
                       ipts_folder=None,
                       start_dir=None, 
                       multiple=False,
                       newdir_toolbar_button=False):
        # go straight to autoreduce/mcp folder
        if start_dir is None:
            start_dir = self.autoreduce_dir

        while not os.path.exists(start_dir):
            start_dir = os.path.dirname(start_dir)

        if ipts_folder is None:
            self.list_input_folders_ui = MyFileSelectorPanel(
                instruction=instruction,
                start_dir=start_dir,
                type="directory",
                newdir_toolbar_button=newdir_toolbar_button,
                multiple=multiple,
                sort_in_reverse=True,
                # sort_increasing=False,
                next=next_function,
            )
            self.list_input_folders_ui.show()
            
        else:
            self.list_input_folders_ui = MyFileSelectorPanelWithJumpFolders(
                instruction=instruction,
                start_dir=start_dir,
                type="directory",
                ipts_folder=self.ipts_folder,
                newdir_toolbar_button=newdir_toolbar_button,
                next=next_function,
                show_jump_to_share=True,
                show_jump_to_home=True,
            )

    # calling main code
    def run_normalization_with_list_of_runs(self, preview=False):
        # sample_run_numbers = self.sample_run_numbers
        # ob_run_numbers = self.ob_run_numbers
        output_folder = self.output_folder

        export_mode = {
            "sample_stack": self.export_corrected_stack_of_sample_data.value,
            "ob_stack": self.export_corrected_stack_of_ob_data.value,
            "normalized_stack": self.export_corrected_stack_of_normalized_data.value,
            # "combined_normalized_stack": self.export_corrected_stack_of_combined_normalized_data.value,
            "sample_integrated": self.export_corrected_integrated_sample_data.value,
            "ob_integrated": self.export_corrected_integrated_ob_data.value,
            "normalized_integrated": self.export_corrected_integrated_normalized_data.value,
            "combined_normalized_integrated": self.export_corrected_integrated_combined_normalized_data.value,
            "x_axis": True,  # always export x axis
        }

        detector_delay_us = None
        if self.instrument == "SNAP":
            detector_delay_us = self.detector_offset_us.value

        self._collect_enabled_measured_background_run_numbers()
        self.retrieve_nexus_file_path()

        sample_dict = {}
        for _full_path in self.dict_sample.keys():
            sample_dict[os.path.basename(_full_path)] = {
                "full_path": _full_path,
                "nexus": self.dict_sample[_full_path]["nexus"],
            }

        ob_dict = {}
        for _full_path in self.dict_ob.keys():
            ob_dict[os.path.basename(_full_path)] = {
                "full_path": _full_path,
                "nexus": self.dict_ob[_full_path]["nexus"],
            }

        dc_dict = {}
        if self.dict_dc:
            logging.info("Dark current runs provided")
            for _full_path in self.dict_dc.keys():
                dc_dict[os.path.basename(_full_path)] = {
                    "full_path": _full_path,
                    "nexus": self.dict_dc[_full_path]["nexus"],
                }

        def _make_background_input_dict(background_dict, label):
            output_dict = {}
            if background_dict:
                logging.info(f"{label} runs provided for Bragg edge mode")
                for _full_path in background_dict.keys():
                    output_dict[os.path.basename(_full_path)] = {
                        "full_path": _full_path,
                        "nexus": background_dict[_full_path]["nexus"],
                    }
            return output_dict

        bragg_edge_cd_sample_background_dict = _make_background_input_dict(
            self.dict_bragg_edge_cd_sample_background,
            "Sample Cd-filter measured background",
        )
        bragg_edge_cd_ob_background_dict = _make_background_input_dict(
            self.dict_bragg_edge_cd_ob_background,
            "OB Cd-filter measured background",
        )
        closed_slits_sample_background_dict = _make_background_input_dict(
            self.dict_closed_slits_sample_background,
            "Sample closed-slits measured background",
        )
        closed_slits_ob_background_dict = _make_background_input_dict(
            self.dict_closed_slits_ob_background,
            "OB closed-slits measured background",
        )

        if self.correct_chips_alignment_flag.value:
            if self.detector_type in [DetectorType.tpx1_legacy, DetectorType.tpx1]:
                correct_chips_alignment_config = timepix1_config
            elif self.detector_type == DetectorType.tpx3:
                correct_chips_alignment_config = timepix3_config
            else:
                correct_chips_alignment_config = None
        else:
            correct_chips_alignment_config = None

        spectra_array = self.spectra_array
        rebin_mode = self.rebin_mode_ui.value
        rebin_custom_schedule = None
        if rebin_mode == RebinMode.custom_schedule:
            try:
                rebin_custom_schedule = self._parse_custom_rebin_schedule()
            except ValueError as exc:
                display(HTML(f"<span style='color:red'>{exc}</span>"))
                raise

        black_filter_background_config = None
        if getattr(self, "black_filter_background_flag", None) is not None and self.black_filter_background_flag.value:
            black_filter_background_config = {
                "enabled": True,
                "background_shape_file": self.black_filter_background_shape_file_ui.value,
                "anchor_energy_eV": self.black_filter_background_anchor_energy_ui.value,
            }
        measured_background_configs = []
        if getattr(self, "bragg_edge_cd_background_flag", None) is not None and self.bragg_edge_cd_background_flag.value:
            measured_background_configs.append(
                {
                    "key": "cd",
                    "enabled": True,
                    "mode": "Cd-filter background correction for Bragg edge mode",
                    "column_label": "Cd-filter",
                    "key_prefix": "bragg_edge_cd",
                    "sample_background_dict": bragg_edge_cd_sample_background_dict,
                    "ob_background_dict": bragg_edge_cd_ob_background_dict,
                    "weight": 1.0,
                }
            )
        if getattr(self, "closed_slits_background_flag", None) is not None and self.closed_slits_background_flag.value:
            measured_background_configs.append(
                {
                    "key": "closed_slits",
                    "enabled": True,
                    "mode": "Closed-slits background correction for Bragg edge mode",
                    "column_label": "closed-slits",
                    "key_prefix": "closed_slits",
                    "sample_background_dict": closed_slits_sample_background_dict,
                    "ob_background_dict": closed_slits_ob_background_dict,
                    "weight": 1.0,
                }
            )
        for _config in measured_background_configs:
            _config.pop("key", None)

        self.normalized_dict = normalization_with_list_of_full_path(
            sample_dict=sample_dict,
            ob_dict=ob_dict,
            dc_dict=dc_dict,
            bragg_edge_cd_sample_background_dict=bragg_edge_cd_sample_background_dict,
            bragg_edge_cd_ob_background_dict=bragg_edge_cd_ob_background_dict,
            spectra_array=spectra_array,
            output_folder=output_folder,
            proton_charge_flag=self.proton_charge_flag.value,
            # replace_ob_zeros_by_nan_flag=self.replace_ob_zeros_by_nan_flag.value,
            replace_ob_zeros_by_local_median_flag=self.replace_ob_zeros_by_local_median_flag.value,
            kernel_size_for_local_median=(self.kernel_size_for_local_median_y.value,
                                          self.kernel_size_for_local_median_x.value,
                                          self.kernel_size_for_local_median_tof.value),
            max_iterations=self.maximum_iterations_ui.value,
            correct_chips_alignment_flag=self.correct_chips_alignment_flag.value,
            correct_chips_alignment_config=correct_chips_alignment_config,
            verbose=True,
            instrument=self.instrument,
            detector_delay_us=detector_delay_us,
            preview=preview,
            distance_source_detector_m=self.distance_source_detector.value,
            export_mode=export_mode,
            combine_samples=self.combine_sample_runs_flag.value,
            roi=self.roi,
            container_roi=self.container_roi,
            container_roi_file=self.container_roi_file,
            rebin_mode=rebin_mode,
            rebin_delta_tof_us=self.rebin_delta_tof_us_ui.value if rebin_mode == RebinMode.linear_tof else None,
            rebin_delta_lambda_a=(
                self.rebin_delta_lambda_a_ui.value if rebin_mode == RebinMode.linear_lambda else None
            ),
            rebin_delta_tof_over_tof=(
                self.rebin_delta_tof_over_tof_ui.value if rebin_mode == RebinMode.log_tof else None
            ),
            rebin_delta_lambda_over_lambda=(
                self.rebin_delta_lambda_over_lambda_ui.value if rebin_mode == RebinMode.log_lambda else None
            ),
            rebin_delta_lambda_squared_a2=(
                self.rebin_delta_lambda_squared_a2_ui.value
                if rebin_mode == RebinMode.inverse_log_lambda
                else None
            ),
            rebin_custom_basis=(
                self.rebin_custom_basis_ui.value if rebin_mode == RebinMode.custom_schedule else None
            ),
            rebin_custom_scale=(
                self.rebin_custom_scale_ui.value if rebin_mode == RebinMode.custom_schedule else None
            ),
            rebin_custom_schedule=rebin_custom_schedule,
            rebin_full_bins_only=self.rebin_full_bins_only_ui.value if rebin_mode != RebinMode.none else False,
            rebin_snap_to_native_grid=(
                self.rebin_snap_to_native_grid_ui.value
                if (
                    rebin_mode != RebinMode.none
                    and hasattr(self, "rebin_snap_to_native_grid_ui")
                    and self._fixed_width_rebin_mode_supports_native_snap()
                )
                else False
            ),
            experimental_uncertainties_flag=self.experimental_uncertainties_flag.value,
            black_filter_background_config=black_filter_background_config,
            measured_background_correction_configs=measured_background_configs,
        )
        
        display(HTML("<span style='color:blue'>Normalization completed</span>"))
        # display(HTML("Log file: /SNS/VENUS/shared/logs/normalization_for_timepix.log"))
        
    def profile_of_roi(self):
        normalized_data = self.normalized_dict.data

        lambda_array = self.normalized_dict.lambda_array
        energy_array = self.normalized_dict.energy_array
        tof_array = self.normalized_dict.tof_array

        def plot_normalized_profile_of_roi(index, left=0, top=0, width=50, height=50):

            _normalized_data = normalized_data[index]
            _integrated = np.nanmean(_normalized_data, axis=0)
            _profile = np.nanmean(_normalized_data[:, top : top + height, left : left + width], axis=0)
            
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[f"Integrated normalized data - {index}", "", f"Profile of ROI - {index}", ""],
                specs=[[{"type": "heatmap"}, {"type": "xy"}], 
                       [{"type": "xy"}, {"type": "xy"}]]
            )
            
            # Add heatmap for integrated data
            # Calculate 2-98% percentile range to remove outliers
            vmin, vmax = np.percentile(_integrated, [2, 98])
            fig.add_trace(
                go.Heatmap(
                    z=_integrated,
                    colorscale="viridis",
                    zmin=vmin,
                    zmax=vmax,
                    colorbar=dict(title="Intensity", x=0.45)
                ),
                row=1, col=1
            )
            
            # Add rectangle overlay for ROI
            fig.add_shape(
                type="rect",
                x0=left, y0=top,
                x1=left + width, y1=top + height,
                line=dict(color="red", width=2),
                fillcolor="rgba(0,0,0,0)",
                row=1, col=1
            )
            
            # Add line plot for profile
            fig.add_trace(
                go.Scatter(
                    x=lambda_array,
                    y=_profile,
                    mode='lines',
                    name='Profile'
                ),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                height=600, width=800,
                showlegend=False
            )
            
            # Update y-axis for heatmap to match imshow orientation
            fig.update_yaxes(autorange="reversed", row=1, col=1)
            
            # Update x and y axis labels for profile plot
            fig.update_xaxes(title_text="lambda_array", row=2, col=1)
            fig.update_yaxes(title_text="Intensity (a.u.)", row=2, col=1)
            
            fig.show()

        _plot_normalized = interactive(widgets.Dropdown(options=list(normalized_data.keys()), 
                                                       description="Sample run:",
                                                       layout=widgets.Layout(width="300px")),
                                      left=widgets.BoundedIntText(value=0, min=0, max=512, step=1, description="left:", layout=widgets.Layout(width="200px")),
                                      top=widgets.BoundedIntText(value=0, min=0, max=512, step=1, description="top:", layout=widgets.Layout(width="200px")),
                                      width=widgets.BoundedIntText(value=50, min=1, max=512, step=1, description="width:", layout=widgets.Layout(width="200px")),
                                      height=widgets.BoundedIntText(value=50, min=1, max=512, step=1, description="height:", layout=widgets.Layout(width="200px")),
                                      function=widgets.Dropdown(options=["mean", "median"], description="Function:", layout=widgets.Layout(width="200px")),
        )
        display(_plot_normalized)

    @classmethod
    def legend(cls) -> None:
        display(HTML("<hr style='height:2px'/>"))
        display(HTML("<h2>Legend</h2>"))
        display(HTML("<ul>"
                     "<li><b><font color='red'>Mandatory steps</font></b> must be performed to ensure proper data preparation and reconstruction.</li>"
                     "<li><b><font color='orange'>Optional but recommended steps</font></b> are not mandatory but should be performed to ensure proper data preparation and reconstruction.</li>"
                     "<li><b><font color='purple'>Optional steps</font></b> are not mandatory but highly recommended to improve the quality of your reconstruction.</li>"
                     "</ul>"))
        display(HTML("<hr style='height:2px'/>"))
