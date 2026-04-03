try:
    from qtpy.uic import loadUi
except ModuleNotFoundError:
    loadUi = None

LOGGER_FILE = "/SNS/users/j35/logger/notebook_logger.log"
# LOGGER_FILE = "/Users/j35/logger/notebook_logger.log"

__all__ = ["load_ui"]


def load_ui(ui_filename, baseinstance):
    if loadUi is None:
        raise ModuleNotFoundError("qtpy is not installed. load_ui is unavailable in this slimmed repository.")
    return loadUi(ui_filename, baseinstance=baseinstance)


interact_me_style = "background-color: lime"
error_style = "background-color: red"
normal_style = ""
