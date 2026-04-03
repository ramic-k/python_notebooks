import os

from IPython.display import display
from ipywidgets import widgets

from __code.ipywe.fileselector import FileSelectorPanel


class FileSelectorPanelWithJumpFolders:
    def __init__(
        self,
        instruction="Select Output Folder",
        start_dir=".",
        type="file",
        next=None,
        multiple=False,
        newdir_toolbar_button=False,
        custom_layout=None,
        filters=None,
        default_filter=None,
        stay_alive=False,
        ipts_folder="./",
        show_jump_to_share=True,
        show_jump_to_home=True,
    ):
        self.type = type
        self.next = next
        self.out = widgets.Output()

        filters = {} if filters is None else filters

        def _show_selector(_start_dir):
            with self.out:
                self.out.clear_output(wait=True)
                self.display_file_selector(
                    instruction=instruction,
                    start_dir=_start_dir,
                    type=type,
                    next=next,
                    multiple=multiple,
                    newdir_toolbar_button=newdir_toolbar_button,
                    custom_layout=custom_layout,
                    filters=filters,
                    default_filter=default_filter,
                    stay_alive=stay_alive,
                )

        def display_file_selector_from_shared(_event):
            shared_start_dir = os.path.join(ipts_folder, "shared")
            if not os.path.exists(shared_start_dir):
                shared_start_dir = os.path.expanduser("~")
            _show_selector(shared_start_dir)

        def display_file_selector_from_home(_event):
            _show_selector(os.path.expanduser("~"))

        ipts = os.path.basename(ipts_folder)
        button_layout = widgets.Layout(width="30%", border="1px solid gray")

        list_buttons = []
        if show_jump_to_share:
            share_button = widgets.Button(
                description=f"Jump to {ipts} Shared Folder",
                button_style="success",
                layout=button_layout,
            )
            share_button.on_click(display_file_selector_from_shared)
            list_buttons.append(share_button)

        if show_jump_to_home:
            home_button = widgets.Button(
                description="Jump to My Home Folder",
                button_style="success",
                layout=button_layout,
            )
            home_button.on_click(display_file_selector_from_home)
            list_buttons.append(home_button)

        self.shortcut_buttons = widgets.HBox(list_buttons)
        display(self.shortcut_buttons)
        display(self.out)

        _show_selector(start_dir)

    def display_file_selector(
        self,
        instruction="",
        start_dir="./",
        multiple=False,
        default_filter=None,
        next=None,
        newdir_toolbar_button=False,
        type="file",
        custom_layout=None,
        filters=None,
        stay_alive=False,
    ):
        filters = {} if filters is None else filters
        self.output_folder_ui = FileSelectorPanel(
            instruction=instruction,
            start_dir=start_dir,
            multiple=multiple,
            next=next,
            newdir_toolbar_button=newdir_toolbar_button,
            type=type,
            custom_layout=custom_layout,
            default_filter=default_filter,
            filters=filters,
            stay_alive=stay_alive,
        )
        self.output_folder_ui.show()
