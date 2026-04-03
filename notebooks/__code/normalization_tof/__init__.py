autoreduce_dir = {
    "VENUS": ["/SNS/VENUS/", "/shared/autoreduce/mcp/images"],
    "SNAP": ["/SNS/SNAP/", "/shared/autoreduce/mcp/"],
}

# shared_dir = {'VENUS': ["/SNS/VENUS/", "/shared/",],
#               'SNAP': ["/SNS/SNAP/", "/shared/",]}

distance_source_detector_m = {
    "VENUS": 25.0,  # in meters
    "SNAP": 14.0,  # in meters
}


class DataType:
    sample: str = "sample"
    ob: str = "ob"
    dc: str = "dc"

class Roi:

    def __init__(self, left: int = 0, top: int = 0, width: int = 1, height: int = 1):
        self.left: int = left
        self.top: int = top
        self.width: int = width
        self.height: int = height

    def __repr__(self):
        return f"Roi(left={self.left}, top={self.top}, width={self.width}, height={self.height})"


class DetectorType:
    tpx1_legacy = "tpx1 - old naming convention (until July 2025)"
    tpx1 = "tpx1 - new naming convention (from August 2025)"
    tpx3 = "tpx3"


class RebinMode:
    none = "No rebin"
    linear_tof = "linear_tof(delta_us)"
    linear_lambda = "linear_lambda(delta_A)"
    log_tof = "log_tof(delta_tof_over_tof)"
    log_lambda = "log_lambda(delta_lambda_over_lambda)"
    inverse_log_lambda = "inverse_log_lambda(delta_lambda_squared_A2)"
    custom_schedule = "custom_schedule(axis_segments)"


class RebinCustomBasis:
    tof = "tof"
    lambda_ = "lambda"
    lambda_squared = "lambda^2"


class RebinCustomScale:
    linear = "linear"
    log = "log"
    reverse_log = "reverse_log"


raw_dir = {
    "VENUS": {
        DetectorType.tpx1_legacy: ["/SNS/VENUS/", "images/mcp/images/"],
        DetectorType.tpx1: ["/SNS/VENUS/", "images/tpx1/"],
        DetectorType.tpx3: ["/SNS/VENUS/", ""],
    },
    "SNAP": {
        DetectorType.tpx1_legacy: ["/SNS/SNAP/", "images/mcp/"],
    },
}


autoreduce_dir = {
    "VENUS": {
        DetectorType.tpx1_legacy: ["/SNS/VENUS/", "shared/autoreduce/mcp/images/"],
        DetectorType.tpx1: ["/SNS/VENUS/", "shared/autoreduce/images/tpx1/"],
        DetectorType.tpx3: ["/SNS/VENUS/", "images/tpx3/"],
    },
    "SNAP": {
        DetectorType.tpx1_legacy: ["/SNS/SNAP/", "images/mcp/"],
    },
}
