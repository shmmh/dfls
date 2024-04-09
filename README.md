# Guide for Util.py functions

To import any function from the file
```python
from utils import parse_data, plot_data
```

## Parsing the data

The `csv` data exported by the oscilloscope has the same structure so we use python to extract all the relevant data into a usable format instead of dealing with `pandas` dataframes directly.

The `parse_data` function takes one argument `file_path` and returns a python dictionary `dict`. The returned `dict` has the following structure.

```python

data = parse_data('relative_path_to_file')

# The data structure of parse_data
{
        "CH1": {},
        "CH2":  {} #if channel 2 data exists
}

# The channels are also python dicts. eg. data['CH1'] has the following structure
{
    'x': # numpy array of time in (s)
    'y': # numpy array of voltages in (V)
    'Trace Info': {} # python dict with additional info about the trace
}

# The Trace Info is not too important but if needed, it has the following structure.
# eg. data['CH1']['Trace Info'] has the following structure

{
    "Source":           # CH1 or CH2,
    "Vertical Units":   # Vertical Axes Unit. Always (V),
    "Horizontal Units": # Horizontal Axes Unit. Always (s),
    "Vertical Scale":   # The Vertical Division of oscilloscope,
    "Vertical Offset":  # The offset in (V),
    "Horizontal Scale": # The Horizontal Division of oscilloscope,
    "Pt Fmt":           # No idea what this is,
    "Yzero":            # The true zero point for the Y axis with the offset,
    "Probe Atten":      # The attenuation factor for the probe. Not important,
    "Date":             # The date the measurement was taken,
    "Time":             # The time the measurement was taken,
    "Model":            # The model of the oscilloscope,
    "Record Length": {
        "value":        # The length of the record,
        "unit":         # The unit of the record length
    },
    "Sample Interval": {
        "value":        # The interval between samples,
        "unit":         # The unit of the sample interval
    },
    "Trigger Point": {
        "value":        # The point at which the oscilloscop triggers,
        "unit":         # The unit of the trigger point
    }
}

```
