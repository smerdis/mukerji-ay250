import numpy as np
import pandas as pd
from scipy.io import loadmat

def load_individual_data(data_file, columns):
    """Function that loads individual psychophysics data stored in .mat files.
    
    data_file: path to .mat file
    columns: list of column names. must match number of columns in the .mat file"""

    subject_code = data_file.split('_filteredData')[0].split('/')[-1] # get subject name from filename
    mat = loadmat(data_file)
    mdata = mat['data']
    mdtype = mdata.dtype
    subj_df = pd.DataFrame(data=mdata, columns=columns)
    
    # Convert the following columns to ints if they exist in this data frame
    cols_to_intify = ["StaircaseNumber", "Eye", "Orientation", "Presentation", "TrialNumberStaircase",
                "ResponseAccuracy", "ProbeInterval", "ProbeLocation", "FileNumber"]
    # Furthermore, make these ones categorical too
    cols_to_categorize = ["Eye", "Presentation", "ResponseAccuracy"]
    for col in cols_to_intify:
        if col in subj_df.columns:
            subj_df[col] = subj_df[col].astype(int)
            if col in cols_to_categorize:
                subj_df[col] = pd.Categorical(subj_df[col])
                
    # Finally, round these ones to avoid floating point errors
    cols_to_round = ["ProbeContrastRecommended", "ProbeContrastUsed"]
    for col in cols_to_round:
        if col in cols_to_intify or col in cols_to_categorize:
            raise Error(f"Column {col} is listed as both integer and floating point!")
        subj_df[col] = np.round(subj_df[col], 2)

    # Make Subject column and put it first
    subj_df['Subject'] = subject_code
    subj_df = subj_df[['Subject', *columns]]
    return subj_df

def load_individual_os_data(data_file):
    """
    Load data for Orientation Suppression task which has 11 columns.

    Staircase number for a given test block (= file number)
    Eye (1=weaker eye, 2= fellow eye)
    Mask Orientation (0=parallel, 90=orthogonal)
    Binocular condition (1= monocular, 2=dichoptic)
    Mask Contrast (michelson)
    Trial number for this staircase
    Probe contrast recommended by staircase algorithm
    Response Accuracy (1=correct, 0=incorrect)
    Probe Contrast used (I don't remember it ever being different from #7 and was really just a sanity check)
    Interval that probe was presented in (1 or 2)
    File number (=test block)
    """
    columns_os = ["StaircaseNumber", "Eye", "Orientation", "Presentation", "MaskContrast", "TrialNumberStaircase",
              "ProbeContrastRecommended", "ResponseAccuracy", "ProbeContrastUsed", "ProbeInterval", "FileNumber"]
    return load_individual_data(data_file, columns_os)

def load_individual_ss_data(data_file):
    """
    Load data for Surround Suppression task which has 12 columns.

    Staircase number for a given test block (= file number)
    Eye (1=weaker eye, 2= fellow eye)
    Mask Orientation (0=parallel, 90=orthogonal)
    Binocular Condition (1= monocular, 2=dichoptic)
    Trial number for this staircase
    Contrast increment recommended by staircase algorithm
    Response Accuracy (1=correct, 0=incorrect)
    Mask Contrast (michelson)
    Probe location (1-4, let me know if you need to know which number represents which quadrant)
    Response (1-4)
    Probe contrast increment used (I don't remember it ever being different from #6 and was really just a sanity check)
    File number (=test block)
    """
    columns_ss = ["StaircaseNumber", "Eye", "Orientation", "Presentation", "TrialNumberStaircase",
              "ProbeContrastRecommended", "ResponseAccuracy", "MaskContrast", "ProbeLocation",
                  "Response", "ProbeContrastUsed",  "FileNumber"]
    return load_individual_data(data_file, columns_ss)