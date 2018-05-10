from unittest import TestCase
import numpy as np
import utils

class TestUtils(TestCase):

    def test_fileload(self):
        os_data_file = "cleanData/ah_filteredData_OS.mat"
        os_df = utils.load_individual_os_data(os_data_file)
        self.assertTrue(np.all(os_df[os_df.Orientation==-1]['MaskContrast']==0))
        self.assertTrue(np.all(os_df[os_df.Orientation==-2]['MaskContrast']==0))