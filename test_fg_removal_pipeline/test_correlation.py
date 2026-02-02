import sys
sys.path.insert(0, "../src/fg_removal_pipe")  # add Folder_2 path to search list
from correlation import pearson_correl
from scipy import stats

def test_corr():
	test_x = [1,2,3,4,4,8]
	test_y = [4,5,2,44,0,10]

	assert np.allclose(pearson_correl(test_x, test_y), stats.pearsonr(test_x, test_y), rtol=1e-6, atol=1e-8)





