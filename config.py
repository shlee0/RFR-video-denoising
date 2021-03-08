import argparse, os


# dataset path root
DATA_HOME = '/home/shlee/dataset/'


DERF_HD = os.path.join(DATA_HOME, 'derf_HD')
DERF_HD_15 = DERF_HD + '_15'
DERF_HD_25 = DERF_HD + '_25'
DERF_HD_40 = DERF_HD + '_40'

DERF_HD_GRAY = os.path.join(DATA_HOME, 'derf_HD_gray')
DERF_HD_GRAY_15 = DERF_HD_GRAY + '_15'
DERF_HD_GRAY_25 = DERF_HD_GRAY + '_25'
DERF_HD_GRAY_40 = DERF_HD_GRAY + '_40'

DAVIS = os.path.join(DATA_HOME, 'davis')
DAVIS_15 = DAVIS + '_15'
DAVIS_25 = DAVIS + '_25'
DAVIS_40 = DAVIS + '_40'



# dataset dict
TEST_DS_DICT = {
	'DERF_HD_15' : {'noisy' : DERF_HD_15, 'clean' : DERF_HD},
	'DERF_HD_25' : {'noisy' : DERF_HD_25, 'clean' : DERF_HD},
	'DERF_HD_40' : {'noisy' : DERF_HD_40, 'clean' : DERF_HD},

	'DERF_HD_GRAY_15' : {'noisy' : DERF_HD_GRAY_15, 'clean' : DERF_HD_GRAY},
	'DERF_HD_GRAY_25' : {'noisy' : DERF_HD_GRAY_25, 'clean' : DERF_HD_GRAY},
	'DERF_HD_GRAY_40' : {'noisy' : DERF_HD_GRAY_40, 'clean' : DERF_HD_GRAY},

	'DAVIS_15' : {'noisy' : DAVIS_15, 'clean' : DAVIS},
	'DAVIS_25' : {'noisy' : DAVIS_25, 'clean' : DAVIS},
	'DAVIS_40' : {'noisy' : DAVIS_40, 'clean' : DAVIS}
}




# args parser
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='-1')
parser.add_argument('--net', type=str, choices=['vnlnet', 'fastdvdnet', 'dncnn'], required=True)
parser.add_argument('--test', type=str, choices=[
	'DERF_HD_15', 'DERF_HD_25', 'DERF_HD_40',
	'DERF_HD_GRAY_15', 'DERF_HD_GRAY_25', 'DERF_HD_GRAY_40',
	'DAVIS_15', 'DAVIS_25', 'DAVIS_40'
], required=True)
parser.add_argument('--offline', action='store_true')
parser.add_argument('--iter', type=int, default='1')
parser.add_argument('--rebuttal', action='store_true')