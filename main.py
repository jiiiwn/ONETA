from natsort import natsorted
from scipy.io import loadmat

from inference import *

if __name__ == "__main__":

    num_style = 30  # Retouching 8 + ISP 9 + UW 2 + LLIE 7 + Dehazing 2 + WB 2
    
    # u mat file load
    u_path = f'/hdd1/jwkim/multistyle_all/matrix/final/u_matrix_all_dataset(train).mat'
    data_u = loadmat(u_path)
    u_mat = data_u['u_matrix']  # 256 * 20
    
    # test data path concat
    rgb_list_test = []
    gt_list_test = []
    test_num_list = []
    
    for i in range(num_style):
        rgb_test_dir = f'/hdd1/jwkim/multistyle_all/{i}/input/test/'
        gt_test_dir = f'/hdd1/jwkim/multistyle_all/{i}/gt/test/'
        
        if i == 26 or i == 27 or i ==29:
            rgb_test_paths, gt_test_paths = get_all_file_paths_asym(rgb_test_dir, gt_test_dir)
        
        else :
            rgb_test_paths = get_all_file_paths(rgb_test_dir)
            gt_test_paths = get_all_file_paths(gt_test_dir)
            
        
        rgb_list_test += natsorted(rgb_test_paths)
        gt_list_test += natsorted(gt_test_paths)
        test_num_list.append(len(rgb_test_paths))

    for i in range(num_style-1):
        test_num_list[i+1] = test_num_list[i] + test_num_list[i+1]
    
    test_num_list.insert(0, 0)
    test_loader = list(zip(rgb_list_test, gt_list_test))

    # save path setting
    save_dir = '/hdd1/jwkim/multistyle_all/result/test/'

    test(test_loader, test_num_list, num_style, u_mat, save_dir)






























