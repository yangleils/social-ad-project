# 路径
path_pre = '../'
path_original_dataset = path_pre + 'original-dataset/'
path_intermediate_dataset = path_pre + 'intermediate-dataset/'
path_modeling_dataset = path_pre + 'modeling-dataset/'
path_model = path_pre + 'model/'
path_submission_dataset = path_pre + 'submission-dataset/'

# 原始文件
csv_ad = 'ad.csv'
csv_app_cat = 'app_categories.csv'
csv_pos = 'position.csv'
csv_test = 'test.csv'
csv_train = 'train.csv'
csv_action = 'user_app_actions.csv'
csv_user = 'user.csv'
csv_user_app = 'user_installedapps.csv'

# 原始文件的 hdf 存储（已经过清洗与裁剪以节省计算性能）
hdf_ad = 'ad.h5'
hdf_app_cat = 'app_cat.h5'
hdf_pos = 'pos.h5'
hdf_test_ol = 'test_ol.h5'
hdf_train = 'train.h5'
hdf_action = 'action.h5'
hdf_user = 'user.h5'
hdf_user_app = 'user_app.h5'

# 计算的中间结果（以此减少内存占用）
hdf_user_app_cat = 'user_app_cat.h5'
hdf_userID_appID_pair_installed = 'userID_appID_pair_installed.h5'

# 从单个原始特征中提取出的特征
hdf_user_pref_cat = 'f_user_pref_cat.h5'
hdf_user_cat_weight = 'f_user_cat_weight.h5'
hdf_app_popularity = 'f_app_popularity.h5'
hdf_user_activity = 'f_user_activity.h5'
hdf_hour_weight = 'f_hour_weight.h5'
hdf_conversion_ratio_connectionType = 'f_conversion_ratio_connectionType.h5'
hdf_conversion_ratio_telecomsOperator = 'f_conversion_ratio_telecomsOperator.h5'
hdf_hour = 'f_hour.h5'
hdf_userID = 'f_userID.h5'

# 特征群文件
hdf_context_dataset_fg = 'fg_context_dataset.h5'
hdf_context_testset_ol_fg = 'fg_context_testset_ol.h5'
hdf_ad_fg = 'fg_ad.h5'
hdf_user_fg = 'fg_user.h5'

# 最终的数据集文件
hdf_dataset = 'dataset.h5'
hdf_testset_ol = 'testset_ol.h5'

# 稀疏矩阵, 一次项
npz_ad = 'ad_csc.npz'
npz_ad_test_ol = 'ad_csc_test_ol.npz'
npz_context = 'context_csc.npz'
npz_context_test_ol = 'context_csc_test_ol.npz'
npz_user = 'user_csc.npz'
npz_user_test_ol = 'user_csc_test_ol.npz'
# 二次项
npz_user_ad = 'user_ad.npz'
npz_user_ad_test_ol = 'user_ad_test_ol.npz'
npz_user_context = 'user_context.npz'
npz_user_context_test_ol = 'user_context_test_ol.npz'
npz_ad_context = 'ad_context.npz'
npz_ad_context_test_ol = 'ad_context_test_ol.npz'

# ndarray
npy_y = 'y.npy'
npy_y_train = 'y_train.npy'
npy_y_test = 'y_test.npy'

# 稀疏矩阵
npz_X_linear = 'X_linear.npz'
npz_X_interactive = 'X_interactive.npz'
npz_X = 'X.npz'
npz_X_test_ol_linear = 'X_test_ol_linear.npz'
npz_X_test_ol_interactive = 'X_test_ol_interactive.npz'
npz_X_test_ol = 'X_test_ol.npz'

npz_X_train = 'X_train.npz'
npz_X_test = 'X_test.npz'

# 提交
csv_submission = 'submission.csv'

