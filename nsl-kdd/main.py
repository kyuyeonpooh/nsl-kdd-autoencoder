
from nsl_kdd import *

print('NSL-KDD dataset preprocessor')
service_list = get_service_list(dirname='input', filename='service.txt')
flag_list = get_flag_list(dirname='input', filename='flag.txt')

# training data
df_train = get_data_frame(dirname='dataset/train')
df_train = to_numeric(df_train, service_list, flag_list)
to_machine_readable(df_train, service_list, flag_list)
print('Train data preprocess finished!')

# test data
df_test = get_data_frame(dirname='dataset/test')
df_test = to_numeric(df_test, service_list, flag_list, test=True)
to_machine_readable(df_test, service_list, flag_list, test=True)
print('Test data preprocess finished!')
