from preprocess import *

print('NSL-KDD dataset preprocessor', end='\n\n')
service_list = get_service_list(dirname='list', filename='service.txt')
flag_list = get_flag_list(dirname='list', filename='flag.txt')

# 100% normal training data
df_train = get_data_frame(dirname='dataset', filename='KDDTrain+.txt')
df_train = to_numeric(df_train, service_list, flag_list)
to_machine_readable(df_train, service_list, flag_list)
print('Train data preprocess finished!', end='\n\n')

# 1% attack mixed training data
df_train_atk = get_data_frame(dirname='dataset', filename='KDDTrain+.txt')
df_train_atk = to_numeric(df_train_atk, service_list, flag_list, attack=True)
to_machine_readable(df_train_atk, service_list, flag_list, attack=True)
print('Train data with attack preprocess finished!', end='\n\n')

# test data
df_test = get_data_frame(dirname='dataset', filename='KDDTest+.txt')
df_test = to_numeric(df_test, service_list, flag_list, test=True)
to_machine_readable(df_test, service_list, flag_list, test=True)
print('Test data preprocess finished!', end='\n\n')
