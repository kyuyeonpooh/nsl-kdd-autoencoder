from nslkdd import *

print('NSL-KDD dataset preprocessor')
service_list = get_service_list(dirname='data', filename='service.txt')
flag_list = get_flag_list(dirname='data', filename='flag.txt')

# training data
df_train = get_data_frame(dirname='train')
df_train = to_numeric(df_train, service_list, flag_list)
to_machine_readable(df_train, service_list, flag_list)
print('Train data preprocess finished!', end='\n\n')

# training data with attack
df_train_atk = get_data_frame(dirname='train')
df_train_atk = to_numeric(df_train_atk, service_list, flag_list, attack=True)
to_machine_readable(df_train_atk, service_list, flag_list, attack=True)
print('Train data with attack preprocess finished!', end='\n\n')

# test data
df_test = get_data_frame(dirname='test')
df_test = to_numeric(df_test, service_list, flag_list, test=True)
to_machine_readable(df_test, service_list, flag_list, test=True)
print('Test data preprocess finished!', end='\n\n')
