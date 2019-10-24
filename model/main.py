from model import *
import itertools

losses = ['categorical_crossentropy', 'mean_absolute_error']
train_type = ['normal', 'normal_with_attack']  # 100% normal, 99% normal

models = []
for loss, train in itertools.product(losses, train_type):
    models.append('autoencoder_' + loss + '_' + train + '.h5')

print(models)

# # Train All
# for loss, train in itertools.product(losses, train_type):
#     print(loss, train)
#     autoencoder_mode(loss, train)
#     break


# Test All
for model in models:
    deploy(model)
    break
