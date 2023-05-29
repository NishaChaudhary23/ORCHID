#Importing necessary packages
import pandas as pd
import tensorflow as tf
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
import os
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications import DenseNet169
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import NASNetLarge
from tensorflow.keras.applications import NASNetMobile
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import Xception
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.optimizers import RMSprop

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

project_path = '/storage/bic/data/oscc/project_1' # ADD YOUR PROJECT PATH HERE
data_path = f'{project_path}/train' # ADD YOUR DATA PATH HERE

# Declaring Model Type
model_type = 'NASNetMobile'

# ImageDataGenerator
# color images
datagen_train = ImageDataGenerator(rescale = 1.0/255.0,validation_split=0.2)
# Training Data
train_generator = datagen_train.flow_from_directory(
	data_path,
	target_size=(300, 300),
	batch_size=32 if model_type not in ['NASNetLarge','NASNetMobile'] else 8,
	class_mode='categorical',
	subset = 'training')
#Validation Data
valid_generator = datagen_train.flow_from_directory(
	data_path,
	target_size=(300, 300),
	batch_size=32 if model_type not in ['NASNetLarge','NASNetMobile'] else 8,
	class_mode='categorical',
	subset = 'validation',
	shuffle=False)


# Creating the model
if model_type == 'DenseNet121':
	densenet = DenseNet121(
	weights='imagenet',
	include_top=False,
	input_shape=(300,300,3)
	)
	for layer in densenet.layers:
			layer.trainable = True
	x = layers.Flatten()(densenet.output)
	x = layers.Dense(1024, activation = 'relu')(x)
	x = layers.Dropout(0.2)(x)
	x = layers.Dense(3, activation = 'softmax')(x)
	model = Model(densenet.input, x)
	model.compile(optimizer = RMSprop(learning_rate = 0.00001), loss = 'categorical_crossentropy', metrics = ['acc'])

if model_type == 'DenseNet169':
	densenet = DenseNet169(
			weights='imagenet',
			include_top=False,
			input_shape=(300,300,3)
			)
	for layer in densenet.layers:
			layer.trainable = True
	x = layers.Flatten()(densenet.output)
	x = layers.Dense(1024, activation = 'relu')(x)
	x = layers.Dropout(0.2)(x)
	x = layers.Dense(3, activation = 'softmax')(x)
	model = Model(densenet.input, x)
	model.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])

if model_type == 'DenseNet201':
	densenet = DenseNet201(
			weights='imagenet',
			include_top=False,
			input_shape=(300,300,3)
			)
	for layer in densenet.layers:
			layer.trainable = True
	x = layers.Flatten()(densenet.output)
	x = layers.Dense(1024, activation = 'relu')(x)
	x = layers.Dropout(0.2)(x)
	x = layers.Dense(3, activation = 'softmax')(x)
	model = Model(densenet.input, x)
	model.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])


if model_type == 'InceptionResNetV2':
	inception = InceptionResNetV2(
			weights='imagenet',
			include_top=False,
			input_shape=(300,300,3)
			)
	for layer in inception.layers:
			layer.trainable = True
	x = layers.Flatten()(inception.output)
	x = layers.Dense(1024, activation = 'relu')(x)
	x = layers.Dropout(0.2)(x)
	x = layers.Dense(3, activation = 'softmax')(x)
	model = Model(inception.input, x)
	model.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])

if model_type == 'InceptionV3':
	inception = InceptionV3(
			weights='imagenet',
			include_top=False,
			input_shape=(300,300,3)
			)
	for layer in inception.layers:
			layer.trainable = True
	x = layers.Flatten()(inception.output)
	x = layers.Dense(1024, activation = 'relu')(x)
	x = layers.Dropout(0.2)(x)
	x = layers.Dense(3, activation = 'softmax')(x)
	model = Model(inception.input, x)
	model.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])

if model_type == 'MobileNetV3Large':
	mobilenet = MobileNetV3Large(
			weights='imagenet',
			include_top=False,
			input_shape=(300,300,3)
			)
	for layer in mobilenet.layers:
			layer.trainable = True
	x = layers.Flatten()(mobilenet.output)
	x = layers.Dense(1024, activation = 'relu')(x)
	x = layers.Dropout(0.2)(x)
	x = layers.Dense(3, activation = 'softmax')(x)
	model = Model(mobilenet.input, x)
	model.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])

if model_type == 'NASNetLarge':
	nasnet = NASNetLarge(
			weights='imagenet',
			include_top=False,
			input_shape=(331,331,3)
			)
	for layer in nasnet.layers:
			layer.trainable = True
	x = layers.Flatten()(nasnet.output)
	x = layers.Dense(1024, activation = 'relu')(x)
	x = layers.Dropout(0.2)(x)
	x = layers.Dense(3, activation = 'softmax')(x)
	model = Model(nasnet.input, x)
	model.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])        

if model_type == 'NASNetMobile':
	nasnet = NASNetMobile(
			include_top=False,
			input_shape=(224,224,3)
			)
	for layer in nasnet.layers:
			layer.trainable = True
	x = layers.Flatten()(nasnet.output)
	x = layers.Dense(1024, activation = 'relu')(x)
	x = layers.Dropout(0.2)(x)
	x = layers.Dense(3, activation = 'softmax')(x)
	model = Model(nasnet.input, x)
	model.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])

if model_type == 'VGG19':
	vgg19 = VGG19(
			weights='imagenet',
			include_top=False,
			input_shape=(300,300,3)
			)
	for layer in vgg19.layers:
			layer.trainable = True
	x = layers.Flatten()(vgg19.output)
	x = layers.Dense(1024, activation = 'relu')(x)
	x = layers.Dropout(0.2)(x)
	x = layers.Dense(3, activation = 'softmax')(x)
	model = Model(vgg19.input, x)
	model.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])

if model_type == 'Xception':
	xception = Xception(
			weights='imagenet',
			include_top=False,
			input_shape=(300,300,3)
			)
	for layer in xception.layers:
			layer.trainable = True
	x = layers.Flatten()(xception.output)
	x = layers.Dense(1024, activation = 'relu')(x)
	x = layers.Dropout(0.2)(x)
	x = layers.Dense(3, activation = 'softmax')(x)
	model = Model(xception.input, x)
	model.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])

if not os.path.exists(f'{project_path}/models/{model_type}'):
	os.makedirs(f'{project_path}/models/{model_type}')
# Model Summary
model.summary()

# Training the model
print("------------------------------------------")
print(f'Training the model {model_type}')
print("------------------------------------------")
history = model.fit(train_generator, validation_data = valid_generator, epochs=20)
print("------------------------------------------")
print(f'Training Complete')
print("------------------------------------------")

# Saving the model
model.save(f'{project_path}/models/{model_type}/{model_type}.h5')
print("------------------------------------------")
print(f'Model saved')
print("------------------------------------------")


#plotting the accuracy and loss
print("------------------------------------------")
print(f'Plotting and supplimentary data')
print("------------------------------------------")
plt.figure(figsize=(10, 10))
plt.plot(history.history['acc'], label='Training Accuracy')
plt.plot(history.history['val_acc'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend(['train', 'test'], loc='upper left')
plt.tight_layout()
plt.savefig(f'{project_path}/models/{model_type}/Accuracy.jpg')


# Saving Training History
hist_df = pd.DataFrame(history.history) 
# save to json:  
hist_json_file = f'{project_path}/models/{model_type}/history.json' 
with open(hist_json_file, mode='w') as f:
	hist_df.to_json(f)
# or save to csv: 
hist_csv_file = f'{project_path}/models/{model_type}/history.csv'
with open(hist_csv_file, mode='w') as f:
	hist_df.to_csv(f)
	

# Loading Model for Testing
loaded_model = load_model(f'{project_path}/models/{model_type}/{model_type}.h5')
outcomes = loaded_model.predict(valid_generator)
y_pred = np.argmax(outcomes, axis=1)

# Computing and saving the confusion matrix
# confusion matrix
confusion = confusion_matrix(valid_generator.classes, y_pred)
plt.figure(figsize=(10, 10))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig(f'{project_path}/models/{model_type}/Confusion_matrix.jpg')
conf_df = pd.DataFrame(confusion, index = ['normal','osmf','oscc'], columns = ['normal','osmf','oscc'])
conf_df.to_csv(f'{project_path}/models/{model_type}/Confusion_matrix.csv')



# Computing and saving the Classification Report
# classification report
target_names = ['normal','osmf','oscc']
report = classification_report(valid_generator.classes, y_pred, target_names=target_names, output_dict=True)
df = pd.DataFrame(report).transpose()
df.to_csv(f'{project_path}/models/{model_type}/Classification_report.csv')

print("------------------------------------------")
print(f'Supplimentary Data Saved')
print("------------------------------------------")
