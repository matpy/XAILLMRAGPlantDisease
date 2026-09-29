from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Load the saved model
model = load_model('best_model.h5')
img_width, img_height = 150, 150

# Evaluate the model on the test dataset
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    'plants/test',
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='binary'
)



test_loss, test_acc = model.evaluate(test_generator)
print(f'Test accuracy: {test_acc}')
