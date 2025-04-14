from src.models.generador_modelos_v3 import HybridModel
from src.transform.procesador_datos import DataManager
from pprint import pprint
from tensorflow.keras.optimizers import Adam # type: ignore
from src.utils.metrics import rmsle

data_manager = DataManager()
model = HybridModel(data_manager.generate_metadata())
x_train, y_train, x_val, y_val = data_manager.build_input_data(load=False)
optimizer = Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='mse', metrics=[rmsle])
model.summary()
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=1024, shuffle=True)