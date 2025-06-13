# Versão melhorada V_2.1 para 100 sorteios
#
# Autor: Sergio Tabarez
# Data: junho 2025
# Descrição: Modelo de Machine Learning Híbrida previsão de dezenas de loteria usando Transformer e LSTM.
# Obs: Treinamento com um pequeno subconjunto de dados devido a demanda computacional
#
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Concatenate
import matplotlib.pyplot as plt

# Carregar e pré-processar dados
file_path = '/content/Lotofacil-original100.xlsx'
df = pd.read_excel(file_path)
dezena_columns = [f'Bola{i}' for i in range(1, 16)]
data = df[dezena_columns].values

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

sequence_length = 30
sequences = []
next_draw = []

for i in range(len(scaled_data) - sequence_length):
    seq = scaled_data[i : i + sequence_length]
    target = scaled_data[i + sequence_length]
    sequences.append(seq)
    next_draw.append(target)

X = np.array(sequences)
y = np.array(next_draw)

train_size = int(len(X) * 0.9)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Bloco Transformer 
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

        # Camadas de atenção
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)

        # Rede feed-forward
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim)
        ])

        # Normalização de camada
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

        # Dropout
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def build(self, input_shape):
        # Projeção para a dimensão embed_dim
        self.dense_proj = Dense(self.embed_dim)
        super().build(input_shape)

    def call(self, inputs):
        # Projeta a entrada para a dimensão correta
        projected_inputs = self.dense_proj(inputs)

        # Camada de atenção
        attn_output = self.att(projected_inputs, projected_inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(projected_inputs + attn_output)

        # Rede feed-forward
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)

        return self.layernorm2(out1 + ffn_output)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.embed_dim)

# Modelo Híbrido Transformer-LSTM
def build_hybrid_model(input_shape):
    inputs = Input(shape=input_shape)

    # Transformer Encoder
    transformer_output = TransformerBlock(embed_dim=64, num_heads=6, ff_dim=128)(inputs)
    transformer_pooled = GlobalAveragePooling1D()(transformer_output)

    # Ramo LSTM
    lstm_output = LSTM(64, activation='relu', return_sequences=False)(inputs)

    # Combinar ambos os ramos
    combined = Concatenate()([transformer_pooled, lstm_output])

    # Camadas Dense
    x = Dense(128, activation='relu')(combined)
    x = Dropout(0.1)(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(15, activation='linear')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                 loss='mse',
                 metrics=['mae'])
    return model

# Construir e treinar o modelo
model = build_hybrid_model(input_shape=(X_train.shape[1], X_train.shape[2]))
model.summary()

epochs = 100
batch_size = 32

history = model.fit(X_train, y_train,
                   epochs=epochs,
                   batch_size=batch_size,
                   validation_data=(X_test, y_test),
                   verbose=1)

# Função de avaliação
def calcular_acertos(previsao, real):
    return len(set(previsao) & set(real))

total_acertos = 0
for i in range(len(X_test)):
    input_sequence = X_test[i].reshape((1, sequence_length, 15))
    predicted_normalized = model.predict(input_sequence, verbose=0)
    predicted_desnormalized = scaler.inverse_transform(predicted_normalized)[0]
    predicted_rounded = np.round(predicted_desnormalized).astype(int)
    predicted_in_range = [int(max(1, min(x, 25))) for x in predicted_rounded]
    predicted_unique = sorted(list(set(predicted_in_range)))

    # Garantir 15 números únicos
    while len(predicted_unique) < 15:
        missing = 15 - len(predicted_unique)
        candidates = [x for x in range(1, 26) if x not in predicted_unique]
        predicted_unique.extend(np.random.choice(candidates, min(missing, len(candidates)), replace=False))
        predicted_unique = sorted(list(set(predicted_unique)))

    real_normalized = y_test[i]
    real_desnormalized = scaler.inverse_transform(np.array([real_normalized]))[0]
    real_rounded = np.round(real_desnormalized).astype(int)
    real_in_range = [int(max(1, min(x, 25))) for x in real_rounded]
    real_unique = sorted(list(set(real_in_range)))

    acertos = calcular_acertos(predicted_unique, real_unique)
    total_acertos += acertos

media_acertos = total_acertos / len(X_test)
print(f"\nMédia de acertos no conjunto de teste: {media_acertos:.2f} dezenas.")

# Previsão para o próximo sorteio
last_sequence = data[-sequence_length:]
scaled_last_sequence = scaler.transform(last_sequence)
scaled_last_sequence = scaled_last_sequence.reshape((1, sequence_length, 15))
predicted_normalized = model.predict(scaled_last_sequence, verbose=0)
predicted_desnormalized = scaler.inverse_transform(predicted_normalized)[0]
predicted_rounded = np.round(predicted_desnormalized).astype(int)
predicted_in_range = [int(max(1, min(x, 25))) for x in predicted_rounded]
predicted_unique = sorted(list(set(predicted_in_range)))

# Garantir 15 números únicos na previsão
while len(predicted_unique) < 15:
    missing = 15 - len(predicted_unique)
    candidates = [x for x in range(1, 26) if x not in predicted_unique]
    predicted_unique.extend(np.random.choice(candidates, min(missing, len(candidates)), replace=False))
    predicted_unique = sorted(list(set(predicted_unique)))

print("\nPrevisão da Lotofácil (modelo Transformer-LSTM):", predicted_unique)
# Plotar gráfico de loss de treinamento e validação
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Loss de Treinamento')
plt.plot(history.history['val_loss'], label='Loss de Validação')
plt.title('Loss durante o Treinamento')
plt.xlabel('Épocas')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
