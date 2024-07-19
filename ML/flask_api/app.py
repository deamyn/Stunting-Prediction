from flask import Flask, request, jsonify
from flask_restful import Api, Resource
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

app = Flask(__name__)
api = Api(app)

# Load machine learning model
model = joblib.load('../model_balita_rbf.pkl')
class PredictStunting(Resource):
    def post(self):
        try:
            # Ambil data JSON dari request
            data = request.get_json()

            # Ekstrak fitur yang dibutuhkan untuk prediksi
            nama_balita = data['nama_balita']
            tgl_lahir_balita = data['tgl_lahir_balita']
            jenis_kelamin = data['jenis_kelamin_balita']
            berat_badan = data['berat_badan']
            panjang_badan = data['panjang_badan']

            # Validasi data input
            if not (40 <= panjang_badan <= 200):
                return jsonify({'error': 'Tinggi badan tidak valid. Harus antara 40 dan 200 cm.'})
            if not (1 <= berat_badan <= 200):
                return jsonify({'error': 'Berat badan tidak valid. Harus antara 1 dan 200 kg.'})

            # Konversi jenis_kelamin ke numerik
            jenis_kelamin_perempuan = 0 if jenis_kelamin.lower() == 'perempuan' else 1

            # Hitung umur dalam bulan
            birth_date = datetime.strptime(tgl_lahir_balita, '%Y-%m-%d')
            current_date = datetime.now()
            age_in_months = (current_date.year - birth_date.year) * 12 + current_date.month - birth_date.month

            # Lakukan prediksi menggunakan model machine learning
            features = pd.DataFrame([{
                'Umur (bulan)': age_in_months,
                'Tinggi Badan (cm)': panjang_badan,
                'Jenis Kelamin_perempuan': jenis_kelamin_perempuan
            }])
            prediction = model.predict(features)

            # Buat response JSON dengan hasil prediksi
            result = {
                'nama_balita': nama_balita,
                'hasil_prediksi': prediction[0]
            }

            return jsonify(result)

        except Exception as e:
            return jsonify({'error': str(e)})

api.add_resource(PredictStunting, '/prediksistunting')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)