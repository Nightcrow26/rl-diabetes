import streamlit as st
import pandas as pd
from helpers import calculate_bmr_and_daily_energy, calculate_nutritional_needs, DietEnv, DQNAgent, recommend_diet

df_sarapan = pd.read_excel('sarapan.xlsx')
df_camilan = pd.read_excel('camilan.xlsx')
df_makan_siang = pd.read_excel('makan_siang.xlsx')
df_makan_malam = pd.read_excel('makan_malam.xlsx')
df_sayur = pd.read_excel('sayuran.xlsx')
df_buah = pd.read_excel('buah.xlsx')

# Input data pasien baru
st.title("Diet Recommendation System")

gender = st.selectbox("Gender", ['male', 'female'])
weight = st.number_input("Weight (kg)", 0)
height = st.number_input("Height (cm)", 0)
age = st.number_input("Age", 0)
activity_level = st.selectbox("Activity Level", ['sedentary', 'light', 'moderate', 'active', 'very active'])

if st.button("Calculate BMR and Energy Needs"):
    bmr, daily_energy = calculate_bmr_and_daily_energy(gender, weight, height, age, activity_level)
    daily_energy_reduced = daily_energy - 944  # Diet: reduce 500 kcal, nasi: 37 gram 48 kalori * 3 = 144 kalori, buah 50 kalori * 6 = 300, sayur 50 kalori
    energy_breakdown, daily_protein, daily_carbs, daily_fats = calculate_nutritional_needs(daily_energy_reduced)
    energy_breakdown_real, daily_protein_real, daily_carbs_real, daily_fats_real = calculate_nutritional_needs(daily_energy)

    # Menyaring df_sarapan
    max_fats_allowed = 10

    max_energy_sarapan = energy_breakdown['sarapan']
    df_sarapan_filtered = df_sarapan[(df_sarapan['kalori'] <= max_energy_sarapan) & (df_sarapan['lemak'] <= max_fats_allowed)]

    # Menyaring df_camilan
    max_energy_camilan = energy_breakdown['camilan_pagi']
    df_camilan_filtered = df_camilan[(df_camilan['kalori'] <= max_energy_camilan) & (df_camilan['lemak'] <= max_fats_allowed)]

    # Menyaring df_makan_siang
    max_energy_siang = energy_breakdown['makan_siang']
    df_makan_siang_filtered = df_makan_siang[(df_makan_siang['kalori'] <= max_energy_siang) & (df_makan_siang['lemak'] <= max_fats_allowed)]

    # Menyaring df_makan_malam
    max_energy_malam = energy_breakdown['makan_malam']
    df_makan_malam_filtered = df_makan_malam[(df_makan_malam['kalori'] <= max_energy_malam) & (df_makan_malam['lemak'] <= max_fats_allowed)]

    st.write(f'BMR: {bmr}')
    st.write(f'Daily Energy: {daily_energy}')
    st.write(f'Daily Protein: {daily_protein_real}')
    st.write(f'Daily Carbs: {daily_carbs_real}')
    st.write(f'Daily Fats: {daily_fats_real}')
    st.write(f'Energy Breakdown: {energy_breakdown_real}')
    st.write("----------")

    # Initialize the environment with new patient's data
    new_env = DietEnv(df_sarapan_filtered, df_camilan_filtered, df_makan_siang_filtered, df_makan_malam_filtered, energy_breakdown, daily_protein, daily_carbs, daily_fats)

    # Load the trained model
    state_size = 5
    action_size = new_env.action_space_size()
    agent = DQNAgent(state_size=state_size, action_size=action_size)  # Adjust action_size according to your environment
    agent.load("dqn_model.hdf5")

    # Generate diet recommendations
    recommendations = recommend_diet(agent, new_env, state_size)

    meal_times = ['Sarapan', 'Camilan Pagi', 'Makan Siang', 'Camilan Siang', 'Makan Malam', 'Camilan Malam']
    for i, rec in enumerate(recommendations):
        if i in [0, 2, 4]:
            df = new_env.df_sarapan if i == 0 else new_env.df_makan_siang if i == 2 else new_env.df_makan_malam
            rec = rec % len(df)
            food_item = df.iloc[rec]
            st.write(f"{meal_times[i]}: {food_item['nama_resep']} - Kalori: {food_item['kalori']}, Protein: {food_item['protein']}, Karbo: {food_item['karbo']}, Lemak: {food_item['lemak']}, Rating: {food_item['rating']}")

            # Add fruit and vegetable recommendations
            fruit_item = df_buah[df_buah['kalori'] <= 50].sample(1).iloc[0]
            veg_item = df_sayur[df_sayur['kalori'] <= 50].sample(1).iloc[0]
            st.write(f"Buah: {fruit_item['nama_buah']} - Kalori: {fruit_item['kalori']}, Protein: {fruit_item['protein']}, Karbo: {fruit_item['karbo']}, Lemak: {fruit_item['lemak']}")
            st.write(f"Sayur: {veg_item['nama_resep']} - Kalori: {veg_item['kalori']}, Protein: {veg_item['protein']}, Karbo: {veg_item['karbo']}, Lemak: {veg_item['lemak']}")
            st.write("----------")
        else:
            df = new_env.df_camilan
            rec = rec % len(df)
            food_item = df.iloc[rec]
            st.write(f"{meal_times[i]}: {food_item['nama_resep']} - Kalori: {food_item['kalori']}, Protein: {food_item['protein']}, Karbo: {food_item['karbo']}, Lemak: {food_item['lemak']}, Rating: {food_item['rating']}")

            # Add fruit recommendation
            fruit_item = df_buah[df_buah['kalori'] <= 50].sample(1).iloc[0]
            st.write(f"Buah: {fruit_item['nama_buah']} - Kalori: {fruit_item['kalori']}, Protein: {fruit_item['protein']}, Karbo: {fruit_item['karbo']}, Lemak: {fruit_item['lemak']}")
            st.write("----------")
