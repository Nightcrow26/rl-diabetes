import streamlit as st
import pandas as pd
from helpers import calculate_bmr_and_daily_energy, calculate_nutritional_needs, DietEnv, DQNAgent, recommend_diet

df_sarapan = pd.read_excel('sarapan.xlsx')
df_camilan = pd.read_excel('camilan.xlsx')
df_makan_siang = pd.read_excel('makan_siang.xlsx')
df_makan_malam = pd.read_excel('makan_malam.xlsx')
df_sayur = pd.read_excel('sayuran.xlsx')
df_buah = pd.read_excel('buah.xlsx')

df_resep_camilan = pd.read_excel('data resep.xlsx', sheet_name='camilan_resep_bahan')
df_resep_sayuran = pd.read_excel('data resep.xlsx', sheet_name='sayuran_resep_bahan')
df_resep_sarapan = pd.read_excel('data resep.xlsx', sheet_name='sarapan_resep_bahan')
df_resep_siang = pd.read_excel('data resep.xlsx', sheet_name='makan_siang_resep_bahan')
df_resep_malam = pd.read_excel('data resep.xlsx', sheet_name='makan_malam_resep_bahan')

# Example CSS to widen tables
st.write(
    f'<style>div.row-widget.stRadio > div{{width: 90%}}</style>',
    unsafe_allow_html=True,
)

# Input data pasien baru
st.title("Diet Recommendation System")

gender = st.selectbox("Gender", ['male', 'female'])
weight = st.number_input("Weight (kg)", 0)
height = st.number_input("Height (cm)", 0)
age = st.number_input("Age", 0)
activity_level = st.selectbox("Activity Level", ['sedentary', 'light', 'moderate', 'active', 'very active'])

if st.button("Calculate BMR and Energy Needs"):
    bmr, daily_energy = calculate_bmr_and_daily_energy(gender, weight, height, age, activity_level)
    daily_energy_reduced = daily_energy - 1142  # Diet: reduce 500 kcal, nasi: 50 gram 64 kalori * 3 = 192 kalori, buah 50 kalori * 6 = 300, sayur 50 kalori * 3 = 150
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

    st.write(f'BMR: {bmr:.2f}')
    st.write(f'Daily Energy: {daily_energy:.2f}')
    st.write(f'Daily Protein: {daily_protein_real:.2f}')
    st.write(f'Daily Carbs: {daily_carbs_real:.2f}')
    st.write(f'Daily Fats: {daily_fats_real:.2f}')
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

    #ditambah kalori nasi
    total_calories = 192
    total_protein = 3.99
    total_carbs = 41.7
    total_fats = 0.42

    for i, rec in enumerate(recommendations):
        if i in [0, 2, 4]:
            df = new_env.df_sarapan if i == 0 else new_env.df_makan_siang if i == 2 else new_env.df_makan_malam
            rec = rec % len(df)
            food_item = df.iloc[rec]
            resep_df = df_resep_sarapan if i == 0 else df_resep_siang if i == 2 else df_resep_malam

            if food_item['nama_resep'] in resep_df['nama_resep'].values:
                bahan = resep_df[resep_df['nama_resep'] == food_item['nama_resep']]['bahan_bahan'].values[0]
                cara_masak = resep_df[resep_df['nama_resep'] == food_item['nama_resep']]['cara_membuat'].values[0]
            else:
                bahan = '-'
                cara_masak = '-'

            sayur_item = df_sayur[df_sayur['kalori'] <= 50].sample(1).iloc[0]
            buah_item = df_buah[df_buah['kalori'] <= 50].sample(1).iloc[0]

            st.write(f"### {meal_times[i]}")
            df_meal = pd.DataFrame({
                'Jenis': ['Makanan Utama', 'Sayur', 'Buah'],
                'Nama': [food_item['nama_resep'], sayur_item['nama_resep'], buah_item['nama_buah']],
                'Kalori': [float(food_item['kalori']), float(sayur_item['kalori']), float(buah_item['kalori'])],
                'Protein (gr)': [float(food_item['protein']), float(sayur_item['protein']), float(buah_item['protein'])],
                'Karbo (gr)': [float(food_item['karbo']), float(sayur_item['karbo']), float(buah_item['karbo'])],
                'Lemak (gr)': [float(food_item['lemak']), float(sayur_item['lemak']), float(buah_item['lemak'])],
                'Bahan': [bahan, df_resep_sayuran[df_resep_sayuran['nama_resep'] == sayur_item['nama_resep']]['bahan_bahan'].values[0], '-'],
                'Cara Masak': [cara_masak, df_resep_sayuran[df_resep_sayuran['nama_resep'] == sayur_item['nama_resep']]['cara_membuat'].values[0], '-']
            })

            # Round values to two decimal places
            df_meal[['Kalori', 'Protein (gr)', 'Karbo (gr)', 'Lemak (gr)']] = df_meal[['Kalori', 'Protein (gr)', 'Karbo (gr)', 'Lemak (gr)']].astype(float).round(2)

            st.dataframe(df_meal)
            meal_calories = df_meal['Kalori'].sum() + 64  # kalori nasi 50 gram
            total_calories += meal_calories
            total_protein += df_meal['Protein (gr)'].sum()
            total_carbs += df_meal['Karbo (gr)'].sum()
            total_fats += df_meal['Lemak (gr)'].sum()
            st.write(f"##### Total Kalori {meal_times[i]}: {meal_calories:.2f}")
            st.write("----------")

        else:
            df = new_env.df_camilan
            rec = rec % len(df)
            food_item = df.iloc[rec]
            resep_df = df_resep_camilan

            if food_item['nama_resep'] in resep_df['nama_resep'].values:
                bahan = resep_df[resep_df['nama_resep'] == food_item['nama_resep']]['bahan_bahan'].values[0]
                cara_masak = resep_df[resep_df['nama_resep'] == food_item['nama_resep']]['cara_membuat'].values[0]
            else:
                bahan = '-'
                cara_masak = '-'

            buah_item = df_buah[df_buah['kalori'] <= 50].sample(1).iloc[0]

            st.write(f"### {meal_times[i]}")
            df_snack = pd.DataFrame({
                'Jenis': ['Camilan', 'Buah'],
                'Nama': [food_item['nama_resep'], buah_item['nama_buah']],
                'Kalori': [float(food_item['kalori']), float(buah_item['kalori'])],
                'Protein (gr)': [float(food_item['protein']), float(buah_item['protein'])],
                'Karbo (gr)': [float(food_item['karbo']), float(buah_item['karbo'])],
                'Lemak (gr)': [float(food_item['lemak']), float(buah_item['lemak'])],
                'Bahan': [bahan, '-'],
                'Cara Masak': [cara_masak, '-']
            })

            # Round values to two decimal places
            df_snack[['Kalori', 'Protein (gr)', 'Karbo (gr)', 'Lemak (gr)']] = df_snack[['Kalori', 'Protein (gr)', 'Karbo (gr)', 'Lemak (gr)']].astype(float).round(2)

            st.dataframe(df_snack)
            snack_calories = df_snack['Kalori'].sum()
            total_calories += snack_calories
            total_protein += df_snack['Protein (gr)'].sum()
            total_carbs += df_snack['Karbo (gr)'].sum()
            total_fats += df_snack['Lemak (gr)'].sum()
            st.write(f"##### Total Kalori {meal_times[i]}: {snack_calories:.2f}")
            st.write("----------")

    st.write("### Total Nutrisi Harian")
    st.write(f"Total Kalori: {total_calories:.2f}")
    st.write(f"Total Protein: {total_protein:.2f} gram")
    st.write(f"Total Karbohidrat: {total_carbs:.2f} gram")
    st.write(f"Total Lemak: {total_fats:.2f} gram")
