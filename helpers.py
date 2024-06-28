import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


class DietEnv:
    def __init__(self, df_sarapan, df_camilan, df_makan_siang, df_makan_malam, energy_breakdown, daily_protein, daily_carbs, daily_fats):
        self.df_sarapan = df_sarapan
        self.df_camilan = df_camilan
        self.df_makan_siang = df_makan_siang
        self.df_makan_malam = df_makan_malam
        self.energy_breakdown = energy_breakdown
        self.daily_protein = daily_protein
        self.daily_carbs = daily_carbs
        self.daily_fats = daily_fats
        self.reset()

    def reset(self):
        self.current_step = 0
        self.total_energy = 0
        self.total_protein = 0
        self.total_carbs = 0
        self.total_fats = 0
        self.total_rating = 0
        self.selected_snacks = set()  # Reset camilan yang sudah terpilih
        self.current_meal_name = None
        return self._next_observation()

    def _next_observation(self):
        return np.array([
            self.total_energy,
            self.total_protein,
            self.total_carbs,
            self.total_fats,
            self.total_rating
        ])

    def _take_action(self, action):
        df = self._get_current_dataframe()
        if df is None or len(df) == 0:
            print("DataFrame kosong atau tidak ada data yang memenuhi kriteria.")
            return

        current_meal = list(self.energy_breakdown.keys())[self.current_step]
        max_energy = self.energy_breakdown[current_meal]
        max_protein = max_energy * 0.12 / 4
        max_carbs = max_energy * 0.68 / 4
        max_fats = max_energy * 0.24 / 9
        max_fats_allowed = 10  # Maksimal lemak dalam gram

        if self.current_step in [1, 3, 5]:  # Camilan
            valid_snacks = []
            for i, food in df.iterrows():
                if (food['kalori'] <= max_energy and
                    food['protein'] <= max_protein and
                    food['karbo'] <= max_carbs and
                    food['lemak'] <= max_fats and
                    food['lemak'] <= max_fats_allowed and
                    i not in self.selected_snacks):
                    valid_snacks.append((food, food['nama_resep']))

            if valid_snacks:
                # Pilih camilan dengan jarak terdekat
                distances = [(np.sqrt(
                                (food['kalori'] - max_energy)**2 +
                                (food['protein'] - max_protein)**2 +
                                (food['karbo'] - max_carbs)**2 +
                                (food['lemak'] - max_fats)**2
                            ), food, name) for food, name in valid_snacks]
                distances.sort(key=lambda x: x[0])
                best_food_name = distances[0][2]
                selected_food = df[df['nama_resep'] == best_food_name].iloc[0]
                self.selected_snacks.add(best_food_name)
            else:
                selected_food = None
        else:
            valid_foods = []
            for i, food in df.iterrows():
                if (food['kalori'] <= max_energy and
                    food['protein'] <= max_protein and
                    food['karbo'] <= max_carbs and
                    food['lemak'] <= max_fats and
                    food['lemak'] <= max_fats_allowed):
                    valid_foods.append((food, food['nama_resep']))

            if valid_foods:
                distances = [(np.sqrt(
                                (food['kalori'] - max_energy)**2 +
                                (food['protein'] - max_protein)**2 +
                                (food['karbo'] - max_carbs)**2 +
                                (food['lemak'] - max_fats)**2
                            ), food, name) for food, name in valid_foods]
                distances.sort(key=lambda x: x[0])
                best_food_name = distances[0][2]
                selected_food = df[df['nama_resep'] == best_food_name].iloc[0]
            else:
                selected_food = None

        if selected_food is not None:
            self.total_energy += selected_food['kalori']
            self.total_protein += selected_food['protein']
            self.total_carbs += selected_food['karbo']
            self.total_fats += selected_food['lemak']
            self.total_rating += selected_food['rating']
            self.current_meal_name = best_food_name
        else:
            self.total_rating -= 1
            self.current_meal_name = None

    def _get_current_dataframe(self):
        current_meal = list(self.energy_breakdown.keys())[self.current_step]
        max_energy = self.energy_breakdown[current_meal]
        max_fats_allowed = 10

        if self.current_step == 0:
            return self.df_sarapan[(self.df_sarapan['kalori'] <= max_energy) & (self.df_sarapan['lemak'] <= max_fats_allowed)]
        elif self.current_step in [1, 3, 5]:
            return self.df_camilan[(self.df_camilan['kalori'] <= max_energy) & (self.df_camilan['lemak'] <= max_fats_allowed) & (~self.df_camilan.index.isin(self.selected_snacks))]
        elif self.current_step == 2:
            return self.df_makan_siang[(self.df_makan_siang['kalori'] <= max_energy) & (self.df_makan_siang['lemak'] <= max_fats_allowed)]
        elif self.current_step == 4:
            return self.df_makan_malam[(self.df_makan_malam['kalori'] <= max_energy) & (self.df_makan_malam['lemak'] <= max_fats_allowed)]

        return None

    def step(self, action):
        self._take_action(action)
        self.current_step += 1

        reward = self._calculate_reward()
        done = self.current_step >= len(self.energy_breakdown)

        obs = self._next_observation()
        return obs, reward, done, {}

    def _calculate_reward(self):
        current_meal = list(self.energy_breakdown.keys())[self.current_step - 1]

        # Ambil target energi, protein, karbohidrat, dan lemak untuk setiap waktu makan
        target_energy_this_meal = self.energy_breakdown[current_meal]
        target_protein_this_meal = self.daily_protein / len(self.energy_breakdown)
        target_carbs_this_meal = self.daily_carbs / len(self.energy_breakdown)
        target_fats_this_meal = self.daily_fats / len(self.energy_breakdown)

        energy_reward = 1 - abs((self.total_energy - target_energy_this_meal) / target_energy_this_meal)
        protein_reward = 1 - abs((self.total_protein - target_protein_this_meal) / target_protein_this_meal)
        carbs_reward = 1 - abs((self.total_carbs - target_carbs_this_meal) / target_carbs_this_meal)
        fats_reward = 1 - abs((self.total_fats - target_fats_this_meal) / target_fats_this_meal)
        rating_reward = self.total_rating / (self.current_step + 1)

        euclidean_distance = self._calculate_euclidean_distance()
        distance_reward = 1 / (1 + euclidean_distance)

        normalized_reward = (energy_reward + protein_reward + carbs_reward + fats_reward + rating_reward + distance_reward) / 6

        return normalized_reward

    def _calculate_euclidean_distance(self):
        current_meal = list(self.energy_breakdown.keys())[self.current_step - 1]

        max_energy = self.energy_breakdown[current_meal]
        max_protein = max_energy * 0.12 / 4
        max_carbs = max_energy * 0.68 / 4
        max_fats = max_energy * 0.24 / 9

        distance = np.sqrt(
            (self.total_energy - max_energy)**2 +
            (self.total_protein - max_protein)**2 +
            (self.total_carbs - max_carbs)**2 +
            (self.total_fats - max_fats)**2
        )

        return distance

    def action_space_sample(self):
        df = self._get_current_dataframe()
        if df is None or len(df) == 0:
            return 0
        return random.randint(0, len(df) - 1)

    def action_space_size(self):
        max_fats_allowed = 10  # Maksimal lemak dalam gram

        df_sarapan_filtered = self.df_sarapan[(self.df_sarapan['kalori'] <= self.energy_breakdown['sarapan']) & (self.df_sarapan['lemak'] <= max_fats_allowed)]
        df_camilan_filtered = self.df_camilan[(self.df_camilan['kalori'] <= self.energy_breakdown['camilan_pagi']) & (self.df_camilan['lemak'] <= max_fats_allowed) & (~self.df_camilan.index.isin(self.selected_snacks))]
        df_makan_siang_filtered = self.df_makan_siang[(self.df_makan_siang['kalori'] <= self.energy_breakdown['makan_siang']) & (self.df_makan_siang['lemak'] <= max_fats_allowed)]
        df_makan_malam_filtered = self.df_makan_malam[(self.df_makan_malam['kalori'] <= self.energy_breakdown['makan_malam']) & (self.df_makan_malam['lemak'] <= max_fats_allowed)]

        df_sizes = [
            len(df_sarapan_filtered),
            len(df_camilan_filtered),
            len(df_makan_siang_filtered),
            len(df_makan_malam_filtered)
        ]
        return max(df_sizes)


def calculate_bmr_and_daily_energy(gender, weight, height, age, activity_level):
    if gender == 'male':
        bmr = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
    else:
        bmr = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)

    activity_multipliers = {
        'sedentary': 1.2,
        'light': 1.375,
        'moderate': 1.55,
        'active': 1.725,
        'very active': 1.9
    }

    daily_energy = bmr * activity_multipliers[activity_level] # Diet: reduce 500 kcal, nasi: 37 gram 48 kalori * 3 = 144 kalori, buah 50 kalori * 6 = 300, sayur 50 kalori * 3
    return bmr, daily_energy

def calculate_nutritional_needs(daily_energy):
    energy_breakdown = {
        'sarapan': daily_energy * 0.20,
        'camilan_pagi': daily_energy * 0.10,
        'makan_siang': daily_energy * 0.25,
        'camilan_siang': daily_energy * 0.10,
        'makan_malam': daily_energy * 0.25,
        'camilan_malam': daily_energy * 0.10
    }

    daily_protein = daily_energy * 0.12 / 4
    daily_carbs = daily_energy * 0.68 / 4
    daily_fats = daily_energy * 0.24 / 9

    return energy_breakdown, daily_protein, daily_carbs, daily_fats

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, name):
        self.model.save(name)

    def load(self, name):
        self.model = load_model(name, custom_objects={'mse': 'mse'})

def recommend_diet(agent, env, state_size):
    state = env.reset()
    recommendations = []
    for time in range(len(env.energy_breakdown)):
        action = agent.act(np.reshape(state, [1, state_size]))
        next_state, reward, done, _ = env.step(action)
        state = next_state
        recommendations.append(action)
    return recommendations
