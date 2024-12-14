import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import matplotlib.pyplot as plt

external_temp = ctrl.Antecedent(np.arange(-10, 51, 1), 'external_temp')
interior_temp = ctrl.Antecedent(np.arange(10, 31, 1), 'interior_temp')
fan_strength = ctrl.Consequent(np.arange(0, 11, 1), 'fan_strength')
ac_setting = ctrl.Consequent(np.arange(16, 31, 1), 'ac_setting')

# temp zewnętrzna
external_temp['cold'] = fuzz.trapmf(external_temp.universe, [-10, -10, 0, 10])
external_temp['moderate'] = fuzz.trimf(external_temp.universe, [5, 20, 35])
external_temp['hot'] = fuzz.trapmf(external_temp.universe, [30, 40, 50, 50])

# temp wewnętrzna 
interior_temp['low'] = fuzz.trapmf(interior_temp.universe, [10, 10, 15, 20])
interior_temp['comfortable'] = fuzz.trimf(interior_temp.universe, [18, 22, 26])
interior_temp['high'] = fuzz.trapmf(interior_temp.universe, [24, 28, 30, 30])

# moc wiatraków
fan_strength['low'] = fuzz.trimf(fan_strength.universe, [0, 0, 5])
fan_strength['medium'] = fuzz.trimf(fan_strength.universe, [3, 5, 8])
fan_strength['high'] = fuzz.trimf(fan_strength.universe, [7, 10, 10])

# ustawienia
ac_setting['cool'] = fuzz.trapmf(ac_setting.universe, [16, 16, 20, 24])
ac_setting['moderate'] = fuzz.trimf(ac_setting.universe, [22, 24, 26])
ac_setting['warm'] = fuzz.trapmf(ac_setting.universe, [25, 28, 30, 30])

# rozmyte zasady
rule1 = ctrl.Rule(external_temp['cold'] & interior_temp['low'], (fan_strength['low'], ac_setting['warm']))
rule2 = ctrl.Rule(external_temp['moderate'] & interior_temp['comfortable'], (fan_strength['medium'], ac_setting['moderate']))
rule3 = ctrl.Rule(external_temp['hot'] & interior_temp['high'], (fan_strength['high'], ac_setting['cool']))
rule4 = ctrl.Rule(external_temp['hot'] & interior_temp['comfortable'], (fan_strength['high'], ac_setting['moderate']))
rule5 = ctrl.Rule(external_temp['cold'] & interior_temp['comfortable'], (fan_strength['medium'], ac_setting['warm']))
rule6 = ctrl.Rule(external_temp['moderate'] & interior_temp['low'], (fan_strength['low'], ac_setting['moderate']))

# system kontroli
ac_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
ac_simulation = ctrl.ControlSystemSimulation(ac_control)

# symulacja
def simulate_ac_control(external, interior):
    try:
        ac_simulation.input['external_temp'] = external
        ac_simulation.input['interior_temp'] = interior
        ac_simulation.compute()

        fan = ac_simulation.output.get('fan_strength', None)
        temp = ac_simulation.output.get('ac_setting', None)

        if fan is None or temp is None:
            raise KeyError("Missing output calculation")

        print(f"External Temp: {external}, Interior Temp: {interior} -> Fan Strength: {fan:.2f}, AC Setting: {temp:.2f}")
        return fan, temp
    except Exception as e:
        print(f"Error with inputs External Temp: {external}, Interior Temp: {interior}. Details: {e}")
        return None, None

# test
scenarios = [(-5, 15), (20, 22), (35, 28), (10, 18), (45, 25)]
results = []
for external, interior in scenarios:
    results.append(simulate_ac_control(external, interior))

external_temp.view()
interior_temp.view()
fan_strength.view()
ac_setting.view()
plt.show()

