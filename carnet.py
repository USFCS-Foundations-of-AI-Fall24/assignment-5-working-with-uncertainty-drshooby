from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from torch.fx.experimental.unification import variables

car_model = BayesianNetwork(
    [
        ("Battery", "Radio"),
        ("Battery", "Ignition"),
        ("Ignition","Starts"),
        ("Gas","Starts"),
        ("Starts","Moves")
    ]
)

# Defining the parameters using CPT
from pgmpy.factors.discrete import TabularCPD

cpd_battery = TabularCPD(
    variable="Battery", variable_card=2, values=[[0.70], [0.30]],
    state_names={"Battery":['Works',"Doesn't work"]},
)

cpd_gas = TabularCPD(
    variable="Gas", variable_card=2, values=[[0.40], [0.60]],
    state_names={"Gas":['Full',"Empty"]},
)

cpd_radio = TabularCPD(
    variable=  "Radio", variable_card=2,
    values=[[0.75, 0.01],[0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Radio": ["turns on", "Doesn't turn on"],
                 "Battery": ['Works',"Doesn't work"]}
)

cpd_ignition = TabularCPD(
    variable=  "Ignition", variable_card=2,
    values=[[0.75, 0.01],[0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Ignition": ["Works", "Doesn't work"],
                 "Battery": ['Works',"Doesn't work"]}
)

cpd_starts = TabularCPD(
    variable="Starts",
    variable_card=2,
    values=[[0.95, 0.05, 0.05, 0.001], [0.05, 0.95, 0.95, 0.9999]],
    evidence=["Ignition", "Gas"],
    evidence_card=[2, 2],
    state_names={"Starts":['yes','no'], "Ignition":["Works", "Doesn't work"], "Gas":['Full',"Empty"]},
)

cpd_moves = TabularCPD(
    variable="Moves", variable_card=2,
    values=[[0.8, 0.01],[0.2, 0.99]],
    evidence=["Starts"],
    evidence_card=[2],
    state_names={"Moves": ["yes", "no"],
                 "Starts": ['yes', 'no'] }
)


# Associating the parameters with the model structure
car_model.add_cpds( cpd_starts, cpd_ignition, cpd_gas, cpd_radio, cpd_battery, cpd_moves)

car_infer = VariableElimination(car_model)

print(car_infer.query(variables=["Moves"],evidence={"Radio":"turns on", "Starts":"yes"}))

def main():
    print("\nProblem 2 pt. 2 queries\n")
    # Given that the car will not move, what is the probability that the battery is not working?
    q = car_infer.query(variables=["Battery"], evidence={"Moves": "no"})
    print(q)
    print("Probability that the battery is not working given the car will not move:", q.values[1])
    # Given that the radio is not working, what is the probability that the car will not start?
    q2 = car_infer.query(variables=["Starts"], evidence={"Radio": "Doesn't turn on"})
    print(q2)
    print("Probability that the car will not start given the radio is not working:", q2.values[1])
    # Given that the battery is working, does the probability of the radio working change if we discover that the car has gas in it?
    q3_pt1 = car_infer.query(variables=["Radio"], evidence={"Battery": "Works"})
    q3_pt2 = car_infer.query(variables=["Radio"], evidence={"Battery": "Works", "Gas": "Full"})

    print(q3_pt1)
    print(q3_pt2)

    choice = "no" if q3_pt1.values[0] == q3_pt2.values[0] else "yes"
    print(f"Given the battery is working, does the probability of the radio working change if we discover that the car has gas in it: {choice}")
    # Given that the car doesn't move, how does the probability of the ignition failing change if we observe that the car dies not have gas in it?
    q4_pt1 = car_infer.query(variables=["Ignition"], evidence={"Moves": "no"})
    q4_pt2 = car_infer.query(variables=["Ignition"], evidence={"Moves": "no", "Gas": "Empty"})

    p1 = q4_pt1.values[1]
    p2 = q4_pt2.values[1]

    if p1 < p2:
        choice = "the probability of the ignition failing increases"
    elif p1 > p2:
        choice = "the probability of the ignition failing decreases"
    else:
        choice = "the probability of the ignition failing doesn't change"

    print(q4_pt1)
    print(q4_pt2)
    print(f"Given that the car doesn't move, {choice} if we observe that the car dies without having gas in it")
    # What is the probability that the car starts if the radio works and it has gas in it?
    q5 = car_infer.query(variables=["Starts"], evidence={"Radio": "turns on", "Gas": "Full"})
    print(q5)
    print("The probability that the car starts if the radio works and it has gas in it:", q5.values[0])

if __name__ == "__main__":
    main()


