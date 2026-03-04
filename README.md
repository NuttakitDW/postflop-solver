# RL1 Training + Solve

# training (online, no bt1 data needed)
/opt/anaconda3/bin/python trainings/train_rl1.py config/phase2.json --episodes 50

# solve with trained model
/opt/anaconda3/bin/python trainings/solve_with_rl1.py config/phase2.json models/rl1_phase2/best.pt

# output file path
model = models/rl1_phase2/
flop = data/out/phase2-rl1.flop