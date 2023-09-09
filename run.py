import os

# Default parameter
for seed in range(0, 10):
    cmd = 'python main.py --seed {}'.format(seed)
    print(cmd)
    os.system(cmd)


# Parameter analysis K
for k in [5, 15, 20, 25, 30]:
    cmd = 'python main.py --k {} --pretrain True'.format(k)
    print(cmd)
    os.system(cmd)
    for seed in range(0, 10):
        cmd = 'python main.py --seed {} --k {}'.format(seed, k)
        print(cmd)
        os.system(cmd)


# Parameter analysis alpha
for alpha in [0.001, 0.01, 1, 10]:
    cmd = 'python main.py --alpha_value {} --pretrain True'.format(alpha)
    print(cmd)
    os.system(cmd)
    for seed in range(0, 10):
        cmd = 'python main.py --seed {} --alpha_value {}'.format(seed, alpha)
        print(cmd)
        os.system(cmd)


# Parameter analysis lambda1, lambda2
for lambda1 in [0.01, 0.1, 1, 10, 100]:
    for lambda2 in [0.01, 0.1, 1, 10, 100]:
        cmd = 'python main.py --lambda1 {} --lambda2 {} --pretrain True'.format(lambda1, lambda2)
        print(cmd)
        os.system(cmd)
        for seed in range(0, 10):
            cmd = 'python main.py --seed {} --lambda1 {} --lambda2 {}'.format(seed, lambda1, lambda2)
            print(cmd)
            os.system(cmd)


# Parameter analysis lambda3
cmd = 'python main.py --pretrain True'
print(cmd)
os.system(cmd)
for lambda3 in [5, 15, 20, 25]:
    for seed in range(0, 10):
        cmd = 'python main.py --seed {} --lambda3 {}'.format(seed, lambda3)
        print(cmd)
        os.system(cmd) 


# Abaltion w/o DRR
"""
step1: remove the label_contrastive_module of scDCRN.py
"""
cmd = 'python main.py --lambda1 0 --lambda2 0 --pretrain True'
print(cmd)
os.system(cmd)
for seed in range(0, 10):
    cmd = 'python main.py --seed {} --lambda1 0 --lambda2 0'.format(seed)
    print(cmd)
    os.system(cmd)


# Abaltion w/o C
cmd = 'python main.py --lambda1 0 --pretrain True'
print(cmd)
os.system(cmd)
for seed in range(0, 10):
    cmd = 'python main.py --seed {} --lambda1 0'.format(seed)
    print(cmd)
    os.system(cmd)


# Abaltion w/o F
"""
step1: remove the label_contrastive_module of scDCRN.py
"""
cmd = 'python main.py --lambda2 0 --pretrain True'
print(cmd)
os.system(cmd)
for seed in range(0, 10):
    cmd = 'python main.py --seed {} --lambda2 0'.format(seed)
    print(cmd)
    os.system(cmd)


# Ablation w/o Multiple
"""
step1: replace the loss of utils.py::distribution_loss 
with loss = F.kl_div(Q[0].log(), P, reduction='batchmean')
"""
cmd = 'python main.py --pretrain True'
print(cmd)
os.system(cmd)
for seed in range(0, 10):
    cmd = 'python main.py --seed {}'.format(seed)
    print(cmd)
    os.system(cmd)


