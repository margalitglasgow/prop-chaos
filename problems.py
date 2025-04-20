import utils
import torch
import numpy as np

# Some teacher neurons that are useful in code below
e1 = np.zeros(2)
e1[0] = 1
e2 = np.zeros(2)
e2[1] = 1
ediag = np.zeros(2)
ediag[:2]= [1/np.sqrt(2), 1/np.sqrt(2)]
ediag2 = np.zeros(2)
ediag2[:2]= [-1/np.sqrt(2), 1/np.sqrt(2)]

# Set up problem instances for SIM Problems.
teachers = torch.tensor([e1]).float()

PP = {}

problem_name = "He3"
k_GT = 3 # Information exponent
k_indices = 1
problem_params_He3 = {"activation": utils.her(k_GT), "label_fun": utils.teacher_fun(teachers, utils.her(k_GT)), "boolean": False, "signed": False, "corr": False, "name": problem_name, "k_GT": k_GT, "k_indices": k_indices}
PP[problem_name] = problem_params_He3

problem_name = "He4"
k_GT = 4 # Information exponent
k_indices = 1
problem_params_He4 = {"activation": utils.her(k_GT), "label_fun": utils.teacher_fun(teachers, utils.her(k_GT)), "boolean": False, "signed": False, "corr": False, "name": problem_name, "k_GT": k_GT, "k_indices": k_indices}
PP[problem_name] = problem_params_He4

problem_name = "He4_small"
k_indices = 1
k_GT = 4 # Information exponent
her_dict_act_small = {4: 1}
her_dict_label_small = {4: 0.5}

poly_dict_act_small = utils.her_dic_to_pol(her_dict_act_small)
activation_small = utils.poly_act_fun(poly_dict_act_small)

poly_dict_label_small = utils.her_dic_to_pol(her_dict_label_small)
label_fun_small = utils.poly_label_fun(poly_dict_label_small)

problem_params_He4_small = {"activation": activation_small, "label_fun": label_fun_small, "boolean": False, "signed": False, "corr": False, "name": problem_name, "k_GT": k_GT, "k_indices": k_indices}
PP[problem_name] = problem_params_He4_small

problem_name = "He4_misspecified"
k_GT = 4 # Information exponent
k_indices = 1
her_dict_act_mis = {4: 1, 6: 1}
her_dict_label_mis = {4: 0.6, 6: 0.3}

poly_dict_act_mis = utils.her_dic_to_pol(her_dict_act_mis)
activation_mis = utils.poly_act_fun(poly_dict_act_mis)

poly_dict_label_mis = utils.her_dic_to_pol(her_dict_label_mis)
label_fun_mis = utils.poly_label_fun(poly_dict_label_mis)
problem_params_He4_misspecified = {"activation": activation_mis, "label_fun": label_fun_mis, "boolean": False, "signed": False, "corr": False, "name": problem_name, "k_GT": k_GT, "k_indices": k_indices}
PP[problem_name] = problem_params_He4_misspecified

problem_name = "Ge4"
k_GT = 4 # Information exponent
k_indices = 1
problem_params_Ge4 = {"activation": utils.ge4_fun(), "label_fun": utils.teacher_fun(teachers, utils.ge4_fun()), "boolean": False, "signed": False, "corr": False, "name": problem_name, "k_GT": k_GT, "k_indices": k_indices}
PP[problem_name] = problem_params_Ge4

problem_name = "NP_mis"
k_GT = 4 # Information exponent
k_indices = 1
problem_params_NP = {"activation": utils.ie4np_fun(), "label_fun": utils.teacher_fun(teachers, utils.ge4_fun()), "boolean": False, "signed": False, "corr": False, "name": problem_name, "k_GT": k_GT, "k_indices": k_indices}
PP[problem_name] = problem_params_NP

problem_name = "He4_orth"
#2 Orthogonal Neurons ---------------
k_GT = 4 # Information exponent
k_indices = 2
teachers_orth = torch.tensor([e1, -e1, e2, -e2]).float()
problem_params_He4_orth = {"activation": utils.her(k_GT), "label_fun": utils.teacher_fun(teachers_orth, utils.her(k_GT)), "boolean": False, "signed": False, "corr": False, "name": problem_name, "k_GT": k_GT, "k_indices": k_indices}
PP[problem_name] = problem_params_He4_orth

problem_name = "He4_nonorth"
#2 NonOrthogonal Neurons ---------------
k_GT = 4 # Information exponent
k_indices = 2
teachers_nonorth = torch.tensor([e1, -e1, ediag, -ediag]).float()
problem_params_He4_nonorth = {"activation": utils.her(k_GT), "label_fun": utils.teacher_fun(teachers_nonorth, utils.her(k_GT)), "boolean": False, "signed": False, "corr": False, "name": problem_name, "k_GT": k_GT, "k_indices": k_indices}
PP[problem_name] = problem_params_He4_nonorth

problem_name = "He4_random_2_8"
# 8 random neurons in 2D
K = 8
k_GT = 4
k_indices = 2
teachers_rand = torch.zeros(size=(K,k_indices))
proj_rand = torch.normal(0,1, size=(K,k_indices))
teachers_rand[:, :k_indices] = proj_rand/torch.norm(proj_rand, dim=1, keepdim=True)
problem_params_He4_rand_2_8 = {"activation": utils.her(k_GT), "label_fun": utils.teacher_fun(teachers_rand, utils.her(k_GT)), "boolean": False, "signed": False, "corr": False, "name": problem_name, "k_GT": k_GT, "k_indices": k_indices}
PP[problem_name] = problem_params_He4_rand_2_8

problem_name = "He4_random_2_16"
# 16 random neurons in 2D
K = 16
k_GT = 4
k_indices = 2
teachers_rand = torch.zeros(size=(K,k_indices))
proj_rand = torch.normal(0,1, size=(K,k_indices))
teachers_rand[:, :k_indices] = proj_rand/torch.norm(proj_rand, dim=1, keepdim=True)
problem_params_He4_rand_2_16 = {"activation": utils.her(k_GT), "label_fun": utils.teacher_fun(teachers_rand, utils.her(k_GT)), "boolean": False, "signed": False, "corr": False, "name": problem_name, "k_GT": k_GT, "k_indices": k_indices}
PP[problem_name] = problem_params_He4_rand_2_16

problem_name = "He4_random_8_8"
# 8 random neurons in 8D
K = 8
k_GT = 4
k_indices = 8
teachers_rand2 = torch.zeros(size=(K,k_indices))
proj_rand2 = torch.normal(0,1, size=(K,k_indices))
teachers_rand2[:, :k_indices] = proj_rand2/torch.norm(proj_rand2, dim=1, keepdim=True)
problem_params_He4_rand_8_8 = {"activation": utils.her(k_GT), "label_fun": utils.teacher_fun(teachers_rand2, utils.her(k_GT)), "boolean": False, "signed": False, "corr": False, "name": problem_name, "k_GT": k_GT, "k_indices": k_indices}
PP[problem_name] = problem_params_He4_rand_8_8

problem_name = "He4_random_16_16"
# 16 random neurons in 16D
K = 16
k_GT = 4
k_indices = 16
teachers_rand2 = torch.zeros(size=(K,k_indices))
proj_rand2 = torch.normal(0,1, size=(K,k_indices))
teachers_rand2[:, :k_indices] = proj_rand2/torch.norm(proj_rand2, dim=1, keepdim=True)
problem_params_He4_rand_16 = {"activation": utils.her(k_GT), "label_fun": utils.teacher_fun(teachers_rand2, utils.her(k_GT)), "boolean": False, "signed": False, "corr": False, "name": problem_name, "k_GT": k_GT, "k_indices": k_indices}
PP[problem_name] = problem_params_He4_rand_16

problem_name = "Man_2"
# Target function is uniform over 2D circle
k_GT = 4
k_indices = 2
problem_params_man_2 = {"activation": utils.her(k_GT), "label_fun": utils.man_fun(k_indices), "boolean": False, "signed": False, "corr": False, "name": problem_name, "k_GT": k_GT, "k_indices": k_indices}
PP[problem_name] = problem_params_man_2

problem_name = "Man_big_2"
# Target function is uniform over 2D circle
k_GT = 4
k_indices = 2
problem_params_man_big_2 = {"activation": utils.her(k_GT), "label_fun": utils.man_big_fun(k_indices), "boolean": False, "signed": False, "corr": False, "name": problem_name, "k_GT": k_GT, "k_indices": k_indices}
PP[problem_name] = problem_params_man_big_2

problem_name = "XOR_2"
# 2-XOR
k_GT = 2
k_indices = 2
problem_params_XOR_2 = {"activation": utils.relu, "label_fun": utils.XOR_fun(k_GT), "boolean": True, "signed": True, "corr": False, "name": problem_name, "k_GT": k_GT, "k_indices": k_indices}
PP[problem_name] = problem_params_XOR_2

problem_name = "XOR_3"
# 3-XOR
k_GT = 3
k_indices = 3
problem_params_XOR_3 = {"activation": utils.relu, "label_fun": utils.XOR_fun(k_GT), "boolean": True, "signed": True, "corr": False, "name": problem_name, "k_GT": k_GT, "k_indices": k_indices}
PP[problem_name] = problem_params_XOR_3

problem_name = "XOR_4"
# 4-XOR
k_GT = 4
k_indices = 4
problem_params_XOR_4 = {"activation": utils.relu, "label_fun": utils.XOR_fun(k_GT), "boolean": True, "signed": True, "corr": False, "name": problem_name, "k_GT": k_GT, "k_indices": k_indices}
PP[problem_name] = problem_params_XOR_4

problem_name = "XOR_4_scaled"
# 4-XOR
k_GT = 4
k_indices = 4
problem_params_XOR_4 = {"activation": utils.relu, "label_fun": utils.XOR_scaled_fun(k_GT), "boolean": True, "signed": True, "corr": False, "name": problem_name, "k_GT": k_GT, "k_indices": k_indices}
PP[problem_name] = problem_params_XOR_4

problem_name = "XOR_5"
# 5-XOR
k_GT = 5
k_indices = 5
problem_params_XOR_5 = {"activation": utils.relu, "label_fun": utils.XOR_fun(k_GT), "boolean": True, "signed": True, "corr": False, "name": problem_name, "k_GT": k_GT, "k_indices": k_indices}
PP[problem_name] = problem_params_XOR_5

problem_name = "Zero"
# Zero Function
k_GT = 0
k_indices = 4
problem_params_zero = {"activation": utils.relu, "label_fun": lambda x: torch.zeros(len(x)), "boolean": False, "signed": True, "corr": False, "name": problem_name, "k_GT": k_GT, "k_indices": k_indices}
PP[problem_name] = problem_params_zero

problem_name = "Hop_4"
# XOR(x1..x4) + XOR(x1...x8) /2
k_GT = 4
k_indices = 8
problem_params_hop_4 = {"activation": utils.relu, "label_fun": utils.hopcase_fun(k_GT), "boolean": True, "signed": True, "corr": False, "name": problem_name, "k_GT": k_GT, "k_indices": k_indices}
PP[problem_name] = problem_params_hop_4

problem_name = "Hop_2"
# XOR(x1..x2) + XOR(x1...x4) /2
k_GT = 2
k_indices = 4
problem_params_hop_2 = {"activation": utils.relu, "label_fun": utils.hopcase_fun(k_GT), "boolean": True, "signed": True, "corr": False, "name": problem_name, "k_GT": k_GT, "k_indices": k_indices}
PP[problem_name] = problem_params_hop_2

problem_name = "Hop_2_4"
# XOR(x1..x2) + XOR(x1...x6) /2
k_GT = 4
k_indices = 6
problem_params_hop_2_4 = {"activation": utils.relu, "label_fun": utils.hop_fun(2, k_GT), "boolean": True, "signed": True, "corr": False, "name": problem_name, "k_GT": k_GT, "k_indices": k_indices}
PP[problem_name] = problem_params_hop_2_4

problem_name = "Hop_1_3"
# x1 + XOR(x1...x4) /2
k_GT = 3
k_indices = 4
problem_params_hop_1_3 = {"activation": utils.relu, "label_fun": utils.hop_fun(1, k_GT), "boolean": True, "signed": True, "corr": False, "name": problem_name, "k_GT": k_GT, "k_indices": k_indices}
PP[problem_name] = problem_params_hop_1_3

problem_name = "stair_3"
# x1 + x1x2 + x1x2x3
k_GT = 3
k_indices = 3
problem_params_stair_3 = {"activation": utils.relu, "label_fun": utils.staircase_fun(k_GT), "boolean": True, "signed": True, "corr": False, "name": problem_name, "k_GT": k_GT, "k_indices": k_indices}
PP[problem_name] = problem_params_stair_3

problem_name = "stair_4"
# x1 + x1x2 + x1x2x3 + x1x2x3x4
k_GT = 4
k_indices = 4
problem_params_stair_4 = {"activation": utils.relu, "label_fun": utils.staircase_fun(k_GT), "boolean": True, "signed": True, "corr": False, "name": problem_name, "k_GT": k_GT, "k_indices": k_indices}
PP[problem_name] = problem_params_stair_4