import pickle
import numpy as np
from math import ceil
from tkinter import *
from tkinter import messagebox


# common functions
def supplies_and_demands(os, ds, soln_arr):

    for o in range(os):
        soln_arr[o, -1] = int(input(f"Supply of origin {o + 1}: "))
    for d in range(ds):
        soln_arr[-1, d] = int(input(f"Demand of destination {d + 1}: "))
    if np.sum(soln_arr[-1, :]) != np.sum(soln_arr[:, -1]):
        print("\nTotal supply and total demand must be EQUAL! Enter again...")
        return supplies_and_demands(os, ds, soln_arr)
    return soln_arr


def round_up(num):
    return int(ceil(num / 100.0)) * 100


def penalty(costs):
    if len(costs) == 1:
        return costs[0]
    else:
        smallest = np.min(costs)
        loc = np.where(costs == np.min(costs))
        second_smallest = np.min(np.delete(costs, loc[0]))
        return second_smallest - smallest


# Johnson Method ----------------------------------------------------------------------------------------------------------------------------------
def johnson_algorithm():



    #sv.set("")

    print("***** Johnson's Algorithm *****")
    jobs = int(input("\nEnter number of jobs: "))
    machines = int(input("Enter number of machines: "))
    mt_data = np.zeros((jobs, machines), dtype='int64')
    mod_mt_data = np.zeros((jobs, 2), dtype='int64')
    sequence = np.zeros((jobs,), dtype='int64')

    print()
    for job in range(jobs):
        print(f"Enter machining time for job {job + 1} in -")
        for machine in range(machines):
            mt_data[job, machine] = int(input(f"machine {machine + 1}: "))
    print(f"\nTable of machining times --> \n{mt_data}")

    for job in range(jobs):
        mod_mt_data[job, 0] = np.sum(mt_data[job, :]) - mt_data[job, -1]
        mod_mt_data[job, 1] = np.sum(mt_data[job, :]) - mt_data[job, 0]
    # print(f"Modified data table --> \n{mod_mt_data}\n")

    pos_seq_idx = 0
    neg_seq_idx = -1

    for each in range(jobs):
        row_loc, col_loc = np.where(mod_mt_data == np.min(mod_mt_data))
        if len(row_loc) == 1:
            if col_loc[0] == 0:
                sequence[pos_seq_idx] = row_loc[0] + 1
                pos_seq_idx += 1
            else:
                sequence[neg_seq_idx] = row_loc[0] + 1
                neg_seq_idx -= 1
            mod_mt_data[row_loc[0], :] = [np.max(mod_mt_data) + 1, np.max(mod_mt_data) + 1]
            # print(mod_mt_data, pos_seq_idx, neg_seq_idx, sequence)
        else:
            temp = np.zeros((len(row_loc), 2), dtype='int64')
            for i in range(len(row_loc)):
                temp[i, :] = [np.sum(mod_mt_data[row_loc[i]]), row_loc[i]]
            # print(f"Temp: {temp}")
            target_row_row_loc, target_row_col_loc = np.where(temp == np.min(temp[:, 0]))
            # print(f"target_row_row_loc, target_row_col_loc:\t", target_row_row_loc, target_row_col_loc)
            target_row = temp[target_row_row_loc[0], 1]
            # print(f"target row: {target_row}")
            if mod_mt_data[target_row, 0] == np.min(mod_mt_data):
                sequence[pos_seq_idx] = target_row + 1
                pos_seq_idx += 1
            else:
                sequence[neg_seq_idx] = target_row + 1
                neg_seq_idx -= 1
            mod_mt_data[target_row, :] = [np.max(mod_mt_data) + 1, np.max(mod_mt_data) + 1]
            # print(mod_mt_data, pos_seq_idx, neg_seq_idx, sequence)

    print(f"\nOptimal Job Sequence: {sequence}")
    sv.set("Johnson's Algorithm")
    Label(f8, text=f"Optimal Job Sequence: {sequence}", font=('OpenSans-Regular', 12), fg='white', bg='#222831').pack(anchor="nw")
    print("----------------------------------------------------------------------------\n\n")


# SAI Method -------------------------------------------------------------------------------------------------------------------------------
def sai_algorithm():
    print("***** SAI Algorithm *****")
    jobs = int(input("\nEnter number of jobs: "))
    machines = int(input("Enter number of machines: "))
    data = np.zeros((jobs, machines), dtype='int32')
    sequence = np.array([], dtype='int32')

    print()
    for job in range(jobs):
        print(f"Enter machining time for job {job + 1} in -")
        for machine in range(machines):
            data[job, machine] = int(input(f"machine {machine + 1}: "))
    print(f"\nTable of machining times --> \n{data}")

    row_mins = np.min(data, axis=1)

    for k in range(len(data)):
        temp_row_mins = np.min(data, axis=1)
        temp_col_mins = np.min(data, axis=0)

        matches = np.array([], dtype='int32')
        for idx in range(len(temp_row_mins)):
            if temp_row_mins[idx] in temp_col_mins:
                if temp_row_mins[idx] in matches:
                    continue
                else:
                    matches = np.append(matches, temp_row_mins[idx])

        # tracking location for optimal sequence
        loc_for_seq = np.where(row_mins == np.min(matches))
        if len(loc_for_seq[0]) > 1:
            for m in range(len(loc_for_seq[0])):
                if loc_for_seq[0][m] + 1 in sequence:
                    continue
                else:
                    sequence = np.append(sequence, loc_for_seq[0][m] + 1)
                    break
        else:
            sequence = np.append(sequence, loc_for_seq[0][0] + 1)

        # tracking location for mt_data update
        loc_for_update = np.where(temp_row_mins == np.min(matches))
        data = np.delete(data, loc_for_update[0][0], 0)

    print(f"\nOptimal Job Sequence: {sequence}")
    sv.set('SAI Algorithm')
    # Label(f8, text=f'Optimal solution of Job sequencing by SAI method:',fg='white',bg="black", font=('Helvetica', 16, 'bold italic')
    #       ).pack()
    Label(f8, text=f"Optimal Job Sequence: {sequence}", fg='white', font=('OpenSans-Regular', 12), bg='#222831').pack(anchor='nw')
    # e.set(f"JS:{h}")
    print("----------------------------------------------------------------------------\n\n")


# Simplex Method --------------------------------------------------------------------------------------------------------------------------------
def simplex_method():
    def max_min_user_input():
        user_input = int(input('''\nEnter "1 to Maximize" or "2 to Minimize" the problem: '''))
        if user_input == 1 or user_input == 2:
            return user_input
        else:
            return max_min_user_input()

    print("***** Simplex Method *****")
    no_of_vars = int(input("\nEnter number of variables: "))
    no_of_constraints = int(input("Enter number of constraints: "))

    reference = {0: "z"}
    for rv in range(no_of_vars):
        reference[rv + 1] = f"Variable {rv + 1}"
    for rc in range(no_of_constraints):
        reference[no_of_vars + rc + 1] = f"Resource {rc + 1}"
    # print(reference)

    arr = np.zeros((no_of_constraints + 1, no_of_vars + no_of_constraints + 3), dtype=float)
    arr[0, 0] = 1
    print("\nEnter the coefficients in objective Function-")
    for nv in range(no_of_vars):
        arr[0, nv + 1] = -1 * float(input(f"of variable {nv + 1}: "))
    for nc in range(no_of_constraints):
        print(f"Enter the coefficients in Constraint {nc + 1}-")
        for inner_nc in range(no_of_vars):
            arr[nc + 1, inner_nc + 1] = float(input(f"of variable {inner_nc + 1}: "))
        arr[nc + 1, -2] = float(input(f"RHS of constraint {nc + 1}: "))
        arr[nc + 1, no_of_vars + nc + 1] = 1

    print(f"\nInitial simplex table:\n{arr}")
    basic_vars = np.arange(no_of_vars + 1, no_of_vars + no_of_constraints + 1)
    basic_vars = np.insert(basic_vars, 0, 0)

    if max_min_user_input() == 1:
        while (arr[0, 1:-2] < 0).any():
            pivot_col = arr[0, 1:-2].argmin() + 1
            for i in range(len(arr)):
                if i == 0: continue
                if arr[i, pivot_col] <= 0:
                    arr[i, -1] = np.inf
                else:
                    arr[i, -1] = arr[i, -2] / arr[i, pivot_col]
            pivot_row = arr[1:, -1].argmin() + 1
            basic_vars[pivot_row] = pivot_col
            # print(f"Pivot column: {pivot_col}\nPivot row: {pivot_row}\nBasic vars: {basic_vars}")
            arr[pivot_row, :-1] = arr[pivot_row, :-1] / arr[pivot_row, pivot_col]
            for i in range(len(arr)):
                if i == pivot_row: continue
                arr[i, :-1] = arr[i, :-1] + (arr[pivot_row, :-1] * (-arr[i, pivot_col]))
            # print(arr)
        max_or_min = "Maximization"
        Label(f8, text=f'Solution of the Maximization Problem', font=('OpenSans-Regular', 12, 'bold'), fg='white',
              bg='#222831').pack()
    else:
        while (arr[0, 1:-2] > 0).any():
            pivot_col = arr[0, 1:-2].argmax() + 1
            for i in range(len(arr)):
                if i == 0: continue
                if arr[i, pivot_col] <= 0:
                    arr[i, -1] = np.inf
                else:
                    arr[i, -1] = arr[i, -2] / arr[i, pivot_col]
            pivot_row = arr[1:, -1].argmin() + 1
            basic_vars[pivot_row] = pivot_col
            # print(f"Pivot column: {pivot_col}\nPivot row: {pivot_row}\nBasic vars: {basic_vars}")
            arr[pivot_row, :-1] = arr[pivot_row, :-1] / arr[pivot_row, pivot_col]
            for i in range(len(arr)):
                if i == pivot_row: continue
                arr[i, :-1] = arr[i, :-1] + (arr[pivot_row, :-1] * (-arr[i, pivot_col]))
            # print(arr)
        max_or_min = "Minimization"
        Label(f8, text=f'Solution of the Minimization Problem', font=('OpenSans-Regular', 12, 'bold'), fg='white',
              bg='#222831').pack()

    sv.set("Simplex Method")
    print(f"\n-----> Solution of the {max_or_min} Problem")
    for (k, v) in reference.items():
        var_val = 0
        if k in basic_vars: var_val = float(arr[np.where(basic_vars == k)[0], -2])
        print(f"Value of {v}: {round(var_val, 4)} unit(s)")
        Label(f8, text=f"Value of {v}: {round(var_val, 4)} unit(s)", font=('OpenSans-Regular', 12), fg='white', bg='#222831').pack(anchor='nw')
    print("----------------------------------------------------------------------------\n\n")


# North West Corner Method ---------------------------------------------------------------------------------------------------------------------
def north_west_corner_method():
    print("***** North West Corner Method *****\n")
    origins_nwc = int(input(f"Enter number of origins: "))
    destinations_nwc = int(input(f"Enter number of destinations: "))
    soln_nwc = np.zeros((origins_nwc + 1, destinations_nwc + 1), dtype='int64')
    solution_nwc = supplies_and_demands(origins_nwc, destinations_nwc, soln_nwc)

    trans_costs_fixed = np.zeros((origins_nwc, destinations_nwc), dtype='int64')
    for o in range(origins_nwc):
        print(f"Transportation cost(s) for delivery from origin {o + 1} to-")
        for d in range(destinations_nwc): trans_costs_fixed[o, d] = int(input(f"Destination {d + 1}: "))
    print(f"\nGiven transportation costs table --> \n{trans_costs_fixed}\n")

    solution_indices_nwc = []
    r, c = 0, 0
    for itr in range(origins_nwc + destinations_nwc - 1):
        solution_indices_nwc.append((r, c))
        if solution_nwc[-1, c] <= solution_nwc[r, -1]:
            solution_nwc[r, c] = solution_nwc[-1, c]
            solution_nwc[r, -1] -= solution_nwc[-1, c]
            c += 1
        else:
            solution_nwc[r, c] = solution_nwc[r, -1]
            solution_nwc[-1, c] -= solution_nwc[r, -1]
            r += 1
        # print(f"{trans_costs_mcm}\t--->Trans cost")
        # print(f"{solution_nwc}\t--->solution_nwc")

    total_trans_cost_nwc = 0
    sv.set("North West Corner Method")
    print("-----> Solution")
    for each in sorted(solution_indices_nwc):
        print(f"Transport {solution_nwc[each]} unit(s) of product from origin {each[0] + 1} to destination {each[1] + 1}")
        Label(f8, text=f"Transport {solution_nwc[each]} unit(s) of product from origin {each[0] + 1} to destination {each[1] + 1}", font=('OpenSans-Regular', 12), fg='white', bg='#222831').pack(
            anchor='nw')
        total_trans_cost_nwc += trans_costs_fixed[each] * solution_nwc[each]

    print(f"Total transportation cost: {total_trans_cost_nwc} unit")
    Label(f8, text=f"Total transportation cost: {total_trans_cost_nwc} unit\n", font=('OpenSans-Regular', 12), bg='#222831', fg='white').pack(anchor='nw')
    print("----------------------------------------------------------------------------\n\n")


# Minimum Cost Method ----------------------------------------------------------------------------------------------------------------------
def minimum_cost_method():
    print("***** Minimum Cost Method *****\n")
    origins_mcm = int(input(f"Enter number of origins: "))
    destinations_mcm = int(input(f"Enter number of destinations: "))
    soln_mcm = np.zeros((origins_mcm + 1, destinations_mcm + 1), dtype='int64')
    solution_mcm = supplies_and_demands(origins_mcm, destinations_mcm, soln_mcm)
    trans_costs_fixed_mcm = np.zeros((origins_mcm, destinations_mcm), dtype='int64')

    for origin in range(origins_mcm):
        print(f"Transportation cost(s) for delivery from origin {origin + 1} to-")
        for destination in range(destinations_mcm): trans_costs_fixed_mcm[origin, destination] = int(input(f"Destination {destination + 1}: "))
    print(f"\nGiven data table --> \n{trans_costs_fixed_mcm}\n")
    blocked_val_mcm = round_up(np.max(trans_costs_fixed_mcm))
    trans_costs_mcm = trans_costs_fixed_mcm.copy()
    solution_indices_mcm = []

    for itr in range(origins_mcm + destinations_mcm - 1):
        min_cost_idx_mcm = np.where(trans_costs_mcm == trans_costs_mcm.min())
        # print(min_cost_idx_mcm)
        min_cost_row_idx_mcm, min_cost_col_idx_mcm = min_cost_idx_mcm[0][0], min_cost_idx_mcm[1][0]
        if solution_mcm[min_cost_row_idx_mcm, -1] <= solution_mcm[-1, min_cost_col_idx_mcm]:
            solution_mcm[min_cost_row_idx_mcm, min_cost_col_idx_mcm] = solution_mcm[min_cost_row_idx_mcm, -1]
            solution_mcm[-1, min_cost_col_idx_mcm] -= solution_mcm[min_cost_row_idx_mcm, -1]
            if itr < origins_mcm + destinations_mcm - 3:
                solution_mcm[min_cost_row_idx_mcm, -1] = trans_costs_mcm[min_cost_row_idx_mcm, :] = blocked_val_mcm
            else:
                solution_mcm[min_cost_row_idx_mcm, -1] = trans_costs_mcm[min_cost_row_idx_mcm, min_cost_col_idx_mcm] = blocked_val_mcm
        else:
            solution_mcm[min_cost_row_idx_mcm, min_cost_col_idx_mcm] = solution_mcm[-1, min_cost_col_idx_mcm]
            solution_mcm[min_cost_row_idx_mcm, -1] -= solution_mcm[-1, min_cost_col_idx_mcm]
            if itr < origins_mcm + destinations_mcm - 3:
                solution_mcm[-1, min_cost_col_idx_mcm] = trans_costs_mcm[:, min_cost_col_idx_mcm] = blocked_val_mcm
            else:
                solution_mcm[-1, min_cost_col_idx_mcm] = trans_costs_mcm[min_cost_row_idx_mcm, min_cost_col_idx_mcm] = blocked_val_mcm
        solution_indices_mcm.append((min_cost_row_idx_mcm, min_cost_col_idx_mcm))
        # print(f"{trans_costs_mcm}\t--->Trans cost")

    total_trans_cost_mcm = 0
    sv.set("Minimum Cost Method")
    print("-----> Solution")
    for each in sorted(solution_indices_mcm):
        print(f"Transport {solution_mcm[each]} unit(s) of product from origin {each[0] + 1} to destination {each[1] + 1}")
        total_trans_cost_mcm += trans_costs_fixed_mcm[each] * solution_mcm[each]
        Label(f8, text=f"Transport {solution_mcm[each]} unit(s) of product from origin {each[0] + 1} to destination {each[1] + 1}", font=('OpenSans-Regular', 12), fg='white', bg='#222831').pack(
            anchor="nw")

    print(f"Total transportation cost: {total_trans_cost_mcm} unit")
    Label(f8, text=f"Total transportation cost: {total_trans_cost_mcm} unit\n", font=('OpenSans-Regular', 12), fg='white', bg="#222831").pack(anchor="nw")
    print("----------------------------------------------------------------------------\n\n")


# Vogel's Approximation Method -----------------------------------------------------------------------------------------------------
def vogel_approximation_method():
    print("***** Vogel's Approximation Method *****\n")
    origins = int(input(f"Enter number of origins: "))
    destinations = int(input(f"Enter number of destinations: "))
    solution_vam = np.zeros((origins + 2, destinations + 2), dtype='int64')
    solution_indices_vam = []

    def supplies_and_demands_vam():
        for o in range(origins):
            solution_vam[o, -2] = int(input(f"Supply of origin {o + 1}: "))
        for d in range(destinations):
            solution_vam[-2, d] = int(input(f"Demand of destination {d + 1}: "))
        if np.sum(solution_vam[-2, :]) != np.sum(solution_vam[:, -2]):
            print("\nTotal supply and total demand must be EQUAL! Enter again...")
            return supplies_and_demands_vam()

    supplies_and_demands_vam()
    trans_costs_fixed = np.zeros((origins, destinations), dtype='int64')
    for i in range(origins):
        print(f"Transportation cost(s) for delivery from origin {i + 1} to-")
        for j in range(destinations): trans_costs_fixed[i, j] = int(input(f"Destination {j + 1}: "))
    blocked_val = round_up(np.max(trans_costs_fixed))
    trans_costs_vam = trans_costs_fixed.copy()

    penalty_rows, penalty_cols = list(range(origins)), list(range(destinations))
    penalty_row_indices, penalty_col_indices = [], []
    penalty_row_indices_check, penalty_col_indices_check = [], []
    # print(f"penalty rows: {penalty_rows}\n penalty cols: {penalty_cols}")
    for i in range(origins + destinations - 1):
        # print(f"\nSTART of iteration: {origin} -----------------------")
        for pr in penalty_rows:
            li = np.array([], dtype='int64')
            for pc in penalty_cols:
                li = np.append(li, trans_costs_fixed[pr, pc])
            solution_vam[pr, -1] = penalty(li)
            penalty_row_indices.append(pr)
            penalty_col_indices.append(-1)
        for pc in penalty_cols:
            li = np.array([], dtype='int64')
            for pr in penalty_rows:
                li = np.append(li, trans_costs_fixed[pr, pc])
            solution_vam[-1, pc] = penalty(li)
            penalty_row_indices.append(-1)
            penalty_col_indices.append(pc)
        # print(f"{solution_nwc}\t--->SOLUTION\nPenalty row indices: {penalty_row_indices}\nPenalty col indices: {penalty_col_indices}")
        if i == 0:
            penalty_row_indices_check = penalty_row_indices
            penalty_col_indices_check = penalty_col_indices

        # Find max penalty and max penalty index
        max_penalty_row_idx, max_penalty_col_idx = None, None
        max_penalty = np.max(solution_vam[penalty_row_indices, penalty_col_indices])
        for zz in penalty_row_indices_check:
            if solution_vam[zz, -1] == max_penalty:
                max_penalty_row_idx = zz
                max_penalty_col_idx = destinations + 1
        if not max_penalty_row_idx:
            for zz in penalty_col_indices_check:
                if solution_vam[-1, zz] == max_penalty:
                    max_penalty_row_idx = origins + 1
                    max_penalty_col_idx = zz
        # print(f"Max penalty: {max_penalty}\nMax pen idx: {max_penalty_row_idx, max_penalty_col_idx}")

        # Find min cost idx
        min_cost_row_idx_vam, min_cost_col_idx_vam = None, None
        non_blocked_indices = np.where(trans_costs_vam != blocked_val)
        blocked_r_indices, blocked_c_indices = np.where(trans_costs_vam == blocked_val)
        blocked_indices = list(zip(blocked_r_indices, blocked_c_indices))
        if max_penalty_row_idx == origins + 1:
            check_row_indices = []
            # print(f"non_blocked_indices: {non_blocked_indices}")
            for aa in range(len(non_blocked_indices[1])):
                if non_blocked_indices[1][aa] == max_penalty_col_idx:
                    check_row_indices.append(non_blocked_indices[0][aa])
            min_cost_vam = np.min(trans_costs_vam[check_row_indices, max_penalty_col_idx])
            # print(f"min_cost_vam from origins+1: {min_cost_vam}")
            min_cost_idx_vam = np.where(trans_costs_vam[:, max_penalty_col_idx] == min_cost_vam)
            # print(f"min_cost_idx_vam from origins+1: {min_cost_idx_vam}")
            # print(f"Blocked indices from origins+1: {blocked_indices}")
            for xxx in min_cost_idx_vam[0]:
                if (xxx, max_penalty_col_idx) in blocked_indices:
                    continue
                else:
                    min_cost_row_idx_vam, min_cost_col_idx_vam = xxx, max_penalty_col_idx
                    break
        if max_penalty_col_idx == destinations + 1:
            check_col_indices = []
            for aa in range(len(non_blocked_indices[0])):
                if non_blocked_indices[0][aa] == max_penalty_row_idx:
                    check_col_indices.append(non_blocked_indices[1][aa])
            min_cost_vam = np.min(trans_costs_vam[max_penalty_row_idx, check_col_indices])
            # print(f"min_cost_vam from destinations+1: {min_cost_vam}")
            min_cost_idx_vam = np.where(trans_costs_vam[max_penalty_row_idx, :] == min_cost_vam)
            # print(f"min_cost_idx_vam from destinations+1: {min_cost_idx_vam}")
            # print(f"Blocked indices from destinations+1: {blocked_indices}")
            for xxx in min_cost_idx_vam[0]:
                if (max_penalty_row_idx, xxx) in blocked_indices:
                    continue
                else:
                    min_cost_row_idx_vam, min_cost_col_idx_vam = max_penalty_row_idx, xxx
                    break
        # print(f"Min cost index: {min_cost_row_idx_vam, min_cost_col_idx_vam}")

        # find one solution_nwc at a time
        if solution_vam[-2, min_cost_col_idx_vam] <= solution_vam[min_cost_row_idx_vam, -2]:
            # print("row penalty is less")
            solution_indices_vam.append((min_cost_row_idx_vam, min_cost_col_idx_vam))
            solution_vam[min_cost_row_idx_vam, min_cost_col_idx_vam] = solution_vam[-2, min_cost_col_idx_vam]
            solution_vam[min_cost_row_idx_vam, -2] -= solution_vam[-2, min_cost_col_idx_vam]
            solution_vam[-2:, min_cost_col_idx_vam] = trans_costs_vam[:, min_cost_col_idx_vam] = blocked_val
            if len(penalty_cols) > 0: penalty_cols.remove(min_cost_col_idx_vam)
        else:
            # print("col penalty is less")
            solution_indices_vam.append((min_cost_row_idx_vam, min_cost_col_idx_vam))
            solution_vam[min_cost_row_idx_vam, min_cost_col_idx_vam] = solution_vam[min_cost_row_idx_vam, -2]
            solution_vam[-2, min_cost_col_idx_vam] -= solution_vam[min_cost_row_idx_vam, -2]
            solution_vam[min_cost_row_idx_vam, -2:] = trans_costs_vam[min_cost_row_idx_vam, :] = blocked_val
            if len(penalty_rows) > 0: penalty_rows.remove(min_cost_row_idx_vam)
        # print(f"{solution_nwc}\t--->SOLUTION\npenalty rows: {penalty_rows}\n penalty cols: {penalty_cols}")
        penalty_row_indices, penalty_col_indices = [], []
        # print(f"STOP of iteration: {origin} -----------------------")

    print(f"\n-----> Solution")
    total_trans_cost_vam = 0
    sv.set("Vogel's Approximation Method")
    for each in sorted(solution_indices_vam):
        print(f"Transport {solution_vam[each]} unit(s) of product from origin {each[0] + 1} to destination {each[1] + 1}")
        Label(f8, text=f"Transport {solution_vam[each]} unit(s) of product from origin {each[0] + 1} to destination {each[1] + 1}", font=('OpenSans-Regular', 12), fg='white', bg="#222831").pack(
            anchor="nw")
        total_trans_cost_vam += trans_costs_fixed[each] * solution_vam[each]

    print(f"Total transportation cost: {total_trans_cost_vam} unit")
    Label(f8, text=f"Total transportation cost: {total_trans_cost_vam} unit\n", font=('OpenSans-Regular', 12),
          fg='white', bg="#222831").pack(anchor="nw")
    print("----------------------------------------------------------------------------\n\n")


# tkinter codes
def resetAll():
    for widgets in f8.winfo_children():
        widgets.destroy()

    sv.set("")


def print_func():
    messagebox.showinfo('Attention!!', 'First connect with an printer, carefully.')


def About_func():
    messagebox.showinfo('About', "Trying to include more algorithms ASAP.\n\nProgramming Language: Python\nModules used:\n  1. Numpy\n  2. Math\n  3. Tkinter\n  4. Pickle\n\nDevelopers - Md Shamim Hasan & Ahamed Al Hassan Sunny")


def feedback_func():
    messagebox.showinfo("Feedback", "If you have any suggestion please let us know. Your opinion matters to us.\n\nEmail: shihab18015@gmail.com\n           aahmedsunny115@gmail.com")


root = Tk()
root.geometry("900x600")
root.configure(bg="#222831")
# root.maxsize(600,425)
root.title('Simple OR Project')
# C = Canvas(root, bg="blue", height=250, width=300)
# root.configure(background="Black")

canvas = Canvas(
    root,
    bg="#222831",
    height=600,
    width=1000,
    bd=0,
    highlightthickness=0,
    relief="ridge")
canvas.place(x=0, y=0)

canvas.create_rectangle(
    0, 0, 0 + 331, 0 + 600,
    fill="#30475e",
    outline="")

canvas.create_text(
    165.0, 235.5,
    text="Hey!\n\nWelcome to a Simple\nOperations Research (OR)\nProject. Using this you\ncan solve various types\nof problems related to OR.\nHope you find it useful! \n\nThank you.",
    fill="#dddddd",
    font=("Montserrat-MediumItalic", int(14.0)))

canvas.create_text(
    186.5, 543.0,
    text="Developed by\nMd. Shamim Hasan\nAhamed Al Hassan Sunny",
    fill="#dddddd",
    font=("Lato-MediumItalic", int(10.0)))

canvas.create_rectangle(
    331, 0, 331 + 669, 0 + 600,
    fill="#222831",
    outline="")

sv = StringVar()
sv.set("")

#command=lambda:[funcA(), funcB(), funcC()]

f2 = Frame(root, bg='#222831', borderwidth=2)
f2.place(x=340, y=10, width=550,
         height=40)
p = Label(f2, textvariable=f'{sv}', font=('Helvetica', 20, 'bold'), fg='white', bg='#222831')
p.pack()

f8 = Frame(canvas, bg='#222831', relief=SUNKEN)
f8.place(x=340, y=145,
         width=550,
         height=400)
f8.pack_propagate(0)
# scb=Scrollbar(f8)
# scb.pack(fill=Y,side=RIGHT)
mnb = Menu(root)

m4 = Menu(mnb, tearoff=0)
m4.add_command(label='Print', command=print_func)
m4.add_command(label='About', command=About_func)
m4.add_command(label='Feedback', command=feedback_func)
mnb.add_cascade(label='File', menu=m4)
root.config(menu=mnb)

m1 = Menu(mnb, tearoff=0)
m1.add_command(label="Johnson's Algorithm", command=lambda:[resetAll(),johnson_algorithm()])
m1.add_command(label='SAI Algorithm', command=lambda:[resetAll(),sai_algorithm()])
mnb.add_cascade(label='Job Sequencing', menu=m1)
root.config(menu=mnb)

m2 = Menu(mnb, tearoff=0)
m2.add_command(label='Simplex Method', command=lambda :[resetAll(),simplex_method()])
mnb.add_cascade(label='Linear Programming', menu=m2)  # file er vitor rakhar jonno
root.config(menu=mnb)

m3 = Menu(mnb, tearoff=0)
m3.add_command(label='North West Corner Method', command=lambda :[resetAll(),north_west_corner_method()])
m3.add_command(label='Minimum Cost Method', command=lambda :[resetAll(),minimum_cost_method()])
m3.add_command(label="Vogel's Approximation Method", command=lambda :[resetAll(),vogel_approximation_method()])
mnb.add_cascade(label='Transportation', menu=m3)  # file er vitor rakhar jonno
root.config(menu=mnb)

Button(root, text="Clear", font=('Helvetica bold', 12), bg="#646464", fg="#ffffff", activebackground="#4B4848", activeforeground="#ffffff", borderwidth=0, highlightthickness=0, relief="flat",
       command=resetAll).place(x=550, y=545, width=126, height=44)

root.resizable(False, False)
root.mainloop()
